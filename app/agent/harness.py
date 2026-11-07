"""The Draco harness behind the HTTP request path.

Every answer request is a Draco workload run through one `Harness`:

    /chat, /chat/stream              -> "app.chat"          (ChatOrchestrator, rag and agent modes)
    /agent/chat, /agent/chat/stream  -> "app.agent.chat"    (AgentRunner, LangGraph supervisor)
    /agent/executions/{id}/resume    -> "app.agent.resume"  (AgentRunner.resume from the durable checkpoint)

so all of them get the same trusted identity (`Identity(subject=user, tenant_id=user)` from the API key),
execution `Policy` (tool allow-list and permissions, checked again on every tool call), budget and
deadline, cancellation, idempotency keys, tenant concurrency cap, audit events and tracing. Inside the
workloads, model calls, tool calls and retrieval go through `app.agent.grounding`, which charges and
events them against the current execution.

Configuration is Draco's own (`DRACO_*`, see `DracoSettings`). Two app defaults differ from Draco's:
the execution deadline is 300 s unless `DRACO_DEFAULT_TIMEOUT_S` is set (thinking-mode artifact runs
take close to a minute on the reference GPU), and without `DRACO_STATE_PATH` executions are kept in a
bounded in-memory store (idempotency replay then covers the newest records of this process only).
"""

import logging
import os
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from draco.adapters.app_agent import agent_resume_workload, agent_runner_workload
from draco.adapters.app_chat import chat_workload
from draco.adapters.app_trace import AppLogExporter
from draco.core.config import DracoSettings
from draco.core.errors import PolicyViolation
from draco.core.identity import Identity
from draco.core.policy import Policy
from draco.core.results import ExecutionResult, Status
from draco.harness import Harness
from draco.harness.runner import scope_key
from draco.state.memory import InMemoryStore

logger = logging.getLogger("rag.draco")

CHAT = "app.chat"
AGENT = "app.agent.chat"
AGENT_RESUME = "app.agent.resume"
APP_DEFAULT_TIMEOUT_S = 300.0
AGENT_PERMISSIONS = frozenset({"read", "write", "generate"})

_harness: Harness | None = None


class BoundedMemoryStore(InMemoryStore):
    """`InMemoryStore` that keeps only the newest `max_executions` records with their events and steps,
    so a long-running API process does not grow without bound. Running executions are never evicted."""

    def __init__(self, max_executions: int = 2000):
        super().__init__()
        self.max_executions = max_executions
        self._order: OrderedDict[str, None] = OrderedDict()

    async def save_execution(self, rec) -> None:
        await super().save_execution(rec)
        self._order[rec.execution_id] = None
        self._order.move_to_end(rec.execution_id)
        for _ in range(len(self._order)):
            if len(self._order) <= self.max_executions:
                break
            oldest = next(iter(self._order))
            if self._exec[oldest].status == "running":
                self._order.move_to_end(oldest)
                continue
            self._evict(oldest)

    def _evict(self, execution_id: str) -> None:
        self._order.pop(execution_id, None)
        self._exec.pop(execution_id, None)
        self._events.pop(execution_id, None)
        for key in [k for k in self._steps if k[0] == execution_id]:
            del self._steps[key]

    async def append_event(self, execution_id: str, name: str, ts: float, data: dict[str, Any]) -> None:
        if execution_id in self._exec:  # never resurrect an evicted execution through a late event
            await super().append_event(execution_id, name, ts, data)


def harness_settings(env: Mapping[str, str] | None = None) -> DracoSettings:
    env = os.environ if env is None else env
    settings = DracoSettings.from_env(env)
    if not env.get("DRACO_DEFAULT_TIMEOUT_S"):
        settings = settings.model_copy(update={"default_timeout_s": APP_DEFAULT_TIMEOUT_S})
    return settings


def build_harness(settings: DracoSettings | None = None, *, store: Any = None) -> Harness:
    settings = settings or harness_settings()
    if store is None:
        if settings.state_path:
            from draco.state.sqlite import SqliteStore

            store = SqliteStore(settings.state_path)
        else:
            store = BoundedMemoryStore()
    exporters = None if settings.trace_path else [AppLogExporter()]
    h = Harness(settings, store=store, exporters=exporters)
    h.register(CHAT, chat_workload(), description="grounded chat (/chat, /chat/stream)")
    h.register(AGENT, agent_runner_workload(), description="LangGraph agent (/agent/chat)")
    h.register(AGENT_RESUME, agent_resume_workload(), description="resume a waiting agent execution")

    def forget_trace(event) -> None:
        # The tracer keeps a per-execution copy for in-process inspection; the exporter and the store
        # already have the events, so drop it once a root execution finishes (bounded memory).
        if event.parent_execution_id is None and event.execution_id:
            getattr(h.tracer, "_by_exec", {}).pop(event.execution_id, None)

    h.on("execution.finished", forget_trace)
    return h


def get_harness() -> Harness:
    global _harness
    if _harness is None:
        _harness = build_harness()
    return _harness


def identity_for(user: str) -> Identity | None:
    """The trusted identity of an authenticated API-key principal. The API maps one key to one user and
    scopes tenancy to that user. With INFERENCE_AUTH_DISABLED (dev only) the principal is the placeholder
    'anonymous', which Draco refuses as an identity: such runs carry an unauthenticated user id instead,
    and are denied when DRACO_REQUIRE_IDENTITY is set."""
    try:
        return Identity(subject=user, tenant_id=user)
    except PolicyViolation:
        return None


def chat_policy() -> Policy:
    from app.inference.tools import TOOLS

    return Policy(permissions=frozenset({"read"}), allowed_tools=frozenset(TOOLS))


def agent_policy(registry, *, allow_web: bool = False) -> Policy:
    """The runner's own per-request permissions, as an execution policy. Approval-gated capabilities are
    allowed because the app enforces the human decision itself: `propose_knowledge_ingestion` only records
    a pending proposal, the graph interrupts before ingestion, and `ingest_approved_source` refuses without
    a durable, unexpired approval of the exact content."""
    perms = AGENT_PERMISSIONS | ({"web"} if allow_web else frozenset())
    return Policy(permissions=frozenset(perms), allowed_tools=frozenset(c.name for c in registry.all()),
                  allow_approval_required=True)


async def run_workload(target: str, payload: Any, user: str, *, policy: Policy, impl_key: str, impl: Any,
                       emit=None, idempotency_key: str | None = None, session_id: str | None = None,
                       harness: Harness | None = None) -> ExecutionResult:
    """Run one request as a Draco execution. `impl` is the dependency-injected orchestrator or runner."""
    h = harness or get_harness()
    identity = identity_for(user)
    if idempotency_key:
        # A completed run with this key is replayed by the harness. A run that stopped at the approval
        # interrupt is answered with its recorded response too, instead of starting a second run.
        prev = await h.store.find_by_key(scope_key(idempotency_key, user if identity else None))
        if prev is not None and prev.status == Status.waiting.value and prev.target == target:
            return ExecutionResult(Status.waiting, prev.execution_id, output=prev.output, replayed=True,
                                   session_id=prev.session_id, agent_id=prev.agent_id,
                                   correlation_id=prev.correlation_id)
    metadata: dict[str, Any] = {impl_key: impl}
    if emit is not None:
        metadata["emit"] = emit
    return await h.run(target, payload, identity=identity, user_id=None if identity else user, policy=policy,
                       idempotency_key=idempotency_key, session_id=session_id, metadata=metadata)
