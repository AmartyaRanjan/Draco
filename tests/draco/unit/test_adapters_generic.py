"""Provider-neutral adapters, exercised with fakes (no app, no network, no credentials)."""
import pytest
from pydantic import BaseModel

from draco.adapters.ingestion import ingest_workload
from draco.adapters.langgraph import graph_workload
from draco.adapters.model_provider import ModelRequest, ModelResponse, generate
from draco.adapters.retrieval import Evidence, RetrievalResult, retrieval_workload
from draco.core import BudgetLimits, DracoSettings, ExecutionContext, Policy, Status
from draco.core.identity import Identity
from draco.harness import Harness
from draco.security import Resource, RoleAuthorizer
from draco.tools import NoArgs, ToolRegistry, ToolRuntime, native_tool

ALICE = Identity("alice", "acme", frozenset({"analyst"}))
OWNER = {"d-acme": "acme", "d-globex": "globex"}


def doc_resource(doc_id):
    return Resource("document", doc_id, tenant_id=OWNER.get(doc_id, "unknown-tenant"))


class SharedIndex:
    """A backend that over-returns across tenants, like a shared vector index with a missing filter."""
    def __init__(self):
        self.calls = 0

    async def retrieve(self, ctx, query, document_ids):
        self.calls += 1
        return RetrievalResult(query, [Evidence("d-acme", "acme text"), Evidence("d-globex", "SECRET globex text")])


async def test_retrieval_withholds_other_tenants_documents_and_audits():
    h = Harness(DracoSettings())
    index = SharedIndex()
    h.register("retrieve", retrieval_workload(index, document_resource=doc_resource))
    denied = []
    h.on("security.denied", lambda e: denied.append((e.data["resource_id"], e.data["stage"] if "stage" in e.data else None)))
    r = await h.run("retrieve", "q", identity=ALICE)
    assert r.status == Status.completed
    assert [e["document_id"] for e in r.output["evidence"]] == ["d-acme"]
    assert "SECRET" not in str(r.output) and denied == [("d-globex", "post_retrieval")]


async def test_retrieval_authorizes_requested_documents_before_the_backend_runs():
    h = Harness(DracoSettings())
    index = SharedIndex()
    h.register("retrieve", retrieval_workload(index, document_resource=doc_resource))
    r = await h.run("retrieve", {"query": "q", "document_ids": ["d-globex"]}, identity=ALICE)
    assert r.status == Status.failed and r.error.code == "policy" and index.calls == 0


async def test_retrieval_fails_closed_without_a_document_resolver_in_identity_mode():
    h = Harness(DracoSettings())
    index = SharedIndex()
    h.register("retrieve", retrieval_workload(index))
    assert (await h.run("retrieve", {"query": "q", "document_ids": ["d-acme"]}, identity=ALICE)).error.code == "policy"
    assert (await h.run("retrieve", "q", identity=ALICE)).error.code == "policy"  # cannot verify what came back
    # a bare tenant claim without identity cannot pass the retrieval check either
    assert (await h.run("retrieve", "q", tenant_id="acme")).error.code == "policy"
    assert (await h.run("retrieve", "q")).status == Status.completed  # single-tenant, identity-less use is unchanged


async def test_retrieval_input_contract():
    h = Harness(DracoSettings())
    h.register("retrieve", retrieval_workload(SharedIndex()))
    assert (await h.run("retrieve", 5)).error.code == "contract"
    assert (await h.run("retrieve", {"query": "q", "document_ids": "d"})).error.code == "contract"


class FakeProvider:
    name = "fake"

    def __init__(self, fail=0):
        self.fail = fail

    async def generate(self, request):
        if self.fail:
            self.fail -= 1
            from draco.core import ProviderError
            raise ProviderError("503", retryable=True)
        return ModelResponse("hi", request.model or "m", input_tokens=7, output_tokens=5)


async def test_model_provider_usage_budget_and_events():
    h = Harness(DracoSettings())
    events = []
    h.on("model.*", lambda e: events.append((e.name, e.data.get("tokens"))))

    async def agent(ctx, n):
        for _ in range(n):
            await generate(ctx, FakeProvider(), ModelRequest([{"role": "user", "content": "x"}]))
        return "ok"
    h.register("agent", agent)
    r = await h.run("agent", 2)
    assert r.usage.model_calls == 2 and r.usage.tokens == 24
    assert events == [("model.before", None), ("model.after", 12)] * 2
    r = await h.run("agent", 3, limits=BudgetLimits(max_tokens=30))
    assert r.status == Status.budget_exhausted and r.error.details["resource"] == "tokens"
    r = await h.run("agent", 3, limits=BudgetLimits(max_model_calls=1))
    assert r.status == Status.budget_exhausted


async def test_model_failure_is_evented_and_retryable_error_preserved():
    h = Harness(DracoSettings())
    seen = []
    h.on("model.failed", lambda e: seen.append(e.data["error"]))

    async def agent(ctx, _):
        return await generate(ctx, FakeProvider(fail=1), ModelRequest([]))
    h.register("agent", agent)
    r = await h.run("agent")
    assert r.status == Status.failed and r.error.retryable and seen == ["ProviderError"]


async def test_ingest_workload_runs_injected_callable_and_authorizes():
    h = Harness(DracoSettings())
    ingested = []
    h.register("ingest", ingest_workload(lambda d, p: ingested.append((d, p)), resource_for=doc_resource))
    assert (await h.run("ingest", {"document_id": "d-acme", "file_path": "/f"}, identity=ALICE)).ok
    r = await h.run("ingest", {"document_id": "d-globex", "file_path": "/f"}, identity=ALICE)
    assert r.error.code == "policy" and ingested == [("d-acme", "/f")]  # no side effect for the foreign document
    assert (await h.run("ingest", {"document_id": "x"})).error.code == "contract"

    async def aingest(d, p):
        ingested.append(("async", d))
    h.register("ingest2", ingest_workload(aingest))
    assert (await h.run("ingest2", {"document_id": "d", "file_path": "/f"})).ok and ingested[-1] == ("async", "d")


class FakeGraph:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.cfg = result, exc, None

    async def ainvoke(self, state, cfg):
        self.cfg = cfg
        if self.exc:
            raise self.exc
        return self.result


async def test_langgraph_adapter_thread_identity_interrupt_and_recursion():
    h = Harness(DracoSettings())
    g = FakeGraph({"n": 2})
    h.register("g", graph_workload(g, to_state=lambda n: {"n": n}, from_state=lambda s: s["n"], recursion_limit=7))
    r = await h.run("g", 1, identity=ALICE, session_id="s1")
    assert r.output == 2 and g.cfg["configurable"]["thread_id"] == r.execution_id and g.cfg["recursion_limit"] == 7
    assert g.cfg["metadata"]["draco_tenant_id"] == "acme" and g.cfg["metadata"]["draco_session_id"] == "s1"

    class Interrupt:
        value = {"approve": "refund"}
    h.register("gi", graph_workload(FakeGraph({"__interrupt__": [Interrupt()]})))
    r = await h.run("gi", 1)
    assert r.status == Status.waiting and r.output == [{"approve": "refund"}] and r.checkpoint_id == r.execution_id

    class GraphRecursionError(Exception):
        pass
    h.register("gr", graph_workload(FakeGraph(exc=GraphRecursionError("limit")), recursion_limit=3))
    r = await h.run("gr", 1)
    assert r.status == Status.budget_exhausted and r.error.details["resource"] == "graph_steps"
    h.register("gx", graph_workload(FakeGraph(exc=ValueError("secret detail"))))
    r = await h.run("gx", 1)
    assert r.status == Status.failed and "secret detail" not in r.error.message


async def test_real_langgraph_interrupt_and_resume_round_trip():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt
    from typing_extensions import TypedDict

    class S(TypedDict, total=False):
        q: str
        answer: str

    def ask(state):
        return {"answer": interrupt({"question": state["q"]})}
    g = StateGraph(S)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    graph = g.compile(checkpointer=InMemorySaver())
    h = Harness(DracoSettings())
    h.register("g", graph_workload(graph, to_state=lambda x: x if isinstance(x, Command) else {"q": x}))
    r = await h.run("g", "approve?")
    assert r.status == Status.waiting and r.output == [{"question": "approve?"}]
    done = await h.resume(r.execution_id, input=Command(resume="yes"))
    assert done.status == Status.completed and done.output["answer"] == "yes"


# -- tools: authorization precedes handler execution -----------------------------------------------------------

async def test_tool_call_requires_authorizer_grant_before_handler_runs():
    reg, ran = ToolRegistry(), []

    @native_tool(reg, name="export", description="x", input_model=NoArgs)
    async def export(args, ctx):
        ran.append(ctx.identity.subject)
        return {"ok": True}
    rt = ToolRuntime(reg)
    h = Harness(DracoSettings(), authorizer=RoleAuthorizer({"analyst": {"tool.call"}}))
    ctx = lambda ident: ExecutionContext(services=h.services, identity=ident, policy=Policy(allowed_tools=frozenset({"export"})))
    assert (await rt.call(ctx(ALICE), "export", {})).status == "success" and ran == ["alice"]
    viewer = Identity("v", "acme", frozenset({"viewer"}))
    r = await rt.call(ctx(viewer), "export", {})
    assert r.error.code.value == "denied" and ran == ["alice"]
    r = await rt.call(ctx(None), "export", {})  # no identity: the grant table cannot match
    assert r.error.code.value == "denied" and ran == ["alice"]
    strict = Harness(DracoSettings(require_identity=True))
    c = ExecutionContext(services=strict.services, policy=Policy(allowed_tools=frozenset({"export"})))
    assert (await rt.call(c, "export", {})).error.code.value == "denied"


async def test_tool_arguments_stay_out_of_audit_events():
    reg = ToolRegistry()

    class Args(BaseModel):
        password: str

    @native_tool(reg, name="login", description="x", input_model=Args)
    async def login(args, ctx):
        return {"ok": True}
    h = Harness(DracoSettings())
    ctx = ExecutionContext(services=h.services, policy=Policy(allowed_tools=frozenset({"login"})))
    await ToolRuntime(reg).call(ctx, "login", {"password": "hunter2-very-secret"})
    assert "hunter2" not in str(h.tracer.trace(ctx.execution_id))


async def test_tool_returning_non_object_is_an_internal_error_not_a_crash():
    reg = ToolRegistry()
    from draco.tools import Outcome, Tool

    async def bad(args, ctx):
        return Outcome(["not", "an", "object"])
    reg.register(Tool(name="bad", description="x", input_model=NoArgs, handler=bad))
    h = Harness(DracoSettings())
    ctx = ExecutionContext(services=h.services, policy=Policy(allowed_tools=frozenset({"bad"})))
    r = await ToolRuntime(reg).call(ctx, "bad", {})
    assert r.error.code.value == "internal"
