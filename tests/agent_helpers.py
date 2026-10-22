"""Shared fakes for agent-layer tests."""
import json

from app.agent.evidence import EvidenceItem, EvidenceStore
from app.agent.registry import CapabilityContext
from app.inference.config import InferenceSettings


class CharCounter:
    """Deterministic counter: 1 token per 4 chars (rounded up)."""
    exact = True

    def count(self, text):
        return (len(text) + 3) // 4


def private(ref, text, doc="d1", chunk=None, **kw):
    return EvidenceItem(ref=ref, kind="private", text=text, document_id=doc, chunk_id=chunk or f"c-{ref}", **kw)


def web(ref, text, url="https://example.org/a", pid="p1", **kw):
    return EvidenceItem(ref=ref, kind="web", text=text, url=url, passage_id=pid, **kw)


def store_with(*items):
    st = EvidenceStore()
    for it in items:
        st.items[it.ref] = it
        st._by_key[it.source_key()] = it.ref
    return st


def isettings(**kw):
    base = dict(base_url="http://x/v1", api_key="k", model="Qwen/Qwen3-4B-FP8", timeout_s=5, max_retries=0,
                max_model_len=8192, max_output_tokens=1000, grounded_temperature=.3, grounded_top_p=.8,
                reasoning_temperature=.6, reasoning_top_p=.95, top_k=20, agent_max_steps=3,
                agent_max_tool_calls=2, agent_workflow_token_budget=3000, tool_timeout_s=2,
                tool_output_tokens=2500, history_tokens=800, memory_tokens=400,
                prompt_overhead_tokens=600, api_keys={"key1": "alice"}, auth_disabled=False, store_path=None)
    return InferenceSettings(**{**base, **kw})


def make_ctx(services=None, evidence=None, user="alice", perms=("read", "write", "web", "generate")):
    return CapabilityContext(user_id=user, trace_id="trace-1", services=services,
                             evidence=evidence or store_with(), permissions=frozenset(perms))


def scripted_llm(responses, log=None):
    """LlmCall that returns scripted strings (dicts are JSON-encoded) in order."""
    resp = list(responses)

    async def call(system, user, max_tokens):
        if log is not None:
            log.append({"system": system, "user": user, "max_tokens": max_tokens})
        r = resp.pop(0)
        if isinstance(r, Exception):
            raise r
        return (r if isinstance(r, str) else json.dumps(r)), {"input_tokens": 10, "output_tokens": 5}
    return call


# ---- service fakes -----------------------------------------------------------------------
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.agent.artifacts import ArtifactStore
from app.agent.catalog import build_registry
from app.agent.config import AgentSettings
from app.agent.services import Services
from app.agent.store import AgentStore
from app.retrieval.schema import Candidate
from app.retrieval.selection import SelectionResult
from app.retrieval.timing import Trace


def asettings(**kw):
    base = dict(store_path="", artifact_dir="", artifact_max_bytes=5_000_000, artifact_ttl_days=1, feedback_retention_days=30,
                max_tool_calls=12, max_retrieval_attempts=2, max_llm_steps=4, max_mode_changes=2, capability_timeout_s=10,
                evidence_compaction_ratio=.85, safety_margin_tokens=128, tool_reserve_tokens=400, summary_target_tokens=700, artifact_output_tokens=2400,
                summary_batch_tokens=2500, summary_max_depth=3, web_search_provider=None, web_search_url=None,
                web_search_api_key=None, web_allowed_ports=(80, 443), web_max_bytes=100_000, web_timeout_s=5,
                web_max_redirects=3, web_allow_private_hosts=False, approval_ttl_hours=24, web_snapshot_max_chars=100_000,
                image_api_url=None, image_api_key=None, image_model=None, analysis_max_bytes=1_000_000)
    return AgentSettings(**{**base, **kw})


def cand(i, text="alpha beta", score=.8, doc="d1", **kw):
    return Candidate(chunk_id=f"c{i}", document_id=doc, text=text, colbert_score=score, rrf_score=.03, page_start=1,
                     page_end=1, selection="passed_threshold", **kw)


class FakeRetrieval:
    def __init__(self, cands=None, fail=None, sufficient=True):
        self.cands, self.fail, self.sufficient, self.calls = cands or [], fail, sufficient, []

    async def retrieve(self, query, document_ids=None, request_id=None, principal=None):
        self.calls.append((query, document_ids))
        if self.fail:
            raise self.fail
        sel = SelectionResult(list(self.cands), len(self.cands), 0, self.sufficient, "raw_colbert_score") if self.cands else None
        tr = Trace(request_id or "r")
        return type("O", (), {"selection": sel, "trace": tr, "degraded": []})()

    def filename_for(self, d):
        return f"{d}.pdf"


def make_services(tmp: Path | None = None, **kw):
    tmp = tmp or Path(tempfile.mkdtemp())
    s = asettings(store_path=str(tmp / "agent.db"), artifact_dir=str(tmp / "art"))
    store = AgentStore(s.store_path)
    docs = {}
    svc = Services(settings=s, store=store, counter=CharCounter(),
                   artifact_store=ArtifactStore(s.artifact_dir, store, max_bytes=s.artifact_max_bytes, ttl_days=s.artifact_ttl_days),
                   document_lookup=lambda d, user: docs.get(d) if docs.get(d, {}).get("owner_id") == user else None,
                   pages_lookup=lambda d, user: [] if docs.get(d, {}).get("owner_id") == user else [])
    svc.extra["docs"], svc.extra["upload_dir"] = docs, str(tmp / "uploads")
    for k, v in kw.items():
        setattr(svc, k, v)
    return svc


REG = build_registry()


async def call(name, args, ctx, sink=None):
    return await REG.execute(name, args, ctx, sink=sink)


# ---- graph-level fakes ---------------------------------------------------------------------
from langchain_core.messages import AIMessage

from app.agent.graph import AgentRunner
from app.agent.runtime import RuntimeAdapter, detect_capabilities


def ai(text="", tool_calls=None, finish="stop", usage=(50, 20), invalid=None):
    return AIMessage(content=text, tool_calls=tool_calls or [], invalid_tool_calls=invalid or [], usage_metadata={"input_tokens": usage[0], "output_tokens": usage[1], "total_tokens": sum(usage)},
                     response_metadata={"finish_reason": finish})


def tc(name, args, id="t1"):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


class FakeChat:
    """Stand-in for ChatOpenAI: records constructor kwargs (incl. extra_body) and messages per call."""
    script: list = []
    log: list = []

    def __init__(self, **kw):
        self.kw, self.tools = kw, None

    def bind_tools(self, specs, tool_choice=None):
        self.tools, self.tool_choice = specs, tool_choice
        return self

    async def ainvoke(self, messages):
        FakeChat.log.append({"kw": self.kw, "messages": messages, "tools": [s["function"]["name"] for s in self.tools] if self.tools else None,
                         "tool_choice": getattr(self, "tool_choice", None)})
        r = FakeChat.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_runner(script, *, model="Qwen/Qwen3-4B-FP8", conv_store=None, inference=None, **svc_kw):
    FakeChat.script, FakeChat.log = list(script), []
    inf = inference or isettings(model=model)
    svc = make_services(**svc_kw)
    caps = detect_capabilities(model)
    adapter = RuntimeAdapter(inf, caps, client_cls=FakeChat)
    from app.agent.catalog import build_registry
    return AgentRunner(svc, build_registry(), adapter, inf, conv_store), svc
