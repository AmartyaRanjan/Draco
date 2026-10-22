"""Starter-app adapters honour Draco's identity and authorization boundary."""
from app.agent.contracts import CapabilityError, ErrorCode
from app.agent.registry import Capability, CapabilityRegistry
from pydantic import BaseModel

from draco.adapters import agent_runner_workload, rag_workload, retrieval_workload
from draco.adapters.app_capabilities import register_app_capabilities
from draco.core import DracoSettings, ExecutionContext, Policy, Status
from draco.core.identity import Identity
from draco.harness import Harness
from draco.security import Resource
from draco.tools import ToolRegistry, ToolRuntime
from tests.agent_helpers import FakeRetrieval, ai, cand, make_runner
from tests.draco.integration.test_rag_adapter import Gen, owned_documents, svc  # noqa: F401

ALICE = Identity("tester", "acme")  # "tester" owns d1..d3 in the owned_documents fixture


def owned_by(tenant):
    return lambda doc_id: Resource("document", doc_id, tenant_id=tenant)


async def test_app_retrieval_filters_foreign_documents_and_rag_refuses_to_generate_from_them(owned_documents):  # noqa: F811
    gen = Gen()
    s = svc(gen=gen)
    h = Harness(DracoSettings(default_timeout_s=5))
    h.register("mine_retrieve", retrieval_workload(s, document_resource=owned_by("acme")))
    h.register("mine_rag", rag_workload(s, document_resource=owned_by("acme")))
    h.register("foreign_retrieve", retrieval_workload(s, document_resource=owned_by("globex")))
    h.register("foreign_rag", rag_workload(s, document_resource=owned_by("globex")))
    r = await h.run("mine_retrieve", "what is x", identity=ALICE)
    assert r.ok and r.output["items"]
    assert (await h.run("mine_rag", {"query": "q"}, identity=ALICE)).ok and gen.calls == 1
    r = await h.run("foreign_retrieve", "what is x", identity=ALICE)  # the index returned them; the caller may not read them
    assert r.ok and r.output["items"] == []
    r = await h.run("foreign_rag", {"query": "q"}, identity=ALICE)
    assert r.status == Status.failed and r.error.code == "policy" and gen.calls == 1  # no generation from foreign evidence
    r = await h.run("foreign_retrieve", {"query": "q", "document_ids": ["d1"]}, identity=ALICE)
    assert r.error.code == "policy"


async def test_agent_runner_adapter_never_falls_back_to_a_shared_user():
    runner, _ = make_runner([ai("A [S1].")], retrieval=FakeRetrieval([cand(1, "alpha")]))
    h = Harness(DracoSettings(default_timeout_s=10))
    h.register("assistant", agent_runner_workload(runner))
    r = await h.run("assistant", "q")  # no user at all
    assert r.status == Status.failed and r.error.code == "policy"


class In(BaseModel):
    n: int = 1


class Out(BaseModel):
    n: int


async def test_app_capabilities_run_as_draco_tools_with_error_mapping():
    app = CapabilityRegistry()
    seen = []

    async def ok(args, cctx):
        seen.append((cctx.user_id, cctx.trace_id))
        return {"n": args.n}

    async def boom(args, cctx):
        raise CapabilityError(ErrorCode.upstream_error, "503", retryable=True)
    app.register(Capability("ok", "g", "d", In, Out, ok))
    app.register(Capability("boom", "g", "d", In, Out, boom, mutates_state=True))
    tools = ToolRegistry()
    assert register_app_capabilities(tools, app, services=object()) == ["ok", "boom"]
    h = Harness(DracoSettings())
    rt = ToolRuntime(tools)
    pol = Policy(allowed_tools=frozenset({"ok", "boom"}))
    ctx = ExecutionContext(services=h.services, user_id="alice", policy=pol)
    r = await rt.call(ctx, "ok", {"n": 4})
    assert r.status == "success" and r.data == {"n": 4} and seen == [("alice", ctx.execution_id)]
    e = await rt.call(ctx, "boom", {})
    assert e.status == "error" and e.error.code.value == "upstream_error" and e.error.retryable
    # no user, no call
    anon = ExecutionContext(services=h.services, policy=pol)
    d = await rt.call(anon, "ok", {})
    assert d.status == "error" and len(seen) == 1
