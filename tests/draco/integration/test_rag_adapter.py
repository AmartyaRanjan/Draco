"""Existing RetrievalService + generator contracts through the harness, with the app's own fakes."""
import asyncio

from app.retrieval.context import TokenCounter
from app.retrieval.generator import GenerationResult, extract_citations
from app.retrieval.schema import RetrievalError
from app.retrieval.service import RetrievalService

from draco.adapters import map_app_error, rag_workload, retrieval_workload
from draco.core import BudgetLimits, DracoSettings, Policy, Status
from draco.harness import Harness
from tests.test_retrieval import FakeReranker, FakeRetriever, cand, settings

D = [cand(1, dense_rank=1, dense_score=.9), cand(2, dense_rank=2, dense_score=.8)]
L = [cand(2, bm25_rank=1, bm25_score=7.), cand(3, bm25_rank=2, bm25_score=3.)]


class Gen:
    def __init__(self, fail=None):
        self.calls, self.fail = 0, fail

    async def generate(self, q, ctx, weak):
        self.calls += 1
        if self.fail:
            raise self.fail
        return GenerationResult("answer [S1]", extract_citations("[S1]", ctx)[0], usage={"total_tokens": 42}, model="fake")


def svc(dense=None, lex=None, gen=None):
    return RetrievalService(settings(), dense=dense or FakeRetriever(D, stage="q"), lexical=lex or FakeRetriever(L, stage="o"),
                            reranker=FakeReranker(), counter=TokenCounter(None), generator=gen,
                            filename_lookup=lambda d: "f.pdf")


def harness(service):
    h = Harness(DracoSettings(default_timeout_s=5))
    h.register("retrieve", retrieval_workload(service))
    h.register("rag", rag_workload(service))
    return h


async def test_retrieval_through_harness_uses_execution_id_and_traces():
    s = svc()
    h = harness(s)
    r = await h.run("retrieve", "what is x", session_id="S")
    assert r.status == Status.completed
    assert r.output["trace"]["request_id"] == r.execution_id  # one identity across both tracing layers
    assert [i["chunk_id"] for i in r.output["items"]][0] in {"c1", "c2", "c3"}
    assert "colbert_rerank" in r.output["trace"]["stages"]
    events = [e["event"] for e in h.tracer.trace(r.execution_id)]
    assert "retrieval.after" in events
    # the service is still directly callable, outside the harness
    direct = await s.retrieve("what is x")
    assert len(direct.context.items) == len(r.output["items"])


async def test_rag_model_call_is_budgeted_and_evented_and_usage_recorded():
    gen = Gen()
    h = harness(svc(gen=gen))
    r = await h.run("rag", {"query": "q"})
    assert r.output["answer"] == "answer [S1]" and r.output["citations"][0]["filename"] == "f.pdf"
    assert r.usage.model_calls == 1 and r.usage.tokens == 42
    r = await h.run("rag", "q", limits=BudgetLimits(max_model_calls=1, max_tokens=10))
    assert r.status == Status.budget_exhausted and r.error.details["resource"] == "tokens"


async def test_abstains_without_model_call_when_no_evidence_or_no_generator():
    gen = Gen()
    h = harness(svc(FakeRetriever([], stage="q"), FakeRetriever([], stage="o"), gen))
    r = await h.run("rag", "q")
    assert r.status == Status.completed and r.output["abstained"] == "no_evidence" and gen.calls == 0
    h = harness(svc())
    assert (await h.run("rag", "q")).output["abstained"] == "generator_not_configured"


async def test_failure_modes_map_to_typed_errors():
    h = harness(svc(FakeRetriever(D, fail=True, stage="q"), FakeRetriever(L, fail=True, stage="o")))
    r = await h.run("retrieve", "q")
    assert r.status == Status.failed and r.error.code == "provider" and r.error.retryable  # 503, both retrievers down
    r = await harness(svc()).run("retrieve", "")
    assert r.error.code == "contract"  # 422 from validation
    r = await harness(svc(gen=Gen(RetrievalError("Generator timed out", "llm", 504)))).run("rag", "q")
    assert r.status == Status.timeout and r.error.code == "timeout"
    r = await harness(svc()).run("retrieve", 123)
    assert r.error.code == "contract"
    assert map_app_error(ValueError("x")) is None


async def test_degraded_retrieval_still_completes_and_reports():
    h = harness(svc(FakeRetriever(D, stage="q"), FakeRetriever(L, fail=True, stage="o")))
    r = await h.run("retrieve", "q")
    assert r.status == Status.completed and r.output["degraded"] == ["bm25"]


async def test_cancellation_and_timeout_propagate_into_retrieval():
    h = harness(svc(FakeRetriever(D, delay=5, stage="q"), FakeRetriever(L, delay=5, stage="o")))
    r = await h.run("retrieve", "q", timeout_s=0.1)
    assert r.status == Status.timeout
    t = asyncio.create_task(h.run("retrieve", "q"))
    await asyncio.sleep(0.1)
    (eid,) = list(h._inflight)
    h.cancel(eid)
    assert (await t).status == Status.cancelled


async def test_concurrent_rag_sessions_do_not_leak():
    h = harness(svc(gen=Gen()))
    rs = await asyncio.gather(*(h.run("rag", f"q{i}", session_id=f"s{i}") for i in range(6)))
    assert len({r.output["trace"]["request_id"] for r in rs}) == 6
    assert [r.session_id for r in rs] == [f"s{i}" for i in range(6)]


async def test_policy_blocks_workload():
    h = harness(svc())
    r = await h.run("retrieve", "q", policy=Policy(allowed_agents=frozenset({"rag"})))
    assert r.error.code == "policy"
