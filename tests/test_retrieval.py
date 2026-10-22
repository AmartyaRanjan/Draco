import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.retrieval import fusion
from app.retrieval.context import TokenCounter, build_context
from app.retrieval.generator import GenerationResult, extract_citations
from app.retrieval.schema import Candidate, RetrievalError
from app.retrieval.selection import FALLBACK, PASSED, Calibrator, select
from app.retrieval.service import RetrievalService
from app.retrieval.settings import RetrievalSettings
from app.retrieval.timing import Trace
from app.db import registry


def cand(i, text=None, **kw):
    return Candidate(chunk_id=f"c{i}", document_id="d1", text=text if text is not None else f"text {i}",
                     page_start=i, page_end=i, **kw)


def settings(**kw):
    base = dict(dense_top_k=30, bm25_top_k=30, rrf_k=60, dense_timeout_s=5, bm25_timeout_s=5,
                colbert_model="m", colbert_device="cpu", colbert_share_backbone=True,
                colbert_batch_size=2, colbert_max_candidates=60, colbert_max_length=512,
                score_threshold=0.5, calibration_file=None, min_selected=3, max_selected=20,
                max_context_tokens=6000, tokenizer=None, llm_base_url=None, llm_api_key=None,
                llm_model=None, llm_timeout_s=5, llm_max_output_tokens=100, llm_temperature=0.0,
                log_level="INFO", diagnostics_verbose=False)
    return RetrievalSettings(**{**base, **kw})


class FusionTests(unittest.TestCase):
    def test_overlap_keeps_both_ranks_and_single_source_survives(self):
        d = [cand(1, dense_rank=1, dense_score=.9), cand(2, dense_rank=2, dense_score=.8)]
        l = [cand(2, bm25_rank=1, bm25_score=12.), cand(3, bm25_rank=2, bm25_score=5.)]
        out, st = fusion.deduplicate(d, l)
        by = {c.chunk_id: c for c in out}
        self.assertEqual((st["overlap"], st["unique"]), (1, 3))
        self.assertEqual((by["c2"].dense_rank, by["c2"].bm25_rank), (2, 1))
        self.assertEqual((by["c2"].dense_score, by["c2"].bm25_score), (.8, 12.))
        self.assertIsNone(by["c3"].dense_rank)

    def test_conflict_is_flagged_not_merged_silently(self):
        out, st = fusion.deduplicate([cand(1, "a", dense_rank=1)], [cand(1, "b", bm25_rank=1)])
        self.assertEqual(st["conflicts"], 1)
        self.assertTrue(out[0].conflict)
        self.assertEqual(out[0].text, "a")

    def test_rrf_scores_one_based_and_missing_rank_zero(self):
        a = cand(1, dense_rank=1, bm25_rank=2)
        b = cand(2, dense_rank=2)
        out = fusion.rrf([b, a], k=60)
        self.assertAlmostEqual(a.rrf_score, 1 / 61 + 1 / 62)
        self.assertAlmostEqual(b.rrf_score, 1 / 62)
        self.assertEqual([c.chunk_id for c in out], ["c1", "c2"])
        self.assertEqual([c.rrf_rank for c in out], [1, 2])

    def test_rrf_ignores_raw_scores_and_ties_are_deterministic(self):
        a = cand(1, dense_rank=1, dense_score=0.1)
        b = cand(2, bm25_rank=1, bm25_score=999.)
        for order in ([a, b], [b, a]):
            self.assertEqual([c.chunk_id for c in fusion.rrf(list(order))], ["c1", "c2"])


class SelectionTests(unittest.TestCase):
    def ranked(self, scores):
        return [cand(i, colbert_score=s) for i, s in enumerate(scores)]

    def test_all_qualifying_kept_up_to_max(self):
        r = select(self.ranked([.9, .8, .7, .6, .2]), .5, 3, 20)
        self.assertEqual((r.passed, r.fallback, len(r.selected)), (4, 0, 4))
        self.assertTrue(r.evidence_sufficient)

    def test_fallback_marked_and_highest_ranked(self):
        r = select(self.ranked([.9, .4, .3, .2]), .5, 3, 20)
        self.assertEqual([c.selection for c in r.selected], [PASSED, FALLBACK, FALLBACK])
        self.assertEqual([c.chunk_id for c in r.selected], ["c0", "c1", "c2"])

    def test_no_fabrication_and_insufficient(self):
        self.assertEqual(len(select(self.ranked([.1, .05]), .5, 3, 20).selected), 2)
        self.assertEqual(len(select(self.ranked([.1]), .5, 3, 20).selected), 1)
        r = select([], .5, 3, 20)
        self.assertEqual(r.selected, [])
        self.assertFalse(r.evidence_sufficient)
        self.assertFalse(select(self.ranked([.1, .2]), .5, 3, 20).evidence_sufficient)

    def test_invalid_and_duplicate_candidates_skipped(self):
        r = [cand(1, colbert_score=.9), cand(1, colbert_score=.9), cand(2, "", colbert_score=.9), cand(3)]
        self.assertEqual([c.chunk_id for c in select(r, .5, 3, 20).selected], ["c1"])

    def test_raw_vs_calibrated_semantics(self):
        r = select(self.ranked([.9]), .5, 1, 20)
        self.assertIsNone(r.selected[0].calibrated_probability)
        self.assertEqual(r.basis, "raw_colbert_score")
        cal = Calibrator([(0.0, 0.0), (1.0, 1.0)])
        r = select(self.ranked([.9, .5]), .8, 1, 20, cal)
        self.assertEqual(r.basis, "calibrated_probability")
        self.assertAlmostEqual(r.selected[0].calibrated_probability, .9)
        self.assertEqual(r.passed, 1)


class ContextTests(unittest.TestCase):
    def test_budget_enforced_and_provenance_kept(self):
        sel = [cand(i, "x" * 100, selection=PASSED) for i in range(1, 6)]
        counter = TokenCounter(None)
        ctx = build_context(sel, {"d1": "a.pdf"}, 300, counter)
        self.assertLess(len(ctx.items), 5)
        self.assertEqual(len(ctx.items) + len(ctx.dropped), 5)
        self.assertLessEqual(ctx.tokens, 300)
        self.assertIn("a.pdf", ctx.items[0].text_block)
        self.assertIn("pages: 1", ctx.items[0].text_block)
        self.assertEqual(ctx.items[0].candidate.text, "x" * 100)  # never truncated

    def test_citations_only_from_real_chunks(self):
        ctx = build_context([cand(1, selection=PASSED)], {"d1": "a.pdf"}, 1000, TokenCounter(None))
        cites, invalid = extract_citations("Yes [S1]. Also [S9] and [S1].", ctx)
        self.assertEqual(len(cites), 1)
        self.assertEqual((cites[0]["filename"], cites[0]["page_start"], cites[0]["chunk_id"]), ("a.pdf", 1, "c1"))
        self.assertEqual(invalid, ["S9"])


class TimingTests(unittest.TestCase):
    def test_exception_records_time_and_failed_stage(self):
        t = Trace("r")
        with self.assertRaises(ValueError):
            with t.stage("boom"):
                raise ValueError
        self.assertEqual(t.stages["boom"]["status"], "error")
        self.assertEqual(t.failed_stage, "boom")
        self.assertGreaterEqual(t.stages["boom"]["ms"], 0)


class FakeRetriever:
    def __init__(self, cands=None, delay=0.0, fail=False, stage="x"):
        self.cands, self.delay, self.fail, self.stage = cands or [], delay, fail, stage

    async def retrieve(self, q, docs, trace):
        with trace.stage(self.stage):
            await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("down")
            return [replace(c) for c in self.cands]


class FakeReranker:
    def __init__(self):
        self.seen = []

    def rerank(self, q, cands):
        self.seen.append([c.chunk_id for c in cands])
        for c in cands:
            c.colbert_score = 1.0 - int(c.chunk_id[1:]) / 10
        return sorted(cands, key=lambda c: -c.colbert_score), {"reranked": len(cands), "skipped": 0,
                                                              "batch_size": 2, "batches": 1}


class FakeGen:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    async def generate(self, q, ctx, weak):
        self.calls.append((ctx, weak))
        if self.fail:
            raise RetrievalError("Generator timed out", "llm", 504)
        return GenerationResult("ans [S1]", extract_citations("[S1]", ctx)[0])


def svc(dense=None, lex=None, gen=None, **kw):
    return RetrievalService(settings(**kw), dense=dense, lexical=lex, reranker=FakeReranker(),
                            counter=TokenCounter(None), generator=gen, filename_lookup=lambda d: "f.pdf")


class ServiceTests(unittest.TestCase):
    D = [cand(1, dense_rank=1, dense_score=.9), cand(2, dense_rank=2, dense_score=.8)]
    L = [cand(2, bm25_rank=1, bm25_score=7.), cand(3, bm25_rank=2, bm25_score=3.)]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_patch = patch.object(registry, "DB_PATH", Path(self.tmp.name) / "documents.db")
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        registry.init_db()
        for document_id in ("d1", "d2", "d3"):
            registry.create_document(document_id, f"{document_id}.pdf", f"/{document_id}", "tester")

    def test_parallel_wall_clock_not_sum_and_rerank_once(self):
        s = svc(FakeRetriever(self.D, .2, stage="qdrant_query"), FakeRetriever(self.L, .2, stage="opensearch_query"))
        out = asyncio.run(s.retrieve("q", principal="tester"))
        out.trace.finish()
        st = out.trace.stages
        self.assertLess(st["hybrid_retrieval"]["ms"], st["qdrant_query"]["ms"] + st["opensearch_query"]["ms"] - 100)
        self.assertEqual(out.trace.metrics["unique_candidates"], 3)
        self.assertEqual(out.trace.metrics["overlap"], 1)
        self.assertEqual(len(s.reranker.seen), 1)
        self.assertEqual(len(set(s.reranker.seen[0])), len(s.reranker.seen[0]))
        for stage in ("request_validation", "deduplication", "rrf_fusion", "colbert_rerank",
                      "selection", "context_assembly", "total"):
            self.assertIn(stage, st)

    def test_degraded_mode_and_both_fail(self):
        out = asyncio.run(svc(FakeRetriever(fail=True), FakeRetriever(self.L)).retrieve("q", principal="tester"))
        self.assertEqual(out.degraded, ["dense"])
        self.assertGreater(len(out.context.items), 0)
        with self.assertRaises(RetrievalError) as cm:
            asyncio.run(svc(FakeRetriever(fail=True), FakeRetriever(fail=True)).retrieve("q", principal="tester"))
        self.assertEqual(cm.exception.stage, "hybrid_retrieval")

    def test_invalid_query_and_empty_candidates(self):
        with self.assertRaises(RetrievalError):
            asyncio.run(svc(FakeRetriever(), FakeRetriever()).retrieve("   "))
        gen = FakeGen()
        out, g = asyncio.run(svc(FakeRetriever(), FakeRetriever(), gen).answer("q", principal="tester"))
        self.assertIsNone(g)
        self.assertEqual(gen.calls, [])
        self.assertEqual(out.trace.metrics["abstained"], "no_evidence")

    def test_generator_receives_selected_context_and_failure_is_controlled(self):
        gen = FakeGen()
        out, g = asyncio.run(svc(FakeRetriever(self.D), FakeRetriever(self.L), gen).answer("q", principal="tester"))
        ctx, weak = gen.calls[0]
        self.assertEqual([e.candidate.chunk_id for e in ctx.items], [c.chunk_id for c in out.selection.selected])
        self.assertEqual(g.citations[0]["filename"], "f.pdf")
        with self.assertRaises(RetrievalError) as cm:
            asyncio.run(svc(FakeRetriever(self.D), FakeRetriever(self.L), FakeGen(fail=True)).answer("q", principal="tester"))
        self.assertEqual(cm.exception.status_code, 504)

    def test_timing_does_not_change_results(self):
        a = asyncio.run(svc(FakeRetriever(self.D), FakeRetriever(self.L)).retrieve("q", principal="tester"))
        b = asyncio.run(svc(FakeRetriever(self.D), FakeRetriever(self.L)).retrieve("q", principal="tester"))
        self.assertEqual([c.chunk_id for c in a.selection.selected], [c.chunk_id for c in b.selection.selected])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_patch = patch.object(registry, "DB_PATH", Path(self.tmp.name) / "documents.db")
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        registry.init_db()
        for document_id in ("d1", "d2", "d3"):
            registry.create_document(document_id, f"{document_id}.pdf", f"/{document_id}", "tester")

    def test_validation_and_error_mapping(self):
        from app.api.chat import current_user
        from app.main import app
        app.dependency_overrides[current_user] = lambda: "tester"
        self.addCleanup(app.dependency_overrides.clear)
        client = TestClient(app)
        self.assertEqual(client.post("/query", json={"query": ""}).status_code, 422)
        s = svc(FakeRetriever(fail=True), FakeRetriever(fail=True))
        with patch("app.api.query.get_service", return_value=s):
            r = client.post("/query", json={"query": "hi"})
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("Traceback", r.text)
        s = svc(FakeRetriever(ServiceTests.D), FakeRetriever(ServiceTests.L), FakeGen())
        with patch("app.api.query.get_service", return_value=s):
            r = client.post("/query", json={"query": "hi", "include_diagnostics": True})
        body = r.json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(body["citations"][0]["chunk_id"], "c1")
        self.assertIn("stages", body["diagnostics"])


if __name__ == "__main__":
    unittest.main()
