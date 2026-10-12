"""Regression tests for RRF with rankless candidates (no heavy deps)."""
import unittest

from app.retrieval import fusion
from app.retrieval.schema import Candidate


def cand(chunk_id, **kw):
    return Candidate(chunk_id=chunk_id, document_id="d1", text=f"text {chunk_id}", **kw)


class RrfRanklessTests(unittest.TestCase):
    def test_rankless_candidate_does_not_crash(self):
        out = fusion.rrf([cand("c1")])
        self.assertEqual(out[0].rrf_score, 0.0)
        self.assertEqual(out[0].rrf_rank, 1)

    def test_rankless_sorts_after_ranked_and_ties_by_chunk_id(self):
        ranked = cand("c9", dense_rank=1, dense_score=0.9)
        a = cand("c2")
        b = cand("c1")
        out = fusion.rrf([a, ranked, b])
        self.assertEqual([c.chunk_id for c in out], ["c9", "c1", "c2"])
        self.assertEqual([c.rrf_rank for c in out], [1, 2, 3])

    def test_existing_ranked_behavior_unchanged(self):
        a = cand("c1", dense_rank=1, bm25_rank=2)
        b = cand("c2", dense_rank=2)
        out = fusion.rrf([b, a], k=60)
        self.assertAlmostEqual(a.rrf_score, 1 / 61 + 1 / 62)
        self.assertAlmostEqual(b.rrf_score, 1 / 62)
        self.assertEqual([c.chunk_id for c in out], ["c1", "c2"])


if __name__ == "__main__":
    unittest.main()
