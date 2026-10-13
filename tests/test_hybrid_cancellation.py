"""Regression tests: hybrid retrieval must propagate cancellation.

_hybrid() uses asyncio.gather(..., return_exceptions=True) and converts every
BaseException result into degraded mode. That incorrectly swallows
asyncio.CancelledError: a cancelled request is reported as degraded and keeps
burning GPU/time instead of cancelling. Cancellation must be re-raised.
"""
import asyncio
import sys
import unittest
from unittest.mock import MagicMock

for _name in (
    "sentence_transformers",
    "qdrant_client",
    "qdrant_client.models",
    "opensearchpy",
    "opensearchpy.helpers",
):
    sys.modules.setdefault(_name, MagicMock())

from app.retrieval.context import TokenCounter  # noqa: E402
from app.retrieval.service import RetrievalService  # noqa: E402
from app.retrieval.settings import RetrievalSettings  # noqa: E402
from app.retrieval.timing import Trace  # noqa: E402


def _settings(**kw):
    base = dict(dense_top_k=30, bm25_top_k=30, rrf_k=60, dense_timeout_s=5, bm25_timeout_s=5,
                colbert_model="m", colbert_device="cpu", colbert_share_backbone=True,
                colbert_batch_size=2, colbert_max_candidates=60, colbert_max_length=512,
                score_threshold=0.5, calibration_file=None, min_selected=3, max_selected=20,
                max_context_tokens=6000, tokenizer=None, llm_base_url=None, llm_api_key=None,
                llm_model=None, llm_timeout_s=5, llm_max_output_tokens=100, llm_temperature=0.0,
                log_level="INFO", diagnostics_verbose=False)
    return RetrievalSettings(**{**base, **kw})


class _CancelRetriever:
    async def retrieve(self, q, docs, trace):
        raise asyncio.CancelledError("cancel-test")


class _OkRetriever:
    async def retrieve(self, q, docs, trace):
        return []


class _FailRetriever:
    async def retrieve(self, q, docs, trace):
        raise RuntimeError("down")


def _svc(dense, lexical):
    return RetrievalService(
        _settings(), dense=dense, lexical=lexical, reranker=MagicMock(),
        counter=TokenCounter(None), generator=None, filename_lookup=lambda d: "f.pdf",
    )


class HybridCancellationTests(unittest.TestCase):
    def test_single_cancellation_propagates(self):
        svc = _svc(_CancelRetriever(), _OkRetriever())
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(svc._hybrid("q", None, Trace("r")))

    def test_double_cancellation_propagates_not_retrieval_error(self):
        svc = _svc(_CancelRetriever(), _CancelRetriever())
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(svc._hybrid("q", None, Trace("r")))

    def test_ordinary_failure_still_degrades(self):
        svc = _svc(_FailRetriever(), _OkRetriever())
        dense, lex, degraded = asyncio.run(svc._hybrid("q", None, Trace("r")))
        self.assertEqual(degraded, ["dense"])
        self.assertEqual(dense, [])


if __name__ == "__main__":
    unittest.main()
