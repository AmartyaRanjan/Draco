"""Offline stand-ins for the retrieval backends/reranker/generator so examples run without Qdrant,
OpenSearch, GPUs or a model server. The real `RetrievalService` runs on top of them unchanged."""
import asyncio

from app.retrieval.context import TokenCounter
from app.retrieval.generator import GenerationResult, extract_citations
from app.retrieval.schema import Candidate
from app.retrieval.service import RetrievalService
from app.retrieval.settings import RetrievalSettings

CORPUS = {
    "c1": "Draco runs every workload through a harness that enforces policy and budgets.",
    "c2": "Plugins declare the services they provide and require; the host orders them.",
    "c3": "Tool calls are validated and authorized outside the model.",
}


def _cand(cid, **rank):
    return Candidate(chunk_id=cid, document_id="handbook", text=CORPUS[cid], page_start=int(cid[1:]),
                     page_end=int(cid[1:]), **rank)


class FakeRetriever:
    def __init__(self, ids, key, stage, fail=False):
        self.ids, self.key, self.stage, self.fail = ids, key, stage, fail

    async def retrieve(self, query, doc_ids, trace):
        with trace.stage(self.stage):
            await asyncio.sleep(0.001)
            if self.fail:
                raise RuntimeError("backend down")
            return [_cand(c, **{f"{self.key}_rank": i + 1, f"{self.key}_score": 1.0 - i / 10})
                    for i, c in enumerate(self.ids)]


class FakeReranker:
    def rerank(self, query, cands):
        for c in cands:
            c.colbert_score = 1.0 - int(c.chunk_id[1:]) / 10
        return sorted(cands, key=lambda c: -c.colbert_score), {"reranked": len(cands), "skipped": 0,
                                                              "batch_size": 8, "batches": 1}


class FakeGenerator:
    async def generate(self, query, ctx, weak):
        answer = f"Based on the handbook [S1]: {ctx.items[0].candidate.text}"
        return GenerationResult(answer, extract_citations(answer, ctx)[0], usage={"total_tokens": 57}, model="offline-fake")


def settings():
    return RetrievalSettings(
        dense_top_k=30, bm25_top_k=30, rrf_k=60, dense_timeout_s=5, bm25_timeout_s=5, colbert_model="m",
        colbert_device="cpu", colbert_share_backbone=True, colbert_batch_size=2, colbert_max_candidates=60,
        colbert_max_length=512, score_threshold=0.5, calibration_file=None, min_selected=3, max_selected=20,
        max_context_tokens=6000, tokenizer=None, llm_base_url=None, llm_api_key=None, llm_model=None,
        llm_timeout_s=5, llm_max_output_tokens=100, llm_temperature=0.0, log_level="INFO", diagnostics_verbose=False)


def retrieval_service(dense=("c1", "c2"), lexical=("c2", "c3"), generator=True, lexical_down=False):
    return RetrievalService(
        settings(), dense=FakeRetriever(list(dense), "dense", "qdrant_query"),
        lexical=FakeRetriever(list(lexical), "bm25", "opensearch_query", fail=lexical_down),
        reranker=FakeReranker(), counter=TokenCounter(None), generator=FakeGenerator() if generator else None,
        filename_lookup=lambda d: "handbook.pdf")
