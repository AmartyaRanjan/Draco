"""Dense (BGE-M3 + Qdrant) and lexical (OpenSearch BM25) retrievers."""
import asyncio
import re
import threading

from qdrant_client import models

from app.core.blocking import run_blocking
from app.db import opensearch as os_db
from app.db import qdrant as qd_db
from app.ingestion import embed
from app.retrieval.schema import Candidate

# One lock serialises GPU use of the shared BGE-M3 backbone (embedding + ColBERT).
GPU_LOCK = threading.Lock()

_SOURCE_FIELDS = ["chunk_id", "document_id", "document_version", "chunk_index",
                  "text", "page_start", "page_end", "section_path"]


def prepare_query(query: str) -> str:
    """Unicode-safe whitespace normalisation only; Hindi/English/mixed pass through."""
    return re.sub(r"\s+", " ", query).strip()


def _cand(src: dict, **kw) -> Candidate:
    return Candidate(**{f: src.get(f) for f in _SOURCE_FIELDS if f in src}, **kw)


def _embed_query(query: str) -> list[float]:
    with GPU_LOCK:
        return embed.embed_texts([query])[0]


class DenseRetriever:
    def __init__(self, top_k: int, timeout_s: float):
        self.top_k, self.timeout_s = top_k, timeout_s

    def _search(self, vector, document_ids):
        flt = None
        if document_ids:
            flt = models.Filter(must=[models.FieldCondition(
                key="document_id", match=models.MatchAny(any=list(document_ids)))])
        res = qd_db.get_qdrant_client().query_points(
            collection_name=qd_db.COLLECTION_NAME, query=vector, limit=self.top_k,
            query_filter=flt, with_payload=True, timeout=max(1, int(self.timeout_s)),
        )
        return [_cand(p.payload, dense_rank=i, dense_score=float(p.score))
                for i, p in enumerate(res.points, 1)]

    async def retrieve(self, query, document_ids, trace) -> list[Candidate]:
        with trace.stage("query_embedding"):
            vector = await run_blocking(_embed_query, query, timeout=self.timeout_s)
        with trace.stage("qdrant_query"):
            return await run_blocking(self._search, vector, document_ids, timeout=self.timeout_s)


class LexicalRetriever:
    def __init__(self, top_k: int, timeout_s: float):
        self.top_k, self.timeout_s = top_k, timeout_s

    def _search(self, query, document_ids):
        flt = [{"terms": {"document_id": list(document_ids)}}] if document_ids else []
        body = {
            "size": self.top_k,
            "_source": _SOURCE_FIELDS,
            "query": {"bool": {
                "must": [{"multi_match": {"query": query, "fields": ["text", "section_path^0.5"]}}],
                "filter": flt,
            }},
        }
        res = os_db.get_opensearch_client().search(
            index=os_db.INDEX_NAME, body=body, request_timeout=self.timeout_s)
        return [_cand(h["_source"], bm25_rank=i, bm25_score=float(h["_score"]))
                for i, h in enumerate(res["hits"]["hits"], 1)]

    async def retrieve(self, query, document_ids, trace) -> list[Candidate]:
        with trace.stage("opensearch_query"):
            return await run_blocking(self._search, query, document_ids, timeout=self.timeout_s)
