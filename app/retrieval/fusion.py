"""Exact chunk_id deduplication and Reciprocal Rank Fusion."""
import logging

from app.retrieval.schema import Candidate

logger = logging.getLogger("rag.retrieval")


def deduplicate(dense: list[Candidate], lexical: list[Candidate]) -> tuple[list[Candidate], dict]:
    """Merge by chunk_id keeping both ranks/scores. Inputs carry their own rank."""
    merged: dict[str, Candidate] = {}
    overlap = conflicts = 0
    for c in dense:
        merged.setdefault(c.chunk_id, c)
    for c in lexical:
        cur = merged.get(c.chunk_id)
        if cur is None:
            merged[c.chunk_id] = c
            continue
        overlap += 1
        if cur.text != c.text or (
            cur.document_id != c.document_id
            or (cur.document_version and c.document_version and cur.document_version != c.document_version)
        ):
            # Inconsistent stores for one ID: keep the Qdrant record, flag it.
            conflicts += 1
            cur.conflict = True
            logger.warning("chunk_conflict", extra={"chunk_id": c.chunk_id})
        cur.bm25_rank, cur.bm25_score = c.bm25_rank, c.bm25_score
    stats = {"overlap": overlap, "unique": len(merged), "conflicts": conflicts}
    return list(merged.values()), stats


def rrf(candidates: list[Candidate], k: int = 60) -> list[Candidate]:
    """score = sum 1/(k+rank), one-based ranks; absent ranking contributes 0.
    Ties break on best single-source rank, then chunk_id (deterministic)."""
    for c in candidates:
        c.rrf_score = sum(1.0 / (k + r) for r in (c.dense_rank, c.bm25_rank) if r is not None)
    ordered = sorted(
        candidates,
        key=lambda c: (-c.rrf_score, min(r for r in (c.dense_rank, c.bm25_rank) if r is not None), c.chunk_id),
    )
    for i, c in enumerate(ordered, 1):
        c.rrf_rank = i
    return ordered
