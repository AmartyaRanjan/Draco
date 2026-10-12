"""Benchmark retrieval variants.

Usage: PYTHONPATH=. python scripts/benchmark_retrieval.py QUERIES.jsonl [--runs 5] [--k 5]
Each line: {"query": "...", "lang": "en|hi|mixed" (optional), "relevant_chunk_ids": ["..."] (optional)}
Quality metrics are computed only for lines that carry labels; none are fabricated.
"""
import argparse
import asyncio
import json
import statistics
import time

from app.retrieval import fusion
from app.retrieval.service import RetrievalService
from app.retrieval.timing import Trace


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p / 100 * len(xs)))], 2) if xs else None


def metrics(ranked_ids, relevant, k):
    hit = [i for i, c in enumerate(ranked_ids[:k]) if c in relevant]
    rr = next((1 / (i + 1) for i, c in enumerate(ranked_ids) if c in relevant), 0.0)
    return {"recall": len(set(ranked_ids[:k]) & relevant) / len(relevant), "mrr": rr, "hit": bool(hit)}


async def main(args):
    svc = RetrievalService()
    rows = [json.loads(l) for l in open(args.queries) if l.strip()]
    lat = {v: [] for v in ("dense", "bm25", "hybrid_rrf", "rerank_stage", "end_to_end_retrieval")}
    qual = {v: [] for v in ("dense", "bm25", "rrf", "colbert", "selected")}
    await svc.retrieve(rows[0]["query"])  # warm-up; cold start is not steady state
    for row in rows:
        q, rel = row["query"], set(row.get("relevant_chunk_ids", []))
        for run in range(args.runs):
            tr = Trace("bench")
            d = await svc.dense.retrieve(q, None, tr)
            l = await svc.lexical.retrieve(q, None, tr)
            lat["dense"].append(tr.stages["query_embedding"]["ms"] + tr.stages["qdrant_query"]["ms"])
            lat["bm25"].append(tr.stages["opensearch_query"]["ms"])
            t0 = time.perf_counter()
            fused = fusion.rrf(fusion.deduplicate(list(d), list(l))[0], svc.s.rrf_k)
            lat["hybrid_rrf"].append((time.perf_counter() - t0) * 1e3)
            out = await svc.retrieve(q)
            st = out.trace.stages
            lat["rerank_stage"].append(st["colbert_rerank"]["ms"])
            lat["end_to_end_retrieval"].append(sum(st[s]["ms"] for s in
                ("hybrid_retrieval", "deduplication", "rrf_fusion", "colbert_rerank", "selection", "context_assembly") if s in st))
        if rel:
            ranked = await asyncio.to_thread(svc.reranker.rerank, q, fused)
            sel = [c.chunk_id for c in out.selection.selected] if out.selection else []
            for name, ids in (("dense", [c.chunk_id for c in d]), ("bm25", [c.chunk_id for c in l]),
                              ("rrf", [c.chunk_id for c in fused]),
                              ("colbert", [c.chunk_id for c in ranked[0]]), ("selected", sel)):
                qual[name].append(metrics(ids, rel, args.k))
    print(json.dumps({"queries": len(rows), "runs": args.runs,
        "latency_ms": {k: {"p50": pct(v, 50), "p95": pct(v, 95)} for k, v in lat.items()},
        f"quality@{args.k}": {k: {m: round(statistics.mean(x[m] for x in v), 3) for m in ("recall", "mrr", "hit")}
                              for k, v in qual.items() if v} or "no labelled queries"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("queries")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    asyncio.run(main(ap.parse_args()))
