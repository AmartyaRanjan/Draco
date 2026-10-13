"""RAG evaluation against the live stack, on tests/eval/rag_gold.jsonl (questions written from the indexed documents).

  PYTHONPATH=. python scripts/eval_rag.py --tag run1 [--no-generation] [--only id1,id2]

Phase 1 (retrieval, no LLM): per-stage ranking quality for dense-only, BM25-only, RRF hybrid, hybrid+ColBERT (the
production ranking) and the final selected context. Metrics follow BEIR/MTEB: hit@k, recall@k, MRR@10, nDCG@10, plus
RAGAS-style context precision / recall on the selected context.
Phase 2 (generation): the production /chat (mode=rag) orchestrator. Deterministic metrics only here: abstention,
key-fact recall, token F1, citation precision/recall. Judge-based metrics (faithfulness, answer relevancy, ...) are
computed by scripts/eval_judge.py from the judge_<tag>.jsonl this script writes.

Gold labels are resolved at run time: a chunk is relevant when its text contains an evidence quote from the gold file,
so the labels survive re-chunking as long as the quote is still in the index.
"""
import argparse
import asyncio
import json
import math
import re
import statistics
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient  # noqa: E402

from app.db import registry  # noqa: E402
from app.db.qdrant import COLLECTION_NAME, get_qdrant_client  # noqa: E402
from app.inference.config import get_inference_settings  # noqa: E402
from app.inference.generation import ChatOrchestrator  # noqa: E402
from app.inference.memory import SqliteConversationStore  # noqa: E402
from app.inference.schemas import ChatRequest, InferenceError  # noqa: E402
from app.retrieval import fusion  # noqa: E402
from app.retrieval.retrievers import prepare_query  # noqa: E402
from app.retrieval.service import RetrievalService  # noqa: E402
from app.retrieval.timing import Trace  # noqa: E402

KS = (1, 3, 5, 10)
SYSTEMS = ("dense", "bm25", "rrf", "hybrid_colbert", "selected")
DEVANAGARI = re.compile("[ऀ-ॿ]")
NUM = re.compile(r"[-+±]?\d[\d,]*\.?\d*")
WORD = re.compile(r"\w+", re.UNICODE)


SCOPE_PREFIX = re.compile(r"^(?:In|For) the .{3,160}?(?:report|roadmap|lab|text|slides|project)(?: \([^)]*\))?'?s?,\s+", re.I)


def strip_scope(question: str) -> str:
    """'In the X report, how many ...' -> 'how many ...'. Measures sensitivity to a document-descriptor prefix."""
    stripped = SCOPE_PREFIX.sub("", question, count=1)
    return stripped[:1].upper() + stripped[1:] if stripped != question else question


def load_corpus():
    names = {d["document_id"]: d["filename"] for d in _documents()}
    client, chunks, off = get_qdrant_client(), [], None
    while True:
        pts, off = client.scroll(COLLECTION_NAME, limit=256, offset=off, with_payload=True)
        chunks += [{**p.payload, "filename": names.get(p.payload["document_id"], "?")} for p in pts]
        if off is None:
            return chunks


def _documents():
    import sqlite3
    con = sqlite3.connect(registry.DB_PATH)
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute("select document_id, filename from documents")]


def resolve_gold(row, chunks):
    """Per evidence item, the set of chunk ids that contain its quote (an item is covered if any of them is retrieved)."""
    items = []
    for ev in row["evidence"]:
        ids = {c["chunk_id"] for c in chunks if c["filename"].startswith(ev["doc"]) and ev["quote"] in c["text"]}
        items.append(ids)
    return items


def rank_metrics(ranking, items):
    """items: list of sets (evidence items, each satisfied by any chunk in its set). Binary relevance."""
    gold = set().union(*items)
    out = {}
    for k in KS:
        top = set(ranking[:k])
        out[f"hit@{k}"] = float(bool(top & gold))
        out[f"recall@{k}"] = sum(bool(top & it) for it in items) / len(items)
    rr = next((1 / i for i, c in enumerate(ranking[:10], 1) if c in gold), 0.0)
    dcg = sum(1 / math.log2(i + 1) for i, c in enumerate(ranking[:10], 1) if c in gold)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), 10) + 1))
    out["mrr@10"], out["ndcg@10"] = rr, dcg / ideal if ideal else 0.0
    return out


def tokens(s):
    return [w.lower() for w in WORD.findall(s)]


def token_f1(pred, ref):
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    common = defaultdict(int)
    for w in r:
        common[w] += 1
    overlap = 0
    for w in p:
        if common[w] > 0:
            common[w] -= 1
            overlap += 1
    if not overlap:
        return 0.0
    pr, rc = overlap / len(p), overlap / len(r)
    return 2 * pr * rc / (pr + rc)


def key_fact_recall(answer, reference):
    """Fraction of numbers in the reference answer that appear in the answer (formatting-insensitive)."""
    norm = lambda s: s.replace(",", "").replace("±", "").replace("+", "").rstrip(".")
    nums = {norm(n) for n in NUM.findall(reference) if len(norm(n)) > 0}
    if not nums:
        return None
    flat = answer.replace(",", "")
    return sum(n in flat for n in nums) / len(nums)


INSUFFICIENT = re.compile(r"insufficient|not (?:contain|mention|provide|find|specif|state|include)|could not find|couldn't find|"
                          r"no (?:relevant )?(?:evidence|information|mention)|cannot (?:find|answer|determine)|unable to|"
                          r"does not (?:say|appear|list)|do not (?:contain|have)|not available", re.I)


async def retrieval_phase(svc, rows, chunks):
    out = {}
    for row in rows:
        q = prepare_query(row["question"])
        trace = Trace("eval")
        dense, lex = await asyncio.gather(svc.dense.retrieve(q, None, trace), svc.lexical.retrieve(q, None, trace))
        unique, _ = fusion.deduplicate(dense, lex)
        fused = fusion.rrf(unique, svc.s.rrf_k)
        ranked, _ = await asyncio.to_thread(svc.reranker.rerank, q, fused) if fused else ([], None)
        res = await svc.retrieve(q)
        sel_ids = [i.candidate.chunk_id for i in res.context.items]
        out[row["id"]] = {
            "dense": [c.chunk_id for c in dense], "bm25": [c.chunk_id for c in lex], "rrf": [c.chunk_id for c in fused],
            "hybrid_colbert": [c.chunk_id for c in ranked], "selected": sel_ids,
            "contexts": [{"chunk_id": i.candidate.chunk_id, "label": i.label, "text": i.candidate.text} for i in res.context.items],
            "evidence_sufficient": bool(res.selection and res.selection.evidence_sufficient),
            "selected_probs": [i.candidate.colbert_score for i in res.context.items],
        }
    return out


async def generation_phase(svc, s, rows, ret, chunks):
    store = SqliteConversationStore(Path("/tmp") / f"eval_rag_{int(time.time())}.db")
    orch = ChatOrchestrator(svc, s, store)
    await orch.run(ChatRequest(message="warm up"), "eval")
    res = {}
    for row in rows:
        rec = {}
        try:
            t0 = time.perf_counter()
            r = await orch.run(ChatRequest(message=row["question"], mode="rag"), "eval")
            rec = {"answer": r.answer, "abstained": r.metadata.abstained, "evidence_sufficient": r.metadata.evidence_sufficient,
                   "cited": [c.chunk_id for c in r.citations], "usage": r.metadata.usage,
                   "latency_ms": round((time.perf_counter() - t0) * 1e3, 1)}
        except InferenceError as exc:
            rec = {"error": exc.code, "answer": "", "abstained": False, "cited": []}
        res[row["id"]] = rec
        print(f"  gen {row['id']:<10} {rec.get('latency_ms', '-'):>8} ms  abstained={rec.get('abstained')}", flush=True)
    return res


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p / 100 * len(xs)))], 1) if xs else None


def aggregate(rows, golds, ret, gen):
    answerable = [r for r in rows if r.get("answerable", True) and r["evidence"]]
    report = {"n_questions": len(rows), "n_answerable": len(answerable), "retrieval": {}, "retrieval_by_category": {}}
    per_q = {}
    for sys_ in SYSTEMS:
        per = []
        for r in answerable:
            m = rank_metrics(ret[r["id"]][sys_], golds[r["id"]])
            per.append((r["category"], m))
            per_q.setdefault(r["id"], {})[sys_] = m
        report["retrieval"][sys_] = {k: mean(m[k] for _, m in per) for k in per[0][1]}
        if sys_ in ("hybrid_colbert", "selected"):
            cats = defaultdict(list)
            for c, m in per:
                cats[c].append(m)
            report["retrieval_by_category"][sys_] = {c: {"n": len(v), "hit@5": mean(x["hit@5"] for x in v), "recall@5": mean(x["recall@5"] for x in v),
                                                         "mrr@10": mean(x["mrr@10"] for x in v)} for c, v in sorted(cats.items())}
    # RAGAS-style context metrics on the selected context
    cp, cr = [], []
    for r in answerable:
        sel, items = ret[r["id"]]["selected"], golds[r["id"]]
        gold = set().union(*items)
        cp.append(sum(c in gold for c in sel) / len(sel) if sel else 0.0)
        cr.append(sum(bool(set(sel) & it) for it in items) / len(items))
    report["context"] = {"precision_selected": mean(cp), "recall_selected": mean(cr),
                         "mean_chunks_selected": mean(len(ret[r["id"]]["selected"]) for r in answerable)}
    # Sufficiency gate on unanswerable questions
    un = [r for r in rows if not r.get("answerable", True)]
    report["evidence_gate"] = {
        "answerable_flagged_sufficient": mean(float(ret[r["id"]]["evidence_sufficient"]) for r in answerable),
        "unanswerable_flagged_sufficient": mean(float(ret[r["id"]]["evidence_sufficient"]) for r in un)}
    if gen:
        g_ans = [r for r in answerable if r["id"] in gen]
        g_un = [r for r in un if r["id"] in gen]
        det = {}
        det["abstain_correct_unanswerable"] = mean(float(gen[r["id"]]["abstained"] or bool(INSUFFICIENT.search(gen[r["id"]]["answer"]))) for r in g_un)
        det["false_abstain_answerable"] = mean(float(gen[r["id"]]["abstained"]) for r in g_ans)
        det["token_f1_vs_reference"] = mean(token_f1(gen[r["id"]]["answer"], r["reference_answer"]) for r in g_ans)
        kf = [key_fact_recall(gen[r["id"]]["answer"], r["reference_answer"]) for r in g_ans]
        det["key_number_recall"] = mean(kf)
        cit_p, cit_r, cit_any = [], [], []
        for r in g_ans:
            cited, items = gen[r["id"]]["cited"], golds[r["id"]]
            gold = set().union(*items)
            if cited:
                cit_p.append(sum(c in gold for c in cited) / len(cited))
            cit_r.append(sum(bool(set(cited) & it) for it in items) / len(items))
            cit_any.append(float(bool(set(cited) & gold)))
        det.update(citation_precision=mean(cit_p), citation_recall=mean(cit_r), cited_any_gold=mean(cit_any),
                   answered_with_citation=mean(float(bool(gen[r["id"]]["cited"])) for r in g_ans))
        hi = [r for r in g_ans if r.get("language", "en") == "hi"]
        det["hindi_script_in_answer"] = mean(float(bool(DEVANAGARI.search(gen[r["id"]]["answer"]))) for r in hi)
        lat = [gen[r["id"]]["latency_ms"] for r in rows if "latency_ms" in gen.get(r["id"], {})]
        det["latency_ms"] = {"p50": pct(lat, 50), "p95": pct(lat, 95), "max": max(lat) if lat else None}
        det["mean_output_tokens"] = mean((gen[r["id"]].get("usage") or {}).get("output_tokens", 0) for r in rows if r["id"] in gen)
        det["errors"] = sum("error" in v for v in gen.values())
        report["generation"] = det
    return report, per_q


async def main(a):
    rows = [json.loads(l) for l in open(a.gold) if l.strip()]
    if a.only:
        rows = [r for r in rows if r["id"] in a.only.split(",")]
    if a.strip_scope:
        rows = [{**r, "question": strip_scope(r["question"])} for r in rows]
    registry.init_db()
    chunks = load_corpus()
    golds, bad = {}, []
    for r in rows:
        if r.get("answerable", True):
            golds[r["id"]] = resolve_gold(r, chunks)
            if any(not it for it in golds[r["id"]]):
                bad.append(r["id"])
    if bad:
        raise SystemExit(f"Gold evidence not found in the index for: {bad}. Re-ingest or fix the gold file.")
    svc = RetrievalService()
    await svc.retrieve("warm up")
    print(f"corpus: {len(chunks)} chunks, {len({c['document_id'] for c in chunks})} documents; {len(rows)} questions")
    ret = await retrieval_phase(svc, rows, chunks)
    gen = None
    if not a.no_generation:
        s = get_inference_settings()
        s = replace(s, base_url=a.base_url or s.base_url, model=a.model or s.model)
        gen = await generation_phase(svc, s, rows, ret, chunks)
    Path("eval_results").mkdir(exist_ok=True)  # raw outputs first: an aggregation bug must never cost a generation run
    Path(f"eval_results/rag_{a.tag}.raw.json").write_text(json.dumps({"retrieval": ret, "generation": gen}, ensure_ascii=False))
    report, per_q = aggregate(rows, golds, ret, gen)
    report.update(tag=a.tag, model=a.model or get_inference_settings().model, corpus_chunks=len(chunks))
    out = Path("eval_results")
    out.mkdir(exist_ok=True)
    (out / f"rag_{a.tag}.json").write_text(json.dumps({"report": report, "per_question": per_q}, indent=1, ensure_ascii=False))
    if gen:  # input for the judge step: everything a grader needs, nothing it does not
        with open(out / f"judge_{a.tag}.jsonl", "w") as f:
            for r in rows:
                g = gen.get(r["id"], {})
                f.write(json.dumps({"id": r["id"], "category": r["category"], "question": r["question"],
                                    "reference_answer": r["reference_answer"], "answerable": r.get("answerable", True),
                                    "answer": g.get("answer", ""), "abstained": g.get("abstained"),
                                    "contexts": [c["text"] for c in ret[r["id"]]["contexts"]],
                                    "gold_contexts": [c["text"] for c in chunks if c["chunk_id"] in set().union(*golds.get(r["id"], [set()]))]},
                                   ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="tests/eval/rag_gold.jsonl")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--only")
    ap.add_argument("--strip-scope", action="store_true", help="drop the leading 'In the X report,' clause from each question")
    ap.add_argument("--no-generation", action="store_true")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    asyncio.run(main(ap.parse_args()))
