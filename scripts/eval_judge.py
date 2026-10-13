"""Aggregate LLM-judge verdicts into RAGAS / TruLens / DeepEval-family metrics.

  python scripts/eval_judge.py --tag as_written [--verdicts eval_results/judge_as_written.verdicts.jsonl]

Verdict rows (one per question, written by the judge from judge_<tag>.jsonl) carry atomic judgements only; every metric
is computed here, deterministically, from those judgements, so the definitions are in code and not in a prompt:

  faithfulness         RAGAS: supported answer claims / answer claims (answerable, non-abstained answers)
  answer_correctness   0 / 0.5 / 1 against the reference (unanswerable: 1 only if the answer declines)
  answer_relevancy     judge score 0-1: does the answer address the question completely and without padding
  context_precision    RAGAS: average precision over the ranked contexts, from per-context relevance flags
  context_recall       RAGAS: reference claims attributable to the contexts / reference claims
  groundedness         TruLens RAG-triad: same quantity as faithfulness, reported under its TruLens name
  context_relevance    TruLens RAG-triad: share of retrieved contexts relevant to the question
  hallucination_rate   DeepEval: share of answers with at least one claim contradicted by or absent from the contexts
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def average_precision(flags):
    hits, total = 0, 0.0
    for i, f in enumerate(flags, 1):
        if f:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def main(a):
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(f"eval_results/judge_{a.tag}.jsonl")}
    verd = {}
    for l in open(a.verdicts or f"eval_results/judge_{a.tag}.verdicts.jsonl"):
        if l.strip():
            v = json.loads(l)
            verd[v["id"]] = v
    missing = sorted(set(meta) - set(verd))
    rows = []
    for i, m in meta.items():
        v = verd.get(i)
        if not v:
            continue
        ans_ok = m["answerable"]
        r = {"id": i, "category": m["category"], "answerable": ans_ok, "abstained": bool(m["abstained"])}
        r["correctness"] = v.get("correctness")
        r["relevancy"] = v.get("relevancy")
        if ans_ok and not m["abstained"] and v.get("claims_total"):
            r["faithfulness"] = v["claims_supported"] / v["claims_total"]
            r["hallucinated"] = float(v["claims_supported"] < v["claims_total"])
        if ans_ok and v.get("context_relevance") is not None:
            fl = v["context_relevance"]
            r["context_precision"] = average_precision(fl)
            r["context_relevance"] = sum(fl) / len(fl) if fl else 0.0
        if ans_ok and v.get("ref_claims_total"):
            r["context_recall"] = v["ref_claims_in_contexts"] / v["ref_claims_total"]
        rows.append(r)
    keys = ["correctness", "relevancy", "faithfulness", "hallucinated", "context_precision", "context_relevance", "context_recall"]
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    summary = {"n_judged": len(rows), "n_missing_verdicts": len(missing), "answerable": {k: avg(r.get(k) for r in ans) for k in keys},
               "unanswerable": {"correct_decline_rate": avg(r["correctness"] for r in un), "n": len(un)}}
    summary["answerable"]["hallucination_rate"] = summary["answerable"].pop("hallucinated")
    summary["answerable"]["groundedness"] = summary["answerable"]["faithfulness"]
    by = defaultdict(list)
    for r in ans:
        by[r["category"]].append(r)
    summary["by_category"] = {c: {"n": len(v), **{k: avg(x.get(k) for x in v) for k in ("correctness", "faithfulness", "context_recall")}} for c, v in sorted(by.items())}
    summary["worst_answers"] = [{"id": r["id"], "correctness": r["correctness"]} for r in sorted(ans, key=lambda r: (r["correctness"] is None, r["correctness"] or 0))[:8]]
    Path(f"eval_results/judged_{a.tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--verdicts")
    main(ap.parse_args())
