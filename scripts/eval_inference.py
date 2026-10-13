"""Evaluate the chat pipeline against tests/eval/chat_eval.jsonl using the real retrieval stack and vLLM.

PYTHONPATH=. python scripts/eval_inference.py --tag fp8 [--base-url URL --model NAME --api-key KEY]
Writes eval_results/<tag>.json and eval_results/<tag>.ragas.jsonl (inputs for RAGAS: user_input,
response, retrieved_contexts; RAGAS itself is not run here). Checks are only those declared in the
dataset's `expect` block; nothing is judged by an LLM.
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from app.inference.config import get_inference_settings
from app.inference.generation import ChatOrchestrator
from app.inference.memory import MemoryItem, SqliteConversationStore
from app.inference.schemas import ChatRequest, InferenceError
from app.retrieval.service import RetrievalService

INSUFFICIENT = re.compile(r"insufficient|not (?:contain|mention|provide|find)|could not find|no (?:relevant )?(?:evidence|information)|cannot|can't|unable|don't|do not", re.I)


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else -1


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p / 100 * len(xs)))], 1) if xs else None


class FixedMemory:
    def __init__(self, texts):
        self.texts = texts

    async def search(self, user_id, query, limit):
        return [MemoryItem(f"m{i}", t) for i, t in enumerate(self.texts)][:limit]


async def main(a):
    s = get_inference_settings()
    s = replace(s, base_url=a.base_url or s.base_url, model=a.model or s.model, api_key=a.api_key or s.api_key)
    retrieval = RetrievalService()
    tmp = tempfile.mkdtemp()
    store = SqliteConversationStore(Path(tmp) / "eval.db")
    rows = [json.loads(l) for l in open(a.dataset) if l.strip()]
    results, ragas = [], []
    await ChatOrchestrator(retrieval, s, store).run(ChatRequest(message="warm up"), "eval")  # exclude cold start
    gpu_before, gpu_peak = gpu_used_mib(), 0
    for row in rows:
        orch = ChatOrchestrator(retrieval, s, store, FixedMemory(row.get("memories", [])) if row.get("memories") else None)
        exp, conv = row.get("expect", {}), None
        rec = {"id": row["id"], "category": row["category"], "checks": {}}
        try:
            for setup in row.get("setup", []):
                conv = (await orch.run(ChatRequest(message=setup, conversation_id=conv), "eval")).metadata.conversation_id
            t0 = time.perf_counter()
            r = await orch.run(ChatRequest(message=row["message"], mode=row.get("mode", "rag"), conversation_id=conv,
                                           document_ids=row.get("document_ids"), thinking=row.get("thinking")), "eval")
            rec["latency_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        except InferenceError as exc:
            rec.update(error=exc.code)
            rec["checks"]["completed"] = False
            results.append(rec)
            continue
        gpu_peak = max(gpu_peak, gpu_used_mib())
        ans, tools = r.answer, [t.tool_name for t in r.tool_trace]
        llm = [v for k, v in r.metadata.timings_ms.items() if k.startswith("llm_call_")]
        rec.update(answer_chars=len(ans), citations=len(r.citations), tools=tools, usage=r.metadata.usage,
                   ttft_ms=llm[0].get("ttft_ms") if llm else None, llm_calls=r.metadata.llm_calls,
                   abstained=r.metadata.abstained, evidence_sufficient=r.metadata.evidence_sufficient)
        c = rec["checks"]
        c["completed"] = bool(ans)
        if "contains_any" in exp:
            c["contains_expected"] = any(k.lower() in ans.lower() for k in exp["contains_any"])
        if exp.get("abstain") is False:
            c["answered"] = not r.metadata.abstained
        if exp.get("abstain_or_insufficient"):
            c["declined"] = r.metadata.abstained or bool(INSUFFICIENT.search(ans))
        if "min_citations" in exp:
            c["min_citations"] = len(r.citations) >= exp["min_citations"]
        if exp.get("language") == "hi":
            c["devanagari"] = bool(re.search("[ऀ-ॿ]", ans))
        if "tool" in exp:
            c["tool_selection"] = (exp["tool"] in tools) if exp["tool"] else not tools
        if tools:
            c["tool_args_valid"] = all(t.status != "invalid_arguments" for t in r.tool_trace)
        if r.citations and not r.metadata.abstained:
            c["citations_resolve"] = all(x.chunk_id and x.document_id for x in r.citations)
        results.append(rec)
        ragas.append({"user_input": row["message"], "response": ans, "id": row["id"], "citations": [x.chunk_id for x in r.citations]})
    checks = {}
    for rec in results:
        for k, v in rec["checks"].items():
            checks.setdefault(k, []).append(v)
    lat = [r["latency_ms"] for r in results if "latency_ms" in r]
    ttft = [r["ttft_ms"] for r in results if r.get("ttft_ms")]
    out = {"tag": a.tag, "model": s.model, "cases": len(rows),
           "check_pass_rate": {k: f"{sum(v)}/{len(v)}" for k, v in checks.items()},
           "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95)}, "ttft_ms": {"p50": pct(ttft, 50), "p95": pct(ttft, 95)},
           "mean_output_tokens": round(statistics.mean([(r.get("usage") or {}).get("output_tokens", 0) for r in results]), 1),
           "gpu_used_mib": {"before": gpu_before, "peak_after_requests": gpu_peak}, "results": results}
    Path("eval_results").mkdir(exist_ok=True)
    Path(f"eval_results/{a.tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    Path(f"eval_results/{a.tag}.ragas.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in ragas))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tests/eval/chat_eval.jsonl")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key")
    asyncio.run(main(ap.parse_args()))
