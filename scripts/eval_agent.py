"""End-to-end evaluation of the agent layer against the live stack (vLLM, Qdrant, OpenSearch, BGE-M3).

  python scripts/eval_agent.py --tag run1 [--probe] [--only id1,id2]

Writes eval_results/agent_<tag>.json with per-case checks, latency, token usage, applied reasoning mode,
tool traces and context-budget snapshots. Not run in CI (needs the services); unit tests cover the logic.
"""
import argparse
import asyncio
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.agent.catalog import build_registry  # noqa: E402
from app.agent.config import get_agent_settings  # noqa: E402
from app.agent.graph import AgentRunner  # noqa: E402
from app.agent.schemas import AgentChatRequest  # noqa: E402
from app.agent.wiring import build_services  # noqa: E402
from app.db import registry  # noqa: E402
from app.inference.config import get_inference_settings  # noqa: E402
from app.retrieval.service import get_service  # noqa: E402


def check(case, r) -> dict:
    e, out = case["expect"], {}
    tools = [t["tool_name"] for t in r["tool_trace"]]
    ans = r["answer"]
    if "intent" in e: out["intent"] = r["metadata"]["intent"] == e["intent"]
    if "tools" in e: out["tools_called"] = all(t in tools for t in e["tools"])
    if "forbidden_tools" in e: out["forbidden_tools_absent"] = not any(t in tools for t in e["forbidden_tools"])
    if "status" in e: out["status"] = r["status"] in e["status"]
    if "contains" in e: out["contains"] = all(c in ans for c in e["contains"])
    if "min_citations" in e: out["citations"] = len(r["citations"]) >= e["min_citations"]
    if "min_artifacts" in e: out["artifacts"] = len(r["artifacts"]) >= e["min_artifacts"]
    if "max_llm_calls" in e: out["llm_calls"] = r["metadata"]["usage"]["llm_calls"] <= e["max_llm_calls"]
    if "warn_contains" in e: out["warning"] = any(e["warn_contains"] in w for w in r["warnings"])
    if "mode_requested" in e: out["mode_requested"] = r["reasoning"][0]["requested"] == e["mode_requested"]
    if "mode_applied" in e: out["mode_applied"] = any(d["applied"] == e["mode_applied"] for d in r["reasoning"])
    if "compaction_rung" in e: out["compaction"] = any(c.get("rung") == e["compaction_rung"] for c in r["context"]["compaction"])
    if e.get("abstains_or_declines"):
        out["declines"] = r["status"] == "abstained" or any(k in ans.lower() for k in ("insufficient", "not contain", "could not find", "no evidence", "do not"))
    out["answer_sane"] = bool(ans.strip()) and "INTERNAL NOTES" not in ans and not any(t in ans for t in tools if t.startswith(("generate_", "run_", "compose_")))
    refs = {c["ref"] for c in r["citations"]}
    out["cited_labels_resolve"] = all(m in refs for m in __import__("re").findall(r"\[([SW]\d+)\]", ans))
    if r["validation"]:
        out["no_invalid_refs"] = not r["validation"]["invalid_refs"]
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cases", default=str(ROOT / "tests/eval/agent_eval.jsonl"))
    ap.add_argument("--only")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    registry.init_db()
    inf, retrieval = get_inference_settings(), get_service()
    svc, adapter = build_services(inf, retrieval)
    if a.probe:
        print("probe:", await adapter.probe())
    cases = [json.loads(l) for l in open(a.cases) if l.strip()]
    if a.only:
        cases = [c for c in cases if c["id"] in a.only.split(",")]
    results = []
    for c in cases:
        s = dataclasses.replace(svc.settings, **c.get("settings", {}))
        svc.settings = s
        runner = AgentRunner(svc, build_registry(), adapter, inf, None)
        req = AgentChatRequest(message=c["message"], allow_web=c.get("allow_web", False), reasoning=c.get("reasoning", "auto"))
        t0, rec = time.perf_counter(), {"id": c["id"], "message": c["message"]}
        try:
            r = (await runner.run(req, "eval")).model_dump()
            rec.update(ok=True, latency_ms=round((time.perf_counter() - t0) * 1e3), checks=check(c, r), status=r["status"],
                       answer=r["answer"][:600], citations=[(x["ref"], x["document_id"], x["chunk_id"], x["url"]) for x in r["citations"]],
                       tools=[(t["tool_name"], t["status"], round(t["duration_ms"])) for t in r["tool_trace"]],
                       reasoning=r["reasoning"], usage=r["metadata"]["usage"], warnings=r["warnings"],
                       compaction=r["context"]["compaction"], budget=[(b["label"], b["used"], b["evidence_limit"]) for b in r["context"]["budget"]],
                       validation=(r["validation"] or {}).get("status"), artifacts=[x["filename"] for x in r["artifacts"]],
                       node_ms={n["node"]: n["ms"] for n in r["metadata"]["nodes"]}, token_counts=r["metadata"]["token_counts"])
        except Exception as exc:
            rec.update(ok=False, latency_ms=round((time.perf_counter() - t0) * 1e3), error=f"{type(exc).__name__}: {exc}")
        results.append(rec)
        passed = all(rec.get("checks", {}).values()) if rec["ok"] else False
        print(f"{'PASS' if passed else 'FAIL'} {c['id']:<18} {rec['latency_ms']:>7} ms  {rec.get('status', rec.get('error'))}")
        if not passed and rec["ok"]:
            print("     failed checks:", [k for k, v in rec["checks"].items() if not v])
    lat = [r["latency_ms"] for r in results if r["ok"]]
    summary = {"cases": len(results), "passed": sum(all(r.get("checks", {}).values()) for r in results if r["ok"]),
               "latency_ms_p50": statistics.median(lat) if lat else None, "latency_ms_max": max(lat) if lat else None,
               "model": inf.model, "tag": a.tag}
    out = ROOT / "eval_results" / f"agent_{a.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=1, default=str))
    print(json.dumps(summary))


asyncio.run(main())
