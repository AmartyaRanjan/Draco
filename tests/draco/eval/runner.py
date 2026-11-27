"""Deterministic evaluation of Draco's control-plane behaviour.

Everything here is deterministic: no model, no network, no randomness (nothing to seed). What it
measures is the *runtime's* decisions (routing validation, tool authorization, termination
semantics). It does not measure answer quality, which is provider-dependent
and belongs to the application built on Draco (for example draco-rag's eval scripts).

Run:  PYTHONPATH=src:.:<site-packages> python -m tests.draco.eval.runner
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from typing import Any

from pydantic import BaseModel

CASES = pathlib.Path(__file__).parent / "cases"


def load(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (CASES / f"{name}.jsonl").read_text().splitlines() if line.strip()]


# ---- routing -----------------------------------------------------------------------------
async def eval_routing(case: dict[str, Any]) -> list[str]:
    from draco.agents import AgentRegistry, Router, RouteRequest
    from draco.core import DracoSettings, Handoff, Policy
    from draco.harness import Harness
    h = Harness(DracoSettings(default_timeout_s=5))
    reg = AgentRegistry(h)

    async def math(ctx, x): return {"answer": x["a"] + x["b"]} if isinstance(x, dict) else {"answer": 5}
    async def writer(ctx, x): return {"text": "w"}
    async def triage(ctx, x): return Handoff("math", "numbers", {"a": 2, "b": 3})
    async def bad(ctx, x): return Handoff("does-not-exist")
    async def ping(ctx, x): return Handoff("pong")
    async def pong(ctx, x): return Handoff("ping")
    for n, f, c in (("math", math, {"calc"}), ("writer", writer, {"write", "summarize"}), ("triage", triage, {"triage"}),
                    ("bad", bad, set()), ("ping", ping, set()), ("pong", pong, set())):
        reg.register(n, f, capabilities=c)
    router = Router(h, reg)
    policy = Policy(allowed_agents=frozenset(case["allowed_agents"])) if "allowed_agents" in case else None
    r = await router.run(RouteRequest(case["input"], agent=case.get("agent"), require=frozenset(case.get("require", []))),
                         policy=policy)
    errs = []
    if r.status.value != case["expect_status"]:
        errs.append(f"status {r.status.value} != {case['expect_status']}")
    if "expect_path" in case and (r.output or {}).get("path") != case["expect_path"]:
        errs.append(f"path {(r.output or {}).get('path')} != {case['expect_path']}")
    if "expect_error" in case and (r.error is None or r.error.code != case["expect_error"]):
        errs.append(f"error {r.error and r.error.code} != {case['expect_error']}")
    return errs


# ---- tools -------------------------------------------------------------------------------
class _Args(BaseModel):
    a: int = 0
    b: int = 0


async def eval_tools(case: dict[str, Any]) -> list[str]:
    from draco.core import DracoSettings, ExecutionContext, Policy
    from draco.harness import Harness
    from draco.tools import CapabilityRegistry, NoArgs, ToolRuntime, native_tool
    reg = CapabilityRegistry()

    @native_tool(reg, name="add", description="add", input_model=_Args)
    async def add(args, ctx): return {"sum": args.a + args.b}

    @native_tool(reg, name="write_db", description="w", input_model=NoArgs, permissions={"write"}, mutates_state=True)
    async def write_db(args, ctx): return {"ok": True}

    @native_tool(reg, name="send_email", description="e", input_model=NoArgs, permissions={"write"}, requires_approval=True)
    async def send_email(args, ctx): return {}
    h = Harness(DracoSettings())
    allowed = None if case["allowed"] is None else frozenset(case["allowed"])
    ctx = ExecutionContext(services=h.services, policy=Policy(allowed_tools=allowed, permissions=frozenset(case["perms"])))
    rt = ToolRuntime(reg)
    if "expose" in case:
        got = sorted(s["function"]["name"] for s in rt.specs(ctx))
        return [] if got == sorted(case["expose"]) else [f"exposed {got} != {sorted(case['expose'])}"]
    res = await rt.call(ctx, case["call"], case["args"])
    got = "success" if res.status == "success" else res.error.code.value
    return [] if got == case["expect"] else [f"{got} != {case['expect']}"]


# ---- termination -------------------------------------------------------------------------
async def eval_termination(case: dict[str, Any]) -> list[str]:
    from draco.core import BudgetLimits, DracoSettings, Handoff, Waiting
    from draco.harness import Harness, call_model
    h = Harness(DracoSettings(default_timeout_s=5))
    sc = case["scenario"]

    async def llm(): return 1

    async def w(ctx, x):
        if sc == "raise": raise ValueError("x")
        if sc == "sleep": await asyncio.sleep(10)
        if sc == "models":
            for _ in range(5): await call_model(ctx, llm)
        if sc == "tools":
            for _ in range(5): ctx.budget.charge_tool_call()
        if sc == "recurse":
            return (await h.run("w", x, parent=ctx)).to_dict()
        if sc == "handoff": return Handoff("elsewhere")
        if sc == "wait": return Waiting("approval")
        return "ok"
    h.register("w", w)
    limits = BudgetLimits(**case["limits"]) if "limits" in case else None
    t = asyncio.create_task(h.run("w", None, timeout_s=case.get("timeout_s"), limits=limits))
    if "cancel_after" in case:
        await asyncio.sleep(case["cancel_after"])
        for eid in list(h._inflight):
            h.cancel(eid)
    r = await t
    errs = [] if r.status.value == case["expect"] else [f"status {r.status.value} != {case['expect']}"]
    if "expect_output_contains" in case and case["expect_output_contains"] not in json.dumps(r.output):
        errs.append("output missing marker")
    if h._inflight:
        errs.append("execution leaked in-flight state")
    return errs


SUITES = {"routing": eval_routing, "tools": eval_tools, "termination": eval_termination}


async def run_suite(name: str) -> dict[str, Any]:
    results = []
    for case in load(name):
        try:
            errs = await SUITES[name](case)
        except Exception as exc:  # a crash is a failed case, reported with its type
            errs = [f"crashed: {type(exc).__name__}: {exc}"]
        results.append({"id": case["id"], "ok": not errs, "errors": errs})
    passed = sum(r["ok"] for r in results)
    return {"suite": name, "passed": passed, "total": len(results), "failures": [r for r in results if not r["ok"]]}


async def run_all() -> list[dict[str, Any]]:
    return [await run_suite(n) for n in SUITES]


if __name__ == "__main__":
    out = asyncio.run(run_all())
    for s in out:
        print(f"{s['suite']:<12} {s['passed']}/{s['total']}")
        for f in s["failures"]:
            print("   FAIL", f["id"], f["errors"])
    sys.exit(0 if all(s["passed"] == s["total"] for s in out) else 1)
