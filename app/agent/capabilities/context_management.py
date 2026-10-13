"""Group I - context_management. summarize_evidence is invoked by the orchestrator when the
deterministic budget manager says compaction is needed; the model is not asked to remember to."""
from app.agent.capabilities.common import require
from app.agent.contracts import CapabilityError, Empty, ErrorCode, Outcome
from app.agent.registry import Capability
from app.agent.summarize import SummarizeInput, SummarizeOutput


async def summarize_evidence(a: SummarizeInput, ctx):
    svc = ctx.services
    summarizer = require(svc.summarizer, "Evidence summarizer")
    store = ctx.evidence
    refs = a.refs if a.refs is not None else [i.ref for i in store.raw()]
    unknown = [r for r in refs if store.get(r) is None]
    if unknown:
        raise CapabilityError(ErrorCode.invalid_input, f"Unknown evidence refs: {unknown[:5]}")
    items = [store.get(r) for r in refs]
    if not items and not a.prior_summaries:
        raise Empty({"summary": {"claims": [], "contradictions": [], "gaps": []}, "rendered": "", "sources": {},
                     "method": "none", "status": "ok", "final_tokens": 0, "target_tokens": a.target_tokens,
                     "within_budget": True, "levels": 0, "usage": {}, "coverage": {"input_items": 0}})
    out: SummarizeOutput = await summarizer.summarize(a, items)
    if out.status == "over_budget":
        raise CapabilityError(ErrorCode.limit_exceeded,
                              f"Summary needs {out.final_tokens} tokens but the target is {out.target_tokens}; "
                              "reduce the scope or raise the budget.")
    return Outcome(out, extra={"summary_tokens": out.final_tokens, "method": out.method, "usage": out.usage,
                               "uncovered": len(out.coverage.get("uncovered_refs", []))})


CAPS = [Capability("summarize_evidence", "context_management",
                   "Compress accumulated evidence into claims with exact source refs, contradictions and gaps.",
                   SummarizeInput, SummarizeOutput, summarize_evidence, cost="slow", llm_exposed=False,
                   timeout_s=120, max_output_chars=40_000)]
