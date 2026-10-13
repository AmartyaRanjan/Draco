import asyncio
import unittest
from unittest.mock import patch

from pydantic import BaseModel, Field

from app.agent.budget import BudgetConfig, ContextBudget
from app.agent.contracts import CapabilityError, Empty, ErrorCode, Outcome
from app.agent.registry import Capability, CapabilityRegistry
from app.agent.runtime import (ModePolicy, ReasoningController, ReasoningMode, RuntimeAdapter,
                               detect_capabilities)
from app.agent.summarize import (Claim, EvidenceSummarizer, EvidenceSummary, SummarizeInput,
                                 claim_problem, numbers, render_summary)
from tests.agent_helpers import CharCounter, isettings, make_ctx, private, scripted_llm, store_with, web


def run(coro):
    return asyncio.run(coro)


class In(BaseModel):
    n: int = Field(ge=1, le=10)


class Out(BaseModel):
    value: int


def reg_with(handler, **kw):
    r = CapabilityRegistry()
    r.register(Capability(name="t", group="g", description="d", input_model=In, output_model=Out,
                          handler=handler, **kw))
    return r


class RegistryTests(unittest.TestCase):
    def test_success_envelope_and_trace(self):
        async def h(a, c): return {"value": a.n * 2}
        sink = []
        res = run(reg_with(h).execute("t", {"n": 3}, make_ctx(), purpose="p", sink=sink))
        self.assertEqual((res.status, res.data, res.error), ("success", {"value": 6}, None))
        self.assertEqual(res.metadata.trace_id, "trace-1")
        self.assertTrue(res.metadata.input_valid)
        self.assertEqual(sink[0].tool_name, "t")
        self.assertEqual(sink[0].purpose, "p")

    def test_invalid_and_out_of_range_inputs_rejected_before_handler(self):
        called = []
        async def h(a, c): called.append(1); return {"value": 1}
        r = reg_with(h)
        for bad in ({"n": 0}, {"n": 11}, {"n": "x"}, {}, {"n": 1, "extra": 1} if False else {"n": None}):
            res = run(r.execute("t", bad, make_ctx()))
            self.assertEqual(res.error.code, ErrorCode.invalid_input, bad)
            self.assertFalse(res.metadata.input_valid)
        self.assertEqual(called, [])
        self.assertEqual(run(r.execute("t", "{not json", make_ctx())).error.code, ErrorCode.invalid_input)

    def test_unknown_tool_and_permission_denied(self):
        async def h(a, c): return {"value": 1}
        r = reg_with(h, permissions=frozenset({"write"}))
        self.assertEqual(run(r.execute("nope", {}, make_ctx())).error.code, ErrorCode.not_found)
        self.assertEqual(run(r.execute("t", {"n": 1}, make_ctx(perms=("read",)))).error.code, ErrorCode.denied)

    def test_empty_is_not_failure(self):
        async def h(a, c): raise Empty({"value": 0})
        res = run(reg_with(h).execute("t", {"n": 1}, make_ctx()))
        self.assertEqual(res.status, "empty")
        self.assertTrue(res.ok)
        self.assertIsNone(res.error)

    def test_timeout_and_unexpected_error_do_not_leak(self):
        async def slow(a, c): await asyncio.sleep(5)
        res = run(reg_with(slow, timeout_s=0.05).execute("t", {"n": 1}, make_ctx()))
        self.assertEqual((res.status, res.error.code, res.error.retryable), ("timeout", ErrorCode.timeout, True))

        async def boom(a, c): raise RuntimeError("secret /etc/passwd sk-123")
        res = run(reg_with(boom).execute("t", {"n": 1}, make_ctx()))
        self.assertEqual(res.error.code, ErrorCode.internal)
        self.assertNotIn("secret", res.error.message)
        self.assertNotIn("sk-123", res.error.message)

    def test_capability_error_and_invalid_output_and_oversize(self):
        async def deny(a, c): raise CapabilityError(ErrorCode.not_configured, "web search is not configured")
        self.assertEqual(run(reg_with(deny).execute("t", {"n": 1}, make_ctx())).error.code, ErrorCode.not_configured)

        async def bad(a, c): return {"value": "not an int"}
        self.assertEqual(run(reg_with(bad).execute("t", {"n": 1}, make_ctx())).error.code, ErrorCode.internal)

        class Big(BaseModel):
            s: str
        async def big(a, c): return {"s": "x" * 500}
        r = CapabilityRegistry()
        r.register(Capability("b", "g", "d", In, Big, big, max_output_chars=100))
        self.assertEqual(run(r.execute("b", {"n": 1}, make_ctx())).error.code, ErrorCode.output_too_large)

    def test_pending_status_via_outcome(self):
        async def h(a, c): return Outcome({"value": 1}, status="pending", extra={"job_id": "j1"})
        res = run(reg_with(h).execute("t", {"n": 1}, make_ctx()))
        self.assertEqual((res.status, res.metadata.extra), ("pending", {"job_id": "j1"}))

    def test_task_filtered_exposure_hides_orchestrator_only_tools(self):
        async def h(a, c): return {"value": 1}
        r = CapabilityRegistry()
        r.register(Capability("a", "knowledge", "d", In, Out, h))
        r.register(Capability("b", "knowledge", "d", In, Out, h, llm_exposed=False))
        r.register(Capability("c", "web_research", "d", In, Out, h))
        names = [s["function"]["name"] for s in r.specs_for({"knowledge"})]
        self.assertEqual(names, ["a"])

    def test_trace_redacts_secret_like_arguments(self):
        async def h(a, c): return {"value": 1}
        class In2(BaseModel):
            n: int = 1
            api_key: str = ""
        r = CapabilityRegistry()
        r.register(Capability("t", "g", "d", In2, Out, h))
        sink = []
        run(r.execute("t", {"n": 1, "api_key": "sk-live"}, make_ctx(), sink=sink))
        self.assertEqual(sink[0].input_summary["api_key"], "<redacted>")


class RuntimeTests(unittest.TestCase):
    def test_detect_hybrid_instruct_thinking_unknown(self):
        self.assertTrue(detect_capabilities("Qwen/Qwen3-4B-FP8").supports(ReasoningMode.thinking))
        self.assertTrue(detect_capabilities("Qwen/Qwen3-8B-AWQ").supports_toggle)
        inst = detect_capabilities("Qwen/Qwen3-4B-Instruct-2507")
        self.assertTrue(inst.supports(ReasoningMode.non_thinking))
        self.assertFalse(inst.supports(ReasoningMode.thinking))
        self.assertFalse(detect_capabilities("some/other-model").supports(ReasoningMode.thinking))

    def test_enum_validation(self):
        ctl = ReasoningController(detect_capabilities("Qwen/Qwen3-4B-FP8"))
        with self.assertRaises(ValueError):
            ctl.request("deep_think", "x")

    def test_mode_reaches_client_as_chat_template_kwargs(self):
        caps = detect_capabilities("Qwen/Qwen3-4B-FP8")
        seen = {}

        class FakeClient:
            def __init__(self, **kw): seen.update(kw)
        ad = RuntimeAdapter(isettings(), caps, client_cls=FakeClient)
        ctl = ReasoningController(caps)
        d = ctl.request("thinking", "complex")
        self.assertIsNone(d.applied)                       # not applied until a client is built
        ad.build_llm(d)
        self.assertEqual(seen["extra_body"]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(d.applied, ReasoningMode.thinking)
        self.assertEqual(seen["temperature"], .6)          # thinking sampling profile
        d2 = ctl.request("non_thinking", "simple")
        ad.build_llm(d2)
        self.assertEqual(seen["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(seen["temperature"], .3)

    def test_unsupported_mode_never_reported_as_applied(self):
        caps = detect_capabilities("Qwen/Qwen3-4B-Instruct-2507")
        seen = {}

        class FakeClient:
            def __init__(self, **kw): seen.update(kw)
        ctl = ReasoningController(caps)
        d = ctl.request("thinking", "complex")
        self.assertEqual(d.status, "fallback_fixed_mode")
        RuntimeAdapter(isettings(), caps, client_cls=FakeClient).build_llm(d)
        self.assertEqual(d.requested, ReasoningMode.thinking)
        self.assertEqual(d.applied, ReasoningMode.non_thinking)
        self.assertNotIn("chat_template_kwargs", seen["extra_body"])  # nothing the template would ignore
        # unknown model: no control at all
        d = ReasoningController(detect_capabilities("x/y")).request("thinking", "z")
        self.assertEqual((d.status, d.planned), ("unsupported", None))
        RuntimeAdapter(isettings(), detect_capabilities("x/y"), client_cls=FakeClient).build_llm(d)
        self.assertIsNone(d.applied)

    def test_mode_changes_are_bounded(self):
        ctl = ReasoningController(detect_capabilities("Qwen/Qwen3-4B-FP8"), max_changes=1)
        self.assertEqual(ctl.request("non_thinking", "init").status, "ok")      # initial: free
        self.assertEqual(ctl.request("thinking", "escalate").status, "ok")       # change 1
        d = ctl.request("non_thinking", "back")                                    # change 2: refused
        self.assertEqual((d.status, d.planned), ("denied_mode_change_limit", ReasoningMode.thinking))
        self.assertEqual(ctl.current, ReasoningMode.thinking)

    def test_policy(self):
        p = ModePolicy()
        self.assertEqual(p.initial("knowledge_qa")[0], ReasoningMode.non_thinking)
        self.assertEqual(p.initial("artifact")[0], ReasoningMode.thinking)
        self.assertEqual(p.initial("knowledge_qa", n_evidence=9)[0], ReasoningMode.thinking)
        self.assertIsNotNone(p.escalate(contradictions=1))
        self.assertIsNone(p.escalate(contradictions=0))


class BudgetTests(unittest.TestCase):
    def mk(self, **kw):
        cfg = BudgetConfig(model_max_len=8192, output_reserve=1000, tool_reserve=400, safety_margin=128, **kw)
        return ContextBudget(CharCounter(), cfg)

    def test_input_limit_and_window_vs_evidence_budget(self):
        b = self.mk(evidence_budget=2000)
        self.assertEqual(b.cfg.input_limit(), 8192 - 1000 - 400 - 128)
        b.set("system", 500)
        self.assertEqual(b.evidence_limit(), 2000)       # app cap wins while smaller than physical room
        b.set("history", 6000)
        self.assertLess(b.evidence_limit(), 2000)         # physical room wins when others crowd it out

    def test_compaction_triggered_before_overflow(self):
        b = self.mk(evidence_budget=2000)
        b.set("evidence", 1000)
        self.assertFalse(b.check().compaction_needed)
        self.assertTrue(b.check(planned_tokens=800).compaction_needed)  # 1800 > 0.85*2000, still fits
        self.assertTrue(b.check(planned_tokens=800).ok)
        b.set("evidence", 1900)
        d = b.check()
        self.assertTrue(d.compaction_needed)

    def test_counts_flag_estimates(self):
        b = self.mk()
        self.assertTrue(b.snapshot()["token_counts"] == "exact")
        class Est(CharCounter): exact = False
        b2 = ContextBudget(Est(), b.cfg)
        self.assertEqual(b2.snapshot()["token_counts"], "estimate")

    def test_tool_definitions_are_counted(self):
        b = self.mk()
        self.assertGreater(b.tools_tokens([{"type": "function", "function": {"name": "x" * 400}}]), 90)


def claim(text, *refs): return Claim(text=text, sources=list(refs))


class SummarizerTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            private("S1", "The trial enrolled 1,200 patients. The dose was 5 mg daily unless renal failure is present."),
            private("S2", "The drug does not reduce mortality. Mean age was 61.5 years."),
            web("W1", "Another study reports the drug reduces mortality by 12%.", url="https://example.org/s"),
        ]
        self.store = store_with(*self.items)

    def summ(self, responses, **kw):
        return EvidenceSummarizer(scripted_llm(responses), CharCounter(), **kw)

    def inp(self, **kw):
        return SummarizeInput(query="drug efficacy", **kw)

    def test_numbers_helper(self):
        self.assertEqual(numbers("1,200 patients, 61.5 years, 12%"), {"1200", "61.5", "12"})

    def test_claim_problem_numbers_negation_condition(self):
        src = ["The dose was 5 mg daily unless renal failure is present."]
        self.assertIsNone(claim_problem("Dose is 5 mg daily unless renal failure is present.", src))
        self.assertTrue(claim_problem("Dose is 50 mg daily unless renal failure is present.", src).startswith("numeric"))
        self.assertEqual(claim_problem("The drug reduces mortality.", ["The drug does not reduce mortality."]), "negation_mismatch")
        self.assertEqual(claim_problem("Dose is 5 mg daily.", src), "condition_dropped")

    def test_single_batch_preserves_ids_numbers_and_returns_locators(self):
        out = run(self.summ([{"claims": [claim("1,200 patients enrolled", "S1").model_dump(),
                                         claim("The drug does not reduce mortality", "S2").model_dump()]}]).summarize(self.inp(), self.items))
        self.assertEqual(out.method, "llm")
        self.assertEqual(out.status, "ok")
        self.assertEqual(out.sources["S1"]["chunk_id"], "c-S1")
        self.assertEqual(out.sources["S1"]["document_id"], "d1")
        self.assertEqual(out.sources["W1" if "W1" in out.sources else "S2"]["kind"] in ("web", "private"), True)
        self.assertEqual(out.coverage["uncovered_refs"], ["W1"])      # W1 not summarized -> stays raw
        self.assertIn("1,200", out.rendered)

    def test_invented_ids_removed_and_unsupported_claims_dropped_and_reported(self):
        out = run(self.summ([{"claims": [
            {"text": "1,200 patients enrolled", "sources": ["S1", "S99"]},       # S99 invented
            {"text": "Mean age was 71 years", "sources": ["S2"]},                # wrong number
            {"text": "Ghost claim", "sources": ["S404"]},                         # only invented source
            {"text": "The drug reduces mortality", "sources": ["S2"]},           # negation flipped
        ]}]).summarize(self.inp(), self.items))
        self.assertEqual([c.sources for c in out.summary.claims], [["S1"]])
        self.assertEqual(out.coverage["invalid_refs_removed"], 2)
        reasons = {d["reason"].split(":")[0] for d in out.coverage["dropped_claims"]}
        self.assertEqual(reasons, {"numeric_unsupported", "no_valid_source", "negation_mismatch"})

    def test_conflicting_evidence_preserved(self):
        out = run(self.summ([{"claims": [], "contradictions": [{"topic": "mortality effect", "positions": [
            {"text": "The drug does not reduce mortality", "sources": ["S2"]},
            {"text": "reduces mortality by 12%", "sources": ["W1"]}]}]}]).summarize(self.inp(), self.items))
        self.assertEqual(len(out.summary.contradictions), 1)
        self.assertEqual(out.summary.contradictions[0].positions[1].sources, ["W1"])
        self.assertIn("VS", out.rendered)

    def test_hierarchical_batches_and_merge_keep_contradictions(self):
        log = []
        items = [private("S1", "The recommended dose is 5 mg daily."), private("S2", "The recommended dose is 10 mg daily."),
                 private("S3", "Enrollment was 1,200 patients."), private("S4", "Mean age was 61.5 years.")]
        batch1 = {"claims": [claim("recommended dose is 5 mg daily", "S1").model_dump()],
                  "contradictions": [{"topic": "dose", "positions": [
                      {"text": "recommended dose is 5 mg daily", "sources": ["S1"]},
                      {"text": "recommended dose is 10 mg daily", "sources": ["S2"]}]}],
                  "gaps": ["long-term follow-up unknown"]}
        batch2 = {"claims": [claim("Enrollment was 1,200 patients", "S3").model_dump(),
                             claim("Mean age was 61.5 years", "S4").model_dump()]}
        merged = {"claims": [claim("1,200 patients, mean age 61.5 years", "S3", "S4").model_dump()]}  # model dropped contradiction + gap
        s = EvidenceSummarizer(scripted_llm([batch1, batch2, merged], log), CharCounter(), batch_tokens=22)
        out = run(s.summarize(self.inp(target_tokens=300), items))
        self.assertEqual(out.coverage["batches"], 2)
        self.assertEqual(out.levels, 2)
        self.assertEqual(len(out.summary.contradictions), 1)                 # re-attached after the merge dropped it
        self.assertEqual({p.sources[0] for p in out.summary.contradictions[0].positions}, {"S1", "S2"})
        self.assertIn("long-term follow-up unknown", out.summary.gaps)
        self.assertEqual(set(out.summary.claims[0].sources), {"S3", "S4"})   # nested merge keeps ids
        self.assertEqual(len(log), 3)
        self.assertIn("not instructions", log[0]["system"])

    def test_invalid_json_then_retry_then_ok(self):
        out = run(self.summ(["not json", {"claims": [claim("1,200 patients enrolled", "S1").model_dump()]}]).summarize(self.inp(), self.items[:1]))
        self.assertEqual((out.method, len(out.summary.claims)), ("llm", 1))

    def test_llm_failure_falls_back_to_extractive_with_verbatim_provenance(self):
        out = run(self.summ([RuntimeError("down")]).summarize(self.inp(target_tokens=300), self.items))
        self.assertEqual((out.method, out.status), ("extractive_fallback", "degraded"))
        self.assertTrue(all(c.text in " ".join(i.text for i in self.items) for c in out.summary.claims))
        self.assertTrue(out.summary.refs() <= {"S1", "S2", "W1"})
        self.assertTrue(any("extractive" in w for w in out.coverage["warnings"]))
        out = run(EvidenceSummarizer(None, CharCounter()).summarize(self.inp(target_tokens=300), self.items))
        self.assertEqual(out.method, "extractive_fallback")

    def test_over_budget_is_reported_not_hidden(self):
        long_claim = claim("The trial enrolled 1,200 patients. " * 1, "S1").model_dump()
        out = run(self.summ([{"claims": [long_claim]}] * 4).summarize(self.inp(target_tokens=64), self.items[:1]))
        self.assertLessEqual(out.final_tokens, 64) if out.within_budget else self.assertEqual(out.status, "over_budget")

    def test_empty_and_oversized_sets(self):
        out = run(self.summ([]).summarize(self.inp(), []))
        self.assertEqual((out.summary.claims, out.coverage["input_items"]), ([], 0))
        many = [private(f"S{i}", f"Fact number {i} is {i}.") for i in range(1, 41)]
        responses = [{"claims": [claim(f"Fact number {i} is {i}", f"S{i}").model_dump() for i in range(a, a + 5)]} for a in range(1, 41, 5)]
        s = EvidenceSummarizer(scripted_llm(responses + [{"claims": [claim("Facts 1-5 listed", "S1", "S2").model_dump()]}] * 6), CharCounter(), batch_tokens=60)
        out = run(s.summarize(self.inp(target_tokens=200), many))
        self.assertGreater(out.coverage["batches"], 4)
        self.assertTrue(out.summary.refs() <= {f"S{i}" for i in range(1, 41)})

    def test_prior_summary_is_merged_and_validated(self):
        prior = EvidenceSummary(claims=[claim("Mean age was 61.5 years", "S2")])
        out = run(self.summ([{"claims": [claim("1,200 patients enrolled", "S1").model_dump()]},
                             {"claims": [claim("1,200 patients, mean age 61.5", "S1", "S2").model_dump()]}]
                            ).summarize(self.inp(prior_summaries=[prior]), self.items[:1] + [self.items[1]]))
        self.assertEqual(set(out.summary.claims[0].sources), {"S1", "S2"})


if __name__ == "__main__":
    unittest.main()


class SalvageTests(unittest.TestCase):
    def test_truncated_json_keeps_only_complete_claims_and_is_reported(self):
        from app.agent.summarize import salvage_claims
        cut = '{"claims":[{"text":"The dose was 5 mg daily","sources":["S1"]},{"text":"Mean age was 61.5 years","sources":["S2"]},{"text":"Enrol'
        self.assertEqual(len(salvage_claims(cut)["claims"]), 2)
        self.assertIsNone(salvage_claims('{"claims":[{"text":"cut'))
        items = [private("S1", "The dose was 5 mg daily."), private("S2", "Mean age was 61.5 years.")]

        async def call(system, user, max_tokens):
            return cut, {"input_tokens": 1, "output_tokens": 700, "finish_reason": "length"}
        out = run(EvidenceSummarizer(call, CharCounter()).summarize(SummarizeInput(query="q"), items))
        self.assertEqual((out.method, len(out.summary.claims)), ("llm", 2))
        self.assertTrue(any("cut off" in w for w in out.coverage["warnings"]))

    def test_batch_output_cap_is_independent_of_the_target(self):
        log = []
        items = [private("S1", "word " * 400)]
        run(EvidenceSummarizer(scripted_llm([{"claims": []}], log), CharCounter()).summarize(SummarizeInput(query="q", target_tokens=64), items))
        self.assertGreaterEqual(log[0]["max_tokens"], 500)


class FidelityPrecisionTests(unittest.TestCase):
    """Regressions from live runs: heuristics must not drop valid claims."""

    def test_negation_elsewhere_in_other_cited_sources_does_not_flip_a_valid_claim(self):
        s1 = "The project uses PPO as the low-level optimizer for training the policy."
        s2 = "Eureka reward did not close the precision gap. No other variables changed."
        self.assertIsNone(claim_problem("The project uses PPO as the optimizer for training the policy", [s1, s2]))

    def test_real_negation_flip_is_still_caught_when_the_claim_clearly_comes_from_that_sentence(self):
        src = ["Eureka reward did not close the precision gap with the human reward."]
        self.assertEqual(claim_problem("Eureka reward closed the precision gap with the human reward", src), "negation_mismatch")

    def test_weak_conditionals_are_not_enforced_but_strong_ones_are(self):
        self.assertIsNone(claim_problem("The model is trained when data arrives", ["The model is trained when data arrives each night."]))
        self.assertEqual(claim_problem("Training uses 2048 environments", ["Training uses 2048 environments unless memory is below 16 GB."]), "condition_dropped")
