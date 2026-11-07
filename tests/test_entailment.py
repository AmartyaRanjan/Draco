"""LLM entailment judge behind Services.entailment: off by default, bounded, and never worse than lexical."""
import asyncio
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from app.agent import config as agent_config
from app.agent.entailment import JudgeError, LlmEntailmentJudge, build_entailment_judge, parse_verdict
from app.agent.runtime import RuntimeAdapter, detect_capabilities

from tests.agent_helpers import FakeChat, ai, call, isettings, make_services, private, store_with
from tests.test_agent_capabilities import ctx_for


def run(c):
    return asyncio.run(c)


def scripted(*labels, log=None):
    """A judge returning the given labels in order (an Exception item is raised)."""
    items = list(labels)

    async def judge(claim, source):
        if log is not None:
            log.append(claim)
        r = items.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    return judge


class ParseTests(unittest.TestCase):
    def test_json_and_unambiguous_free_text(self):
        self.assertEqual(parse_verdict('{"verdict": "Entailed"}'), "entailed")
        self.assertEqual(parse_verdict("The claim is contradicted by the source."), "contradicted")
        for bad in ("", "{}", '{"verdict": "maybe"}', "entailed or neutral", "[1, 2]"):
            with self.assertRaises(JudgeError, msg=bad):
                parse_verdict(bad)


class JudgeTests(unittest.TestCase):
    def test_prompt_marks_the_source_untrusted_and_bounds_it(self):
        seen = {}

        async def llm(system, user, max_tokens):
            seen.update(system=system, user=user, max_tokens=max_tokens)
            return '{"verdict": "neutral"}', {}
        judge = LlmEntailmentJudge(llm, max_source_chars=600)
        self.assertEqual(run(judge("Revenue rose.", "x" * 5000)), "neutral")
        self.assertIn("untrusted", seen["system"])
        self.assertLess(len(seen["user"]), 700)
        self.assertLessEqual(seen["max_tokens"], 64)

    def test_timeout_raises_judge_error(self):
        async def slow(system, user, max_tokens):
            await asyncio.sleep(2)
        with self.assertRaises(JudgeError):
            run(LlmEntailmentJudge(slow, timeout_s=0.05)("c", "s"))

    def test_built_from_settings_over_the_inference_client(self):
        svc = make_services()
        adapter = RuntimeAdapter(isettings(), detect_capabilities("Qwen/Qwen3-4B-FP8"), client_cls=FakeChat)
        self.assertIsNone(build_entailment_judge(svc.settings, adapter))  # off by default
        judge = build_entailment_judge(replace(svc.settings, entailment_judge=True, entailment_timeout_s=3), adapter)
        FakeChat.script, FakeChat.log = [ai('{"verdict": "contradicted"}')], []
        self.assertEqual(run(judge("Revenue was 9 million.", "Revenue was 5 million.")), "contradicted")
        kw = FakeChat.log[0]["kw"]
        self.assertEqual(kw["extra_body"]["response_format"], {"type": "json_object"})
        self.assertEqual(kw["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(judge.timeout_s, 3)

    def test_settings_from_environment(self):
        agent_config.get_agent_settings.cache_clear()
        self.addCleanup(agent_config.get_agent_settings.cache_clear)
        with patch.dict(os.environ, {"AGENT_ENTAILMENT_JUDGE": "true", "AGENT_ENTAILMENT_TIMEOUT_S": "4",
                                     "AGENT_ENTAILMENT_MAX_CLAIMS": "3"}):
            s = agent_config.get_agent_settings()
        self.assertEqual((s.entailment_judge, s.entailment_timeout_s, s.entailment_max_claims), (True, 4.0, 3))
        agent_config.get_agent_settings.cache_clear()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_ENTAILMENT_JUDGE", None)
            self.assertFalse(agent_config.get_agent_settings().entailment_judge)
        with self.assertRaises(ValueError):
            replace(s, entailment_timeout_s=0).validate()


class ValidationWithJudgeTests(unittest.TestCase):
    def setUp(self):
        self.ev = store_with(private("S1", "Revenue was 5 million dollars in 2023 for the company."))

    def v(self, answer, judge, **settings):
        svc = make_services(entailment=judge)
        if settings:
            svc.settings = replace(svc.settings, **settings)
        return run(call("validate_citations", {"answer": answer}, ctx_for(svc, evidence=self.ev))).data

    def test_contradicted_makes_a_lexically_supported_claim_unsupported(self):
        d = self.v("Revenue was 5 million dollars in 2023 [S1].", scripted("contradicted"))
        self.assertEqual((d["status"], d["claims"][0]["verdict"]), ("fail", "unsupported"))
        self.assertEqual(d["judge"], {"judged": 1, "changed": 1, "errors": 0, "skipped": 0})
        self.assertIn("entailment_judge", d["method"])

    def test_neutral_demotes_and_entailed_promotes_a_paraphrase(self):
        d = self.v("Revenue was 5 million dollars in 2023 [S1].", scripted("neutral"))
        self.assertEqual(d["claims"][0]["verdict"], "weak")
        para = "The firm earned income of 5 million dollars during 2023 [S1]."
        lexical = self.v(para, None)["claims"][0]["verdict"]
        self.assertEqual(lexical, "weak")
        d = self.v(para, scripted("entailed"))
        self.assertEqual((d["claims"][0]["verdict"], d["claims"][0]["reason"]), ("supported", "validator model: entailed"))

    def test_entailed_never_overrides_number_checks(self):
        d = self.v("Revenue was 9 million dollars in 2023 [S1].", scripted("entailed"))
        self.assertEqual(d["claims"][0]["verdict"], "unsupported")  # judged only when not already unsupported
        self.assertEqual(d["judge"]["judged"], 0)

    def test_judge_failure_keeps_the_lexical_verdict(self):
        d = self.v("Revenue was 5 million dollars in 2023 [S1].", scripted(JudgeError("timed out")))
        self.assertEqual((d["status"], d["claims"][0]["verdict"]), ("pass", "supported"))
        self.assertEqual(d["judge"]["errors"], 1)

    def test_judged_claims_are_capped(self):
        log = []
        ans = " ".join(["Revenue was 5 million dollars in 2023 [S1]."] * 3)
        d = self.v(ans, scripted("entailed", "entailed", log=log), entailment_max_claims=2)
        self.assertEqual(len(log), 2)
        self.assertEqual(d["judge"]["skipped"], 1)

    def test_without_a_judge_the_output_is_unchanged(self):
        d = self.v("Revenue was 5 million dollars in 2023 [S1].", None)
        self.assertIsNone(d["judge"])
        self.assertEqual(d["method"], "identifier_resolution + lexical_overlap")


if __name__ == "__main__":
    unittest.main()
