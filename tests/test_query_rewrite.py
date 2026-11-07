"""Query rewriting before /chat retrieval (evals.md 4.1: the long document-descriptor prefix)."""
import asyncio
import unittest

from app.retrieval import rewrite
from langchain_core.messages import HumanMessage

from tests.test_inference import FakeLLM, Harness, ai, isettings
from tests.test_retrieval import FakeRetriever, svc

# Verbatim from tests/eval/rag_gold.jsonl, the case evals.md reports as the dominant retrieval failure.
DESCRIPTOR_Q = ("In the 2DOF physics-informed adaptive domain randomization research roadmap, "
                "how often is the DR distribution initially recommended to be updated?")
STRIPPED_Q = "How often is the DR distribution initially recommended to be updated?"


class RuleTests(unittest.TestCase):
    def test_evals_descriptor_prefix_is_stripped(self):
        r = rewrite.rewrite_rules(DESCRIPTOR_Q)
        self.assertEqual(r.query, STRIPPED_Q)
        self.assertEqual(r.removed, "2DOF physics-informed adaptive domain randomization research roadmap")
        self.assertEqual(r.method, "rules")
        self.assertTrue(r.changed)

    def test_variants_of_long_descriptors(self):
        cases = {
            "For the 2DOF physics-informed adaptive domain randomization project (v2), list the milestones.":
                "List the milestones.",
            "According to the quarterly sales and operations planning report, how did Q3 revenue change?":
                "How did Q3 revenue change?",
            "In the reinforcement learning course lecture slides: what is TD(0)?": "What is TD(0)?",
            "  In the   very long internal engineering design document,   why was Redis chosen? ":
                "Why was Redis chosen?",
        }
        for q, want in cases.items():
            self.assertEqual(rewrite.rewrite_rules(q).query, want, q)

    def test_short_or_absent_descriptors_are_kept(self):
        for q in ("In the transformer paper, what does the decoder attend to?",
                  STRIPPED_Q,
                  "In 2024, what did revenue look like in the annual report?",
                  "In the 2DOF physics-informed adaptive domain randomization research roadmap, why?"):
            r = rewrite.rewrite_rules(q)
            self.assertEqual(r.query, " ".join(q.split()), q)
            self.assertEqual(r.method, "none")

    def test_matches_the_eval_scripts_strip_scope_on_the_gold_set(self):
        # scripts/eval_rag.py --strip-scope is what measured hit@5 0.79 -> 0.97. The rule must agree with it
        # on every gold question except short (< 4-word) descriptors, which are deliberately kept.
        import json
        import re
        from pathlib import Path
        scope = re.compile(r"^(?:In|For) the .{3,160}?(?:report|roadmap|lab|text|slides|project)"
                           r"(?: \([^)]*\))?'?s?,\s+", re.I)
        gold = Path(__file__).parent / "eval" / "rag_gold.jsonl"
        disagreements = []
        for line in gold.read_text().splitlines():
            q = json.loads(line)["question"]
            ref = scope.sub("", q, count=1)
            ref = ref[:1].upper() + ref[1:] if ref != q else q
            ours = rewrite.rewrite_rules(q).query
            if ours != " ".join(ref.split()):
                m = scope.match(q)
                disagreements.append(len(m.group(0).split()) - 3 if m else None)
        # Only the 2-word "trajectory-sampling lab" descriptor differs, by design.
        self.assertLessEqual(len(disagreements), 1)
        self.assertTrue(all(d is not None for d in disagreements))

    def test_off_mode_only_normalises(self):
        r = asyncio.run(rewrite.rewrite_query("  " + DESCRIPTOR_Q, mode="off"))
        self.assertEqual(r.query, DESCRIPTOR_Q)
        self.assertEqual(r.method, "none")


class LlmRewriteTests(unittest.TestCase):
    def factory(self, script, log):
        return lambda s, thinking, max_tokens, streaming=False: FakeLLM(script, log, thinking, streaming)

    def test_llm_resolves_follow_up_and_rules_still_apply(self):
        log = []
        hist = [{"role": "user", "content": "Tell me about the Atlas robot."},
                {"role": "assistant", "content": "Atlas is a humanoid robot [S1]."}]
        r = asyncio.run(rewrite.rewrite_query("what about its battery?", hist, mode="llm",
                                              llm_factory=self.factory([ai("Query: \"Atlas robot battery capacity\"")], log),
                                              settings=isettings()))
        self.assertEqual(r.query, "Atlas robot battery capacity")
        self.assertEqual(r.method, "llm")
        self.assertFalse(log[0]["thinking"])
        prompt = log[0]["messages"][1].content
        self.assertIn("Atlas is a humanoid robot", prompt)
        self.assertIn("what about its battery?", prompt)

    def test_llm_failure_or_empty_output_falls_back_to_rules(self):
        for script in ([RuntimeError("down")], [ai("")], [ai("x" * 5000)]):
            class Boom(FakeLLM):
                async def ainvoke(self, messages):
                    m = self._next(messages)
                    if isinstance(m, Exception):
                        raise m
                    return m
            factory = lambda s, thinking, max_tokens, streaming=False, sc=script: Boom(sc, [], thinking, streaming)
            r = asyncio.run(rewrite.rewrite_query(DESCRIPTOR_Q, None, mode="llm", llm_factory=factory,
                                                  settings=isettings()))
            self.assertEqual(r.query, STRIPPED_Q)
            self.assertEqual(r.method, "llm_fallback")

    def test_llm_timeout_falls_back(self):
        class Slow(FakeLLM):
            async def ainvoke(self, messages):
                await asyncio.sleep(1)
        factory = lambda s, thinking, max_tokens, streaming=False: Slow([], [], thinking, streaming)
        r = asyncio.run(rewrite.rewrite_query("q one two", None, mode="llm", llm_factory=factory,
                                              settings=isettings(), timeout_s=0.01))
        self.assertEqual((r.query, r.method), ("q one two", "llm_fallback"))

    def test_trace_has_no_query_text(self):
        d = rewrite.rewrite_rules(DESCRIPTOR_Q).as_trace()
        self.assertEqual(d, {"method": "rules", "changed": True, "descriptor_removed": True})


class RecordingRetriever(FakeRetriever):
    def __init__(self, cands, seen):
        super().__init__(cands)
        self.seen = seen

    async def retrieve(self, q, docs, trace):
        self.seen.append(q)
        return await super().retrieve(q, docs, trace)


class ChatHookTests(Harness):
    def test_chat_retrieves_with_rewritten_query_but_answers_the_original(self):
        seen = []
        r = svc(RecordingRetriever(self.D, seen), RecordingRetriever(self.L, seen))
        o = self.orch([ai("Threshold is 80% [S1].")], retrieval=r)
        from app.inference.schemas import ChatRequest
        resp = asyncio.run(o.run(ChatRequest(message=DESCRIPTOR_Q), "alice"))
        self.assertEqual(seen, [STRIPPED_Q] * 2)
        last = self.log[0]["messages"][-1]
        self.assertIsInstance(last, HumanMessage)
        self.assertIn(f"QUESTION: {DESCRIPTOR_Q}", last.content)
        self.assertIn("query_rewrite", resp.metadata.timings_ms)

    def test_rewrite_off_passes_the_message_through(self):
        seen = []
        r = svc(RecordingRetriever(self.D, seen), RecordingRetriever(self.L, seen))
        o = self.orch([ai("ok [S1]")], retrieval=r, query_rewrite="off")
        from app.inference.schemas import ChatRequest
        asyncio.run(o.run(ChatRequest(message=DESCRIPTOR_Q), "alice"))
        self.assertEqual(seen, [DESCRIPTOR_Q] * 2)

    def test_settings_validate_rewrite_mode(self):
        with self.assertRaises(ValueError):
            isettings(query_rewrite="magic").validate()


if __name__ == "__main__":
    unittest.main()
