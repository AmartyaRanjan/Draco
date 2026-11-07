"""Rolling conversation summaries: generated past a token threshold, stored, then read by the store."""
import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.inference.memory import RollingSummarizer, SqliteConversationStore
from app.retrieval.context import TokenCounter

from tests.test_inference import FakeLLM, Harness, ai, isettings


def _resp(rid):
    return json.dumps({"metadata": {"request_id": rid}})


class WordCounter:
    def count(self, text):
        return len(text.split())


class StoreTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "c.db"
        self.store = SqliteConversationStore(self.path)
        for i in range(3):
            self.store.save_turn("alice", "c1", f"q{i}", _resp(f"r{i}"), f"a{i}", {})

    def test_summary_state_and_cas_save(self):
        st = self.store.summary_state("alice", "c1")
        self.assertEqual((st.summary, st.summary_upto, len(st.messages)), (None, 0, 6))
        upto = st.messages[3]["id"]
        self.assertTrue(self.store.save_summary("alice", "c1", "S1", upto, 0))
        self.assertFalse(self.store.save_summary("alice", "c1", "stale", upto + 1, 0))  # expected_upto mismatch
        self.assertFalse(self.store.save_summary("alice", "c1", "back", upto - 1, upto))  # never moves backwards
        self.assertFalse(self.store.save_summary("bob", "c1", "x", upto + 1, upto))  # owner-bound
        ctx = self.store.load("alice", "c1", 20)
        self.assertEqual(ctx.summary, "S1")
        self.assertEqual([m["content"] for m in ctx.recent], ["q2", "a2"])  # covered messages not repeated

    def test_summary_state_is_user_scoped(self):
        with self.assertRaises(PermissionError):
            self.store.summary_state("bob", "c1")
        self.assertIsNone(self.store.summary_state("alice", "missing"))

    def test_old_database_is_migrated(self):
        old = Path(self.path.parent) / "old.db"
        with sqlite3.connect(old) as c:
            c.executescript("""
            CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, summary TEXT,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
              role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            INSERT INTO conversations (conversation_id, user_id, summary) VALUES ('c', 'alice', 'legacy');
            INSERT INTO messages (conversation_id, role, content) VALUES ('c', 'user', 'hi');""")
        c.close()
        store = SqliteConversationStore(old)
        ctx = store.load("alice", "c", 20)
        self.assertEqual((ctx.summary, [m["content"] for m in ctx.recent]), ("legacy", ["hi"]))
        self.assertEqual(store.summary_state("alice", "c").summary_upto, 0)


class SummarizerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = SqliteConversationStore(Path(tmp.name) / "c.db")
        self.log = []

    def summarizer(self, script, **kw):
        s = isettings(summary_enabled=True, summary_trigger_tokens=30, history_tokens=12,
                      summary_input_tokens=256, **kw)
        factory = lambda st, thinking, max_tokens, streaming=False: FakeLLM(script, self.log, thinking, streaming)
        return RollingSummarizer(self.store, s, factory, WordCounter())

    def _turns(self, n, words=5, start=0):
        for i in range(start, start + n):
            self.store.save_turn("alice", "c1", " ".join([f"q{i}"] * words), _resp(f"r{i}"),
                                 " ".join([f"a{i}"] * words), {})

    def test_below_threshold_does_nothing(self):
        self._turns(2)  # 20 words <= 30
        self.assertFalse(asyncio.run(self.summarizer([]).maybe_summarize("alice", "c1")))
        self.assertEqual(self.log, [])

    def test_older_turns_are_summarised_newest_stay_verbatim(self):
        self._turns(4)  # 40 words > 30; history_tokens=12 keeps the newest 2 messages (10 words)
        ok = asyncio.run(self.summarizer([ai("<think>x</think>User asked q0..q2.")]).maybe_summarize("alice", "c1"))
        self.assertTrue(ok)
        call = self.log[0]
        self.assertFalse(call["thinking"])
        prompt = call["messages"][1].content
        self.assertIn("User: q0", prompt)
        self.assertIn("Assistant: a2", prompt)
        self.assertNotIn("q3", prompt)
        ctx = self.store.load("alice", "c1", 20)
        self.assertEqual(ctx.summary, "User asked q0..q2.")
        self.assertEqual([m["content"].split()[0] for m in ctx.recent], ["q3", "a3"])

    def test_previous_summary_is_folded_in_and_batches_are_bounded(self):
        self._turns(4)
        st = self.store.summary_state("alice", "c1")
        self.store.save_summary("alice", "c1", "OLD", st.messages[1]["id"], 0)
        self._turns(2, start=4)
        # uncovered: 6 earlier + 4 new = 10 msgs, 50 words; the newest 2 stay verbatim, and an input
        # budget of 12 words per call forces several folds over the other 8.
        s = self.summarizer([ai(f"S-{i}") for i in range(8)])
        s.s = isettings(summary_enabled=True, summary_trigger_tokens=30, history_tokens=12, summary_input_tokens=12)
        self.assertTrue(asyncio.run(s.maybe_summarize("alice", "c1")))
        self.assertIn("PREVIOUS SUMMARY:\nOLD", self.log[0]["messages"][1].content)
        self.assertGreater(len(self.log), 1)
        self.assertIn("PREVIOUS SUMMARY:\nS-0", self.log[1]["messages"][1].content)
        self.assertTrue(self.store.load("alice", "c1", 20).summary.startswith("S-"))

    def test_model_failure_leaves_state_untouched(self):
        self._turns(4)

        class Boom(FakeLLM):
            async def ainvoke(self, messages):
                raise RuntimeError("down")
        s = self.summarizer([])
        s.llm_factory = lambda st, thinking, max_tokens, streaming=False: Boom([], [], thinking, streaming)
        task_result = asyncio.run(self._schedule_and_drain(s))
        self.assertIsNone(task_result)
        self.assertIsNone(self.store.load("alice", "c1", 20).summary)
        # empty model output is not stored either
        self.assertFalse(asyncio.run(self.summarizer([ai("")]).maybe_summarize("alice", "c1")))

    async def _schedule_and_drain(self, s):
        t = s.schedule("alice", "c1")
        self.assertIsNone(s.schedule("alice", "c1"))  # one at a time per conversation
        await s.drain()
        return t.result()

    def test_store_without_summary_support_is_ignored(self):
        class Plain:
            def load(self, *a): ...
        s = self.summarizer([])
        s.store = Plain()
        self.assertFalse(asyncio.run(s.maybe_summarize("alice", "c1")))


class OrchestratorHookTests(Harness):
    def test_off_by_default(self):
        self.assertIsNone(self.orch([]).summarizer)

    def test_enabled_summary_runs_after_a_persisted_turn(self):
        script = [ai(" ".join(["long"] * 40) + " [S1]"), ai("Summary of turn one.")]
        o = self.orch(script, summary_enabled=True, summary_trigger_tokens=10, history_tokens=20)
        o.counter = TokenCounter(None)

        async def go():
            from app.inference.schemas import ChatRequest
            r = await o.run(ChatRequest(message="what is x?", conversation_id="conv1"), "alice")
            await o.summarizer.drain()
            return r
        asyncio.run(go())
        self.assertEqual(len(self.log), 2)
        self.assertEqual(self.store.load("alice", "conv1", 20).summary, "Summary of turn one.")


if __name__ == "__main__":
    unittest.main()
