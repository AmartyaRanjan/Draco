import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import openai
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import ValidationError

from app.db import registry
from app.inference import tools as tools_mod
from app.inference.client import map_error
from app.inference.config import InferenceSettings
from app.inference.evidence import EvidenceRegistry
from app.inference.generation import ChatOrchestrator, ThinkFilter
from app.inference.memory import MemoryItem, SqliteConversationStore
from app.inference.schemas import ChatRequest, ChatResponse, InferenceError
from app.retrieval.context import TokenCounter
from tests.test_retrieval import FakeRetriever, ServiceTests, cand, svc


def isettings(**kw):
    base = dict(base_url="http://x/v1", api_key="k", model="m", timeout_s=5, max_retries=0,
                max_model_len=8192, max_output_tokens=1000, grounded_temperature=.3, grounded_top_p=.8,
                reasoning_temperature=.6, reasoning_top_p=.95, top_k=20, agent_max_steps=3,
                agent_max_tool_calls=2, agent_workflow_token_budget=3000, tool_timeout_s=2,
                tool_output_tokens=2500, history_tokens=800, memory_tokens=400,
                prompt_overhead_tokens=600, api_keys={"key1": "alice"}, auth_disabled=False, store_path=None)
    return InferenceSettings(**{**base, **kw})


class FakeLLM:
    """Scripted model. `script` items are AIMessage; calls are recorded."""

    def __init__(self, script, log, thinking, streaming):
        self.script, self.log, self.thinking, self.streaming = script, log, thinking, streaming
        self.tools = False

    def bind_tools(self, specs):
        self.tools = True
        return self

    def _next(self, messages):
        self.log.append({"messages": messages, "thinking": self.thinking, "tools": self.tools})
        return self.script.pop(0)

    async def ainvoke(self, messages):
        return self._next(messages)

    async def astream(self, messages):
        m = self._next(messages)
        if isinstance(m, Exception):
            raise m
        half = len(m.content) // 2
        yield AIMessageChunk(content=m.content[:half])
        yield AIMessageChunk(content=m.content[half:],
                             tool_call_chunks=[{"name": t["name"], "args": json.dumps(t["args"]),
                                                "id": t["id"], "index": i} for i, t in enumerate(m.tool_calls)],
                             response_metadata=m.response_metadata, usage_metadata=m.usage_metadata)


def ai(content="", tool_calls=None, finish="stop", out=10):
    return AIMessage(content=content, tool_calls=tool_calls or [], response_metadata={"finish_reason": finish},
                     usage_metadata={"input_tokens": 100, "output_tokens": out, "total_tokens": 100 + out})


def tc(name, args, i="t1"):
    return {"name": name, "args": args, "id": i, "type": "tool_call"}


class Harness(unittest.TestCase):
    D, L = ServiceTests.D, ServiceTests.L

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_patch = patch.object(registry, "DB_PATH", Path(tmp.name) / "documents.db")
        db_patch.start()
        self.addCleanup(db_patch.stop)
        registry.init_db()
        for document_id in ("d1", "d2", "d3"):
            registry.create_document(document_id, f"{document_id}.pdf", f"/{document_id}", "alice")
        self.store = SqliteConversationStore(Path(tmp.name) / "c.db")
        self.log = []

    def orch(self, script, retrieval=None, memory=None, **kw):
        r = retrieval or svc(FakeRetriever(self.D), FakeRetriever(self.L))
        factory = lambda s, thinking, max_tokens, streaming=False: FakeLLM(script, self.log, thinking, streaming)
        return ChatOrchestrator(r, isettings(**kw), self.store, memory, llm_factory=factory,
                                counter=TokenCounter(None))

    def run_(self, o, **req):
        return asyncio.run(o.run(ChatRequest(message="what is x?", **req), "alice"))


class RagModeTests(Harness):
    def test_valid_citations_come_from_evidence_and_invalid_are_stripped(self):
        r = self.run_(self.orch([ai("A is true [S1] and B [S9].")]))
        self.assertEqual([c.label for c in r.citations], ["S1"])
        self.assertIn(r.citations[0].chunk_id, {"c1", "c2", "c3"})
        self.assertNotIn("[S9]", r.answer)
        self.assertEqual(r.metadata.invalid_citation_markers, ["S9"])
        self.assertFalse(r.metadata.thinking)
        self.assertEqual(r.tool_trace, [])
        ChatResponse.model_validate(r.model_dump())

    def test_no_evidence_abstains_without_calling_model(self):
        o = self.orch([], retrieval=svc(FakeRetriever(), FakeRetriever()))
        r = self.run_(o)
        self.assertTrue(r.metadata.abstained)
        self.assertEqual(self.log, [])
        self.assertEqual(r.citations, [])

    def test_history_memory_evidence_are_separate_and_persisted(self):
        class Mem:
            async def search(self, u, q, n):
                return [MemoryItem("m1", "user prefers Hindi")]
        o = self.orch([ai("first [S1]"), ai("second [S1]")], memory=Mem())
        r1 = self.run_(o)
        r2 = asyncio.run(o.run(ChatRequest(message="and then?", conversation_id=r1.metadata.conversation_id), "alice"))
        msgs = self.log[1]["messages"]
        self.assertIn("user prefers Hindi", msgs[0].content)
        self.assertIn("never cite", msgs[0].content)
        self.assertEqual([m.content for m in msgs[1:3]], ["what is x?", "first [S1]"])  # verbatim history
        self.assertIn("EVIDENCE:", msgs[-1].content)
        self.assertNotIn("user prefers Hindi", msgs[-1].content)
        self.assertEqual(len(self.store.load("alice", r1.metadata.conversation_id, 20).recent), 4)
        with self.assertRaises(PermissionError):
            self.store.load("bob", r1.metadata.conversation_id, 20)
        with self.assertRaises(InferenceError) as cm:
            asyncio.run(o.run(ChatRequest(message="x", conversation_id=r1.metadata.conversation_id), "bob"))
        self.assertEqual(cm.exception.status_code, 404)

    def test_thinking_exhausting_budget_retries_once_without_thinking(self):
        o = self.orch([ai("", finish="length"), ai("ok [S1]")])
        r = self.run_(o, thinking=True)
        self.assertEqual([c["thinking"] for c in self.log], [True, False])
        self.assertFalse(r.metadata.thinking)
        self.assertEqual(r.metadata.llm_calls, 2)

    def test_generator_failure_is_controlled_and_not_persisted(self):
        o = self.orch([openai.APIConnectionError(request=None)])
        events = []

        async def emit(e):
            events.append(e["event"])
        with self.assertRaises(InferenceError) as cm:
            asyncio.run(o.run(ChatRequest(message="q", conversation_id="c1"), "alice", emit))
        self.assertEqual(cm.exception.status_code, 503)
        self.assertEqual(events[-1], "error")
        self.assertEqual(self.store.load("alice", "c1", 5).recent, [])

    def test_streaming_events_and_think_leak_filtered(self):
        events = []

        async def emit(e):
            events.append(e)
        o = self.orch([ai("<think>secret plan</think>Answer [S1]")])
        r = asyncio.run(o.run(ChatRequest(message="q"), "alice", emit))
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "done")
        streamed = "".join(e["text"] for e in events if e["event"] == "token")
        self.assertNotIn("secret", streamed)
        self.assertNotIn("secret", r.answer)
        self.assertEqual(events[-1]["response"]["citations"][0]["label"], "S1")

    def test_cancellation_persists_nothing(self):
        class Slow(FakeLLM):
            async def astream(self, messages):
                await asyncio.sleep(5)
                yield AIMessageChunk(content="x")
        o = self.orch([])
        o.llm_factory = lambda s, thinking, max_tokens, streaming=False: Slow([], [], thinking, streaming)

        async def go():
            async def emit(e):
                pass
            t = asyncio.create_task(o.run(ChatRequest(message="q", conversation_id="cx"), "alice", emit))
            await asyncio.sleep(.2)
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
        asyncio.run(go())
        self.assertEqual(self.store.load("alice", "cx", 5).recent, [])


class AgentModeTests(Harness):
    def test_tool_execution_recorded_by_orchestrator(self):
        o = self.orch([ai("", [tc("search_documents", {"query": "x"})]), ai("Result [S1].")])
        r = self.run_(o, mode="agent")
        self.assertEqual(len(r.tool_trace), 1)
        t = r.tool_trace[0]
        self.assertEqual((t.tool_name, t.status, t.input), ("search_documents", "ok", {"query": "x", "document_ids": None}))
        self.assertGreater(t.duration_ms, 0)
        self.assertIn("evidence blocks", t.output_summary)
        self.assertTrue(r.metadata.thinking)
        self.assertEqual(r.citations[0].label, "S1")
        self.assertTrue(self.log[0]["tools"])
        run = json.loads(next(iter(self._runs()))["tool_outputs_json"])
        self.assertIn(t.output_ref, run)  # full output stored internally, only summary exposed
        self.assertNotIn("output", t.model_dump())

    def _runs(self):
        with self.store._conn() as c:
            return [dict(x) for x in c.execute("SELECT * FROM inference_runs")]

    def test_invalid_arguments_unknown_tool_and_limit(self):
        script = [ai("", [tc("search_documents", {"query": ""}, "a"), tc("nope", {}, "b")]),
                  ai("", [tc("list_documents", {}, "c"), tc("list_documents", {}, "d")]),
                  ai("done")]
        with patch.object(tools_mod.registry, "get_connection", side_effect=RuntimeError("db down")):
            r = self.run_(self.orch(script, agent_max_tool_calls=3), mode="agent")
        self.assertEqual([t.status for t in r.tool_trace], ["invalid_arguments", "invalid_arguments", "error", "limit_exceeded"])
        self.assertNotIn("db down", json.dumps(r.model_dump()))

    def test_tool_timeout_and_step_limit_force_final_answer(self):
        async def slow(args, ctx):
            await asyncio.sleep(3)
        slow_tool = tools_mod.Tool("search_documents", "d", "p", tools_mod.SearchArgs, slow)
        script = [ai("", [tc("search_documents", {"query": "x"}, str(i))]) for i in range(2)] + [ai("final")]
        with patch.dict(tools_mod.TOOLS, {"search_documents": slow_tool}):
            r = self.run_(self.orch(script, agent_max_steps=2, tool_timeout_s=.2), mode="agent")
        self.assertEqual([t.status for t in r.tool_trace], ["timeout", "timeout"])
        self.assertFalse(self.log[-1]["tools"])  # final call is forced without tools
        self.assertEqual(r.answer, "final")
        self.assertEqual(r.metadata.llm_calls, 3)

    def test_workflow_token_budget_forces_final(self):
        script = [ai("", [tc("list_documents", {})], out=500), ai("final")]
        with patch.object(tools_mod.registry, "get_connection", side_effect=RuntimeError):
            r = self.run_(self.orch(script, agent_workflow_token_budget=400), mode="agent")
        self.assertEqual(r.answer, "final")
        self.assertFalse(self.log[-1]["tools"])

    def test_document_scope_cannot_be_widened_by_model(self):
        o = self.orch([ai("", [tc("search_documents", {"query": "x", "document_ids": ["other"]})]), ai("done")])
        r = self.run_(o, mode="agent", document_ids=["d1"])
        self.assertEqual(r.tool_trace[0].status, "denied")


class UnitTests(unittest.TestCase):
    def test_request_schema_bounds(self):
        for bad in ({"message": ""}, {"message": "x", "max_output_tokens": 5000}, {"message": "x", "mode": "z"}):
            with self.assertRaises(ValidationError):
                ChatRequest(**bad)

    def test_think_filter_across_chunk_boundaries(self):
        f = ThinkFilter()
        out = "".join(f.feed(p) for p in ["Hi <thi", "nk>hidden", " stuff</th", "ink> there"])
        self.assertEqual(out, "Hi  there")

    def test_evidence_registry_stable_labels_and_budget(self):
        reg = EvidenceRegistry(TokenCounter(None))
        c1, c2 = cand(1, "a" * 40), cand(2, "b" * 4000)
        text, labels = reg.add_many([c1, c2], {}, 100)
        self.assertEqual(labels, ["S1"])  # oversized chunk skipped, not truncated
        _, again = reg.add_many([c1], {}, 100)
        self.assertEqual(again, ["S1"])
        self.assertIsNone(reg.citation("S2"))
        self.assertEqual(reg.citation("S1").chunk_id, "c1")

    def test_error_mapping_hides_internals(self):
        self.assertEqual(map_error(openai.APITimeoutError(request=None)).status_code, 504)
        e = map_error(RuntimeError("secret path /etc"))
        self.assertNotIn("secret", str(e))

    def test_settings_validation(self):
        with self.assertRaises(ValueError):
            isettings(max_model_len=1500).validate()
        with self.assertRaises(ValueError):
            isettings(api_keys={}, auth_disabled=False).validate()
        isettings().validate()
        self.assertEqual(isettings().evidence_budget(), 8192 - 1000 - 800 - 400 - 600)


class ApiTests(unittest.TestCase):
    def client(self, orch):
        from app.api import chat
        from app.main import app
        app.dependency_overrides[chat.get_orchestrator] = lambda: orch
        self.addCleanup(app.dependency_overrides.clear)
        return TestClient(app)

    def test_auth_required_and_streaming_sse(self):
        h = Harness()
        h.setUp()
        orch = h.orch([ai("Hello [S1]"), ai("Hello [S1]")])
        with patch("app.api.chat.get_inference_settings", return_value=isettings()):
            c = self.client(orch)
            self.assertEqual(c.post("/chat", json={"message": "q"}).status_code, 401)
            self.assertEqual(c.post("/chat", json={"message": "q"}, headers={"x-api-key": "bad"}).status_code, 401)
            r = c.post("/chat", json={"message": "q"}, headers={"x-api-key": "key1"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["citations"][0]["label"], "S1")
            with c.stream("POST", "/chat/stream", json={"message": "q"}, headers={"x-api-key": "key1"}) as s:
                body = "".join(s.iter_text())
        self.assertIn("event: start", body)
        self.assertIn("event: token", body)
        self.assertTrue(body.rstrip().endswith("}") and "event: done" in body)


if __name__ == "__main__":
    unittest.main()


class ReducedOutputCapTests(unittest.TestCase):
    MSG = ("Error code: 400 - This model's maximum context length is 8192 tokens. However, you requested 2400 output tokens "
           "and your prompt contains at least 5793 input tokens, for a total of at least 8193 tokens.")

    def test_shrinks_geometrically_on_overflow(self):
        from app.inference.client import reduced_output_cap
        self.assertEqual(reduced_output_cap(Exception(self.MSG), 2400), 1680)
        self.assertEqual(reduced_output_cap(Exception(self.MSG), 1680), 1176)

    def test_none_below_floor_or_for_other_errors(self):
        from app.inference.client import reduced_output_cap
        self.assertIsNone(reduced_output_cap(Exception(self.MSG), 300))
        self.assertIsNone(reduced_output_cap(Exception("Error code: 400 - invalid tool schema"), 2400))
