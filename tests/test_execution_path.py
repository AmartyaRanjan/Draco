"""Both HTTP answer paths run as Draco workloads: identity, policy, budget, deadline, idempotency,
cancellation and tracing are the harness's, and /agent/chat streams over SSE."""
import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
from app.agent import harness as H
from app.agent.grounding import GovernedModel, begin_tool, resolve_markers
from app.api import agent as agent_api
from app.api import chat as chat_api
from app.inference.schemas import ChatRequest
from app.main import app
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from draco.core import DracoSettings, Status
from draco.core.errors import PolicyViolation
from draco.core.policy import Policy
from tests.agent_helpers import FakeChat, FakeRetrieval, cand, isettings, make_runner
from tests.agent_helpers import ai as agent_ai
from tests.test_agent_capabilities import HTML, fetcher
from tests.test_inference import Harness as InferenceCase
from tests.test_inference import ai, tc

KEYS = {"k-alice": "alice", "k-bob": "bob"}


def hdr(user="alice", key=None):
    h = {"X-API-Key": f"k-{user}"}
    if key:
        h["Idempotency-Key"] = key
    return h


def sse(body: str) -> list[dict]:
    out = []
    for block in body.strip().split("\n\n"):
        data = next((ln[6:] for ln in block.splitlines() if ln.startswith("data: ")), None)
        if data:
            out.append(json.loads(data))
    return out


class HarnessMixin:
    def use_harness(self, settings=None):
        h = H.build_harness(settings or DracoSettings(default_timeout_s=30))
        p = patch.object(H, "_harness", h)
        p.start()
        self.addCleanup(p.stop)
        return h

    @staticmethod
    def record(h, eid):
        return asyncio.run(h.store.get_execution(eid))

    @staticmethod
    def event_names(h, eid):
        return [e["name"] for e in asyncio.run(h.store.list_events(eid))]


class ChatPathTests(InferenceCase, HarnessMixin):
    def client(self, orch):
        app.dependency_overrides[chat_api.get_orchestrator] = lambda: orch
        self.addCleanup(app.dependency_overrides.clear)
        p = patch("app.api.chat.get_inference_settings", return_value=isettings(api_keys=KEYS))
        p.start()
        self.addCleanup(p.stop)
        return TestClient(app)

    def test_chat_is_a_draco_execution_with_one_identity(self):
        h = self.use_harness()
        c = self.client(self.orch([ai("Hello [S1]")]))
        r = c.post("/chat", json={"message": "q"}, headers=hdr())
        self.assertEqual(r.status_code, 200, r.text)
        rid = r.json()["metadata"]["request_id"]
        rec = self.record(h, rid)  # the response's request id IS the Draco execution id
        self.assertEqual((rec.target, rec.status, rec.user_id, rec.tenant_id), ("app.chat", "completed", "alice", "alice"))
        self.assertEqual(rec.policy["allowed_tools"], ["list_documents", "search_documents"])
        names = self.event_names(h, rid)
        for e in ("execution.started", "security.allowed", "retrieval.after", "model.before", "model.after",
                  "execution.finished"):
            self.assertIn(e, names)
        self.assertEqual(rec.usage["model_calls"], 1)
        self.assertEqual(rec.usage["tokens"], 110)
        self.assertNotIn(rid, h.tracer._by_exec)  # in-process trace copy released after the run

    def test_agent_mode_tool_calls_are_authorized_and_charged(self):
        h = self.use_harness()
        c = self.client(self.orch([ai("", [tc("search_documents", {"query": "x"})]), ai("Result [S1].")]))
        r = c.post("/chat", json={"message": "q", "mode": "agent"}, headers=hdr())
        self.assertEqual(r.status_code, 200, r.text)
        rid = r.json()["metadata"]["request_id"]
        self.assertEqual(r.json()["tool_trace"][0]["status"], "ok")
        rec = self.record(h, rid)
        self.assertEqual((rec.usage["model_calls"], rec.usage["tool_calls"]), (2, 1))
        names = self.event_names(h, rid)
        self.assertLess(names.index("tool.before"), names.index("tool.after"))

    def test_execution_policy_denies_a_tool_the_model_chose(self):
        h = self.use_harness()
        c = self.client(self.orch([ai("", [tc("search_documents", {"query": "x"})]), ai("No evidence.")]))
        with patch.object(chat_api, "chat_policy", lambda: Policy(allowed_tools=frozenset({"list_documents"}))):
            r = c.post("/chat", json={"message": "q", "mode": "agent"}, headers=hdr())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["tool_trace"][0]["status"], "denied")
        self.assertIn("tool.denied", self.event_names(h, r.json()["metadata"]["request_id"]))

    def test_idempotency_key_replays_without_rerunning_and_is_scoped_per_user(self):
        self.use_harness()
        c = self.client(self.orch([ai("Hello [S1]"), ai("Bob [S1]")]))
        first = c.post("/chat", json={"message": "q"}, headers=hdr(key="k1"))
        again = c.post("/chat", json={"message": "q"}, headers=hdr(key="k1"))
        self.assertEqual(first.json(), again.json())
        self.assertEqual(len(self.log), 1)  # the model ran once
        with c.stream("POST", "/chat/stream", json={"message": "q"}, headers=hdr(key="k1")) as s:
            events = sse("".join(s.iter_text()))
        self.assertEqual([e["event"] for e in events], ["start", "done"])
        self.assertEqual(events[-1]["response"], first.json())
        bob = c.post("/chat", json={"message": "q"}, headers=hdr("bob", key="k1"))  # same key, other tenant
        self.assertNotEqual(bob.json()["metadata"]["request_id"], first.json()["metadata"]["request_id"])
        self.assertTrue(bob.json()["metadata"]["abstained"])  # bob's own run: alice's documents are not his

    def test_model_call_budget_is_enforced_for_json_and_stream(self):
        self.use_harness(DracoSettings(default_timeout_s=30, max_model_calls=1))
        c = self.client(self.orch([ai("", [tc("search_documents", {"query": "x"}, str(i))]) for i in range(2)]))
        r = c.post("/chat", json={"message": "q", "mode": "agent"}, headers=hdr())
        self.assertEqual(r.status_code, 429, r.text)
        self.assertIn("model_calls", r.json()["detail"])
        with c.stream("POST", "/chat/stream", json={"message": "q", "mode": "agent"}, headers=hdr()) as s:
            events = sse("".join(s.iter_text()))
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(events[-1]["code"], "budget")

    def test_execution_deadline_is_enforced(self):
        self.use_harness(DracoSettings(default_timeout_s=0.3))

        class Slow:
            def bind_tools(self, specs):
                return self

            async def ainvoke(self, messages):
                await asyncio.sleep(5)

        orch = self.orch([])
        orch.llm_factory = lambda *a, **k: Slow()
        r = self.client(orch).post("/chat", json={"message": "q"}, headers=hdr())
        self.assertEqual(r.status_code, 504, r.text)
        self.assertIn("deadline", r.json()["detail"])

    def test_app_errors_keep_their_status_and_text(self):
        self.use_harness()
        c = self.client(self.orch([ai("x")]))
        r = c.post("/chat", json={"message": "q", "conversation_id": "conv-x"}, headers=hdr())
        self.assertEqual(r.status_code, 200)
        r = c.post("/chat", json={"message": "q", "conversation_id": "conv-x"}, headers=hdr("bob"))
        self.assertEqual((r.status_code, r.json()["detail"]), (404, "Conversation not found (conversation_forbidden)"))

    def test_cancelling_the_execution_records_it_and_persists_nothing(self):
        h = self.use_harness()

        class Slow:
            async def astream(self, messages):
                yield AIMessage(content="par")
                await asyncio.sleep(5)

        orch = self.orch([])
        orch.llm_factory = lambda *a, **k: Slow()
        seen = []

        async def go():
            async def emit(ev):
                seen.append(ev)
            t = asyncio.create_task(H.run_workload(H.CHAT, {"message": "q", "conversation_id": "cz"}, "alice",
                                                   policy=H.chat_policy(), impl_key="orchestrator", impl=orch,
                                                   emit=emit))
            await asyncio.sleep(0.2)
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
            return seen[0]["request_id"]
        rid = asyncio.run(go())
        self.assertEqual(self.record(h, rid).status, "cancelled")
        self.assertEqual(self.store.load("alice", "cz", 5).recent, [])


class AgentPathTests(unittest.TestCase, HarnessMixin):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.h = self.use_harness()
        self.client = TestClient(app)
        self.use_runner(make_runner([agent_ai("Quarterly revenue grew 12 percent [S1].")],
                                    retrieval=FakeRetrieval([cand(1, "Quarterly revenue grew 12 percent in 2023.")])))
        p = patch("app.api.chat.get_inference_settings", lambda: isettings(api_keys=KEYS))
        p.start()
        self.addCleanup(p.stop)

    def use_runner(self, pair):
        self.runner, self.svc = pair
        self.svc.settings = replace(self.svc.settings, checkpoint_path=str(Path(self.tmp.name) / "ckpt.db"))
        p = patch.object(agent_api, "_runner", self.runner)
        p.start()
        self.addCleanup(p.stop)

    def test_agent_chat_is_a_draco_execution_with_one_identity(self):
        r = self.client.post("/agent/chat", json={"message": "How much did revenue grow?"}, headers=hdr())
        self.assertEqual(r.status_code, 200, r.text)
        rid = r.json()["metadata"]["request_id"]
        rec = self.record(self.h, rid)  # Draco execution id == LangGraph thread == AgentStore execution
        self.assertEqual((rec.target, rec.status, rec.tenant_id), ("app.agent.chat", "completed", "alice"))
        self.assertEqual(self.client.get(f"/agent/executions/{rid}", headers=hdr()).json()["status"], "completed")
        names = self.event_names(self.h, rid)
        for e in ("tool.before", "tool.after", "retrieval.after", "model.before", "model.after"):
            self.assertIn(e, names)
        self.assertGreaterEqual(rec.usage["tool_calls"], 2)  # search + citation validation

    def test_capabilities_obey_the_execution_policy(self):
        with patch.object(agent_api, "agent_policy",
                          lambda reg, allow_web=False: Policy(allowed_tools=frozenset({"validate_citations"}))):
            r = self.client.post("/agent/chat", json={"message": "How much did revenue grow?"}, headers=hdr())
        body = r.json()
        self.assertEqual(body["status"], "abstained")
        search = next(t for t in body["tool_trace"] if t["tool_name"] == "search_knowledge_base")
        self.assertEqual((search["status"], search["error_code"]), ("error", "denied"))
        self.assertIn("tool.denied", self.event_names(self.h, body["metadata"]["request_id"]))

    def test_runner_failure_keeps_its_200_body_and_is_a_failed_execution(self):
        async def boom(run, state):
            raise RuntimeError("node bug")
        self.runner.n_plan = boom
        r = self.client.post("/agent/chat", json={"message": "How much did revenue grow?"}, headers=hdr())
        self.assertEqual((r.status_code, r.json()["status"]), (200, "failed"))
        rec = self.record(self.h, r.json()["metadata"]["request_id"])
        self.assertEqual((rec.status, rec.error["code"]), ("failed", "provider"))

    def test_model_errors_keep_their_status_and_text(self):
        from app.inference.schemas import InferenceError
        FakeChat.script = [InferenceError("Inference service unavailable", "llm_unavailable", 503)]
        r = self.client.post("/agent/chat", json={"message": "How much did revenue grow?"}, headers=hdr())
        self.assertEqual((r.status_code, r.json()["detail"]), (503, "Inference service unavailable (llm_unavailable)"))

    def test_stream_emits_steps_tools_tokens_and_done(self):
        self.assertEqual(self.client.post("/agent/chat/stream", json={"message": "q"}).status_code, 401)
        with self.client.stream("POST", "/agent/chat/stream", json={"message": "How much did revenue grow?"},
                                headers=hdr()) as s:
            events = sse("".join(s.iter_text()))
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "done")
        for k in ("step", "tool_call", "tool_result", "token"):
            self.assertIn(k, kinds)
        calls = [e["tool"] for e in events if e["event"] == "tool_call"]
        self.assertEqual(calls[0], "search_knowledge_base")
        streamed = "".join(e["text"] for e in events if e["event"] == "token")
        self.assertEqual(streamed, "Quarterly revenue grew 12 percent [S1].")
        done = events[-1]["response"]
        self.assertEqual((done["status"], done["citations"][0]["chunk_id"]), ("ok", "c1"))
        self.assertEqual(done["metadata"]["request_id"], events[0]["request_id"])
        self.assertEqual(self.record(self.h, events[0]["request_id"]).status, "completed")

    def test_stream_ends_with_awaiting_approval_and_the_run_resumes(self):
        class P:
            name = "fake"

            async def search(self, q, n):
                from app.agent.web import SearchHit
                return [SearchHit("https://example.org/report", "Report", "snippet")]

        self.use_runner(make_runner(
            [agent_ai("Quarterly revenue grew 12 percent [W1]. The page is not saved yet.")],
            retrieval=FakeRetrieval([]), search_provider=P(),
            fetcher=fetcher(lambda q: httpx.Response(200, headers={"content-type": "text/html"}, text=HTML)),
            ingest_pdf=lambda *a: {"document_id": "x", "job_id": "j", "status": "queued"}))
        msg = {"message": "search the web for quarterly revenue growth and save this to the knowledge base",
               "allow_web": True}
        with self.client.stream("POST", "/agent/chat/stream", json=msg, headers=hdr()) as s:
            events = sse("".join(s.iter_text()))
        last = events[-1]
        self.assertEqual(last["event"], "awaiting_approval")
        self.assertNotIn("done", [e["event"] for e in events])
        self.assertEqual(len(last["pending_approvals"]), 1)
        self.assertEqual(last["response"]["status"], "awaiting_approval")
        eid = last["execution_id"]
        self.assertEqual(self.record(self.h, eid).status, "waiting")
        self.assertEqual(self.client.get(f"/agent/executions/{eid}", headers=hdr()).json()["status"], "waiting")
        self.assertEqual(self.client.post(f"/agent/executions/{eid}/resume", json={"decision": "reject"},
                                          headers=hdr("bob")).status_code, 404)
        r = self.client.post(f"/agent/executions/{eid}/resume", json={"decision": "reject"}, headers=hdr())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["metadata"]["approval_decision"], "reject")
        resumed = [rec for rec in self.h.store._exec.values() if rec.target == "app.agent.resume"]
        self.assertEqual([(x.status, x.input["execution_id"]) for x in resumed if x.user_id == "alice"],
                         [("completed", eid)])

    def test_waiting_run_is_replayed_for_its_idempotency_key(self):
        class P:
            name = "fake"

            async def search(self, q, n):
                from app.agent.web import SearchHit
                return [SearchHit("https://example.org/report", "Report", "snippet")]

        self.use_runner(make_runner(
            [agent_ai("Quarterly revenue grew 12 percent [W1].")], retrieval=FakeRetrieval([]), search_provider=P(),
            fetcher=fetcher(lambda q: httpx.Response(200, headers={"content-type": "text/html"}, text=HTML))))
        msg = {"message": "search the web for quarterly revenue growth and save this to the knowledge base",
               "allow_web": True}
        first = self.client.post("/agent/chat", json=msg, headers=hdr(key="save-1"))
        again = self.client.post("/agent/chat", json=msg, headers=hdr(key="save-1"))
        self.assertEqual(first.json()["status"], "awaiting_approval")
        self.assertEqual(first.json(), again.json())  # no second run, no second proposal
        self.assertEqual(len(FakeChat.log), 1)  # the model ran once


class GroundingTests(unittest.TestCase):
    def test_resolve_markers_keeps_only_known_labels(self):
        import re
        ans, ok, bad = resolve_markers("A [S1] B [S9] C [S1]", re.compile(r"\[(S\d+)\]"), {"S1": "one"}.get)
        self.assertEqual((ans, ok, bad), ("A [S1] B  C [S1]", ["one"], ["S9"]))

    def test_governed_model_is_a_passthrough_outside_a_harness_and_budgeted_inside(self):
        class M:
            calls = 0

            async def ainvoke(self, messages):
                M.calls += 1
                return AIMessage(content="x", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7})

        m = GovernedModel(M())
        self.assertEqual(asyncio.run(m.ainvoke([])).content, "x")
        h = H.build_harness(DracoSettings(default_timeout_s=5, max_model_calls=1))

        async def work(ctx, inp):
            await m.ainvoke([])
            if inp == "twice":
                await m.ainvoke([])
            return "ok"
        h.register("w", work)
        r = asyncio.run(h.run("w", "once", user_id="alice"))
        self.assertEqual((r.status, r.usage.model_calls, r.usage.tokens), (Status.completed, 1, 7))
        r = asyncio.run(h.run("w", "twice", user_id="alice"))
        self.assertEqual(r.status, Status.budget_exhausted)

    def test_tools_are_default_deny_under_a_harness(self):
        h = H.build_harness(DracoSettings(default_timeout_s=5))

        async def work(ctx, inp):
            try:
                await begin_tool("search_documents")
            except PolicyViolation:
                return "denied"
            return "allowed"
        h.register("w", work)
        self.assertEqual(asyncio.run(h.run("w", None, user_id="alice")).output, "denied")
        r = asyncio.run(h.run("w", None, user_id="alice", policy=Policy(allowed_tools=frozenset({"search_documents"}))))
        self.assertEqual((r.output, r.usage.tool_calls), ("allowed", 1))

    def test_bounded_store_evicts_finished_executions_only(self):
        store = H.BoundedMemoryStore(max_executions=2)
        h = H.build_harness(DracoSettings(default_timeout_s=5), store=store)

        async def work(ctx, inp):
            return inp
        h.register("w", work)
        ids = [asyncio.run(h.run("w", i, user_id="alice")).execution_id for i in range(4)]
        self.assertEqual(list(store._exec), ids[2:])
        self.assertEqual(set(store._events), set(ids[2:]))

    def test_settings_default_to_an_app_deadline(self):
        self.assertEqual(H.harness_settings({}).default_timeout_s, H.APP_DEFAULT_TIMEOUT_S)
        self.assertEqual(H.harness_settings({"DRACO_DEFAULT_TIMEOUT_S": "12"}).default_timeout_s, 12)

    def test_placeholder_principal_runs_without_a_trusted_identity(self):
        self.assertIsNone(H.identity_for("anonymous"))
        self.assertEqual(H.identity_for("alice").tenant_id, "alice")

    def test_chat_request_round_trips_through_the_workload_input(self):
        self.assertEqual(ChatRequest.model_validate(ChatRequest(message="q").model_dump(mode="json")).message, "q")


if __name__ == "__main__":
    unittest.main()
