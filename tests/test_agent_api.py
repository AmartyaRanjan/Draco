import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.agent.catalog import build_registry
from app.agent.graph import AgentRunner
from app.agent.runtime import RuntimeAdapter, detect_capabilities
from app.api import agent as agent_api
from app.api.chat import current_user
from app.inference.config import InferenceSettings
from app.main import app
from tests.agent_helpers import FakeChat, FakeRetrieval, ai, cand, isettings, make_runner
from tests.test_agent_capabilities import HTML, fetcher

KEYS = {"k-alice": "alice", "k-bob": "bob"}


def H(user="alice"):
    return {"X-API-Key": f"k-{user}"}


class AgentApiTests(unittest.TestCase):
    def setUp(self):
        self.ingested = []
        self.runner, self.svc = make_runner(
            [ai("Quarterly revenue grew 12 percent [S1].")], retrieval=FakeRetrieval([cand(1, "Quarterly revenue grew 12 percent in 2023.")]),
            fetcher=fetcher(lambda q: httpx.Response(200, headers={"content-type": "text/html"}, text=HTML)),
            ingest_pdf=lambda *a: self.ingested.append(a) or {"document_id": "doc-9", "job_id": "rq-9", "status": "queued", "duplicate": False})
        self.svc.extra["docs"]["doc-9"] = {"document_id": "doc-9", "owner_id": "alice", "filename": "x.pdf", "status": "queued", "file_path": "/x"}
        self.patches = [patch.object(agent_api, "_runner", self.runner),
                        patch("app.api.chat.get_inference_settings", lambda: isettings(api_keys=KEYS))]
        for p in self.patches:
            p.start()
        self.client = TestClient(app)

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def seed_proposal(self, user="alice"):
        from app.agent.registry import CapabilityContext
        from app.agent.evidence import EvidenceStore
        ctx = CapabilityContext(user, "t", self.svc, EvidenceStore(), frozenset({"read", "write", "web"}))
        f = asyncio.run(self.runner.reg.execute("fetch_webpage", {"url": "https://example.org/report"}, ctx))
        r = asyncio.run(self.runner.reg.execute("propose_knowledge_ingestion", {"page_id": f.data["page_id"], "reason": "needed for answers"}, ctx))
        return r.data["source_id"]

    def test_auth_required_everywhere(self):
        for method, path in (("post", "/agent/chat"), ("get", "/agent/capabilities"), ("post", "/agent/sources/x/approve"),
                             ("get", "/agent/artifacts/x"), ("post", "/agent/feedback")):
            r = self.client.request(method.upper(), path, json={"message": "x"})
            self.assertEqual(r.status_code, 401, path)

    def test_chat_returns_grounded_response(self):
        r = self.client.post("/agent/chat", json={"message": "How much did revenue grow?"}, headers=H())
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual((body["status"], body["citations"][0]["chunk_id"]), ("ok", "c1"))
        r = self.client.post("/agent/chat", json={"message": ""}, headers=H())
        self.assertEqual(r.status_code, 422)

    def test_capabilities_listing_reports_unconfigured_providers(self):
        r = self.client.get("/agent/capabilities", headers=H()).json()
        self.assertEqual(len(r["capabilities"]), 21)
        self.assertIn("web_search", r["unconfigured"])
        self.assertIn("image_generation", r["unconfigured"])
        self.assertTrue(r["runtime"]["supports_thinking_toggle"])
        by = {c["name"]: c for c in r["capabilities"]}
        self.assertTrue(by["ingest_approved_source"]["requires_approval"])
        self.assertFalse(by["ingest_approved_source"]["llm_exposed"])
        self.assertTrue(by["web_search"]["untrusted_output"])

    def test_approval_flow_over_http_is_user_scoped(self):
        sid = self.seed_proposal()
        self.assertEqual(self.client.post(f"/agent/sources/{sid}/ingest", headers=H()).status_code, 409)   # not approved yet
        self.assertEqual(self.client.post(f"/agent/sources/{sid}/approve", headers=H("bob")).status_code, 404)  # not bob's source
        r = self.client.post(f"/agent/sources/{sid}/approve", json={"ingest": True}, headers=H())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["ingestion"]["data"]["document_id"], "doc-9")
        self.client.post(f"/agent/sources/{sid}/ingest", headers=H())
        self.assertEqual(len(self.ingested), 1)                                                            # idempotent
        prov = self.client.get(f"/agent/sources/{sid}", headers=H()).json()
        self.assertEqual(prov["data"]["approval"]["decision"], "approved")
        self.assertEqual(self.client.get(f"/agent/sources/{sid}", headers=H("bob")).status_code, 200)       # empty result, not bob's data
        self.assertEqual(self.client.get(f"/agent/sources/{sid}", headers=H("bob")).json()["status"], "empty")

    def test_reject_blocks_ingestion(self):
        sid = self.seed_proposal()
        self.assertEqual(self.client.post(f"/agent/sources/{sid}/reject", headers=H()).status_code, 200)
        self.assertEqual(self.client.post(f"/agent/sources/{sid}/ingest", headers=H()).status_code, 409)
        self.assertEqual(self.ingested, [])

    def test_artifact_download_is_owner_only(self):
        from app.agent.registry import CapabilityContext
        from app.agent.evidence import EvidenceStore
        ctx = CapabilityContext("alice", "t", self.svc, EvidenceStore(), frozenset({"read", "generate"}))
        r = asyncio.run(self.runner.reg.execute("generate_document", {"title": "T", "sections": [{"heading": "H", "paragraphs": ["hi"]}]}, ctx))
        aid = r.data["artifact"]["artifact_id"]
        ok = self.client.get(f"/agent/artifacts/{aid}", headers=H())
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.headers["x-content-type-options"], "nosniff")
        self.assertTrue(ok.content.startswith(b"PK"))
        self.assertEqual(self.client.get(f"/agent/artifacts/{aid}", headers=H("bob")).status_code, 404)
        self.assertEqual(self.client.get("/agent/artifacts/ghost", headers=H()).status_code, 404)

    def test_feedback_endpoint(self):
        r = self.client.post("/agent/feedback", json={"target_type": "answer", "target_id": "req-1", "rating": 1}, headers=H())
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["data"]["authorizes_ingestion"])
        self.assertEqual(self.client.post("/agent/feedback", json={"target_type": "answer", "target_id": "r", "rating": 9}, headers=H()).status_code, 422)


if __name__ == "__main__":
    unittest.main()
