import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.agent.evidence import EvidenceStore
from app.agent.registry import CapabilityContext
from app.db import registry
from app.main import app
from app.inference.generation import ChatOrchestrator
from app.inference.memory import SqliteConversationStore
from app.inference.schemas import ChatRequest
from app.retrieval.context import TokenCounter
from app.retrieval.service import RetrievalService, resolve_filters
from tests.agent_helpers import call, make_services
from tests.test_inference import isettings
from tests.test_retrieval import FakeGen, FakeReranker, settings


class CaptureRetriever:
    def __init__(self):
        self.filters = []

    async def retrieve(self, query, document_ids, trace):
        self.filters.append(document_ids)
        return []


class DocumentOwnershipTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db_patch = patch.object(registry, "DB_PATH", Path(temp.name) / "documents.db")
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        registry.init_db()
        registry.create_document("alice-doc", "alice.pdf", "/private/alice.pdf", "alice")
        registry.create_document("bob-doc", "bob.pdf", "/private/bob.pdf", "bob")
        registry.create_document("legacy-doc", "legacy.pdf", "/private/legacy.pdf")

    def test_filters_are_intersected_with_owner_and_fail_closed(self):
        self.assertEqual(resolve_filters(None, "alice"), ["alice-doc"])
        self.assertEqual(resolve_filters(["bob-doc", "alice-doc", "legacy-doc"], "alice"), ["alice-doc"])
        self.assertEqual(resolve_filters(["alice-doc"], None), [])
        self.assertEqual(resolve_filters(None, " "), [])

    def test_retrieval_cannot_be_widened_by_ids_or_missing_filters(self):
        dense, lexical = CaptureRetriever(), CaptureRetriever()
        service = RetrievalService(settings(), dense=dense, lexical=lexical, reranker=FakeReranker(),
                                   counter=TokenCounter(None), generator=FakeGen())

        asyncio.run(service.retrieve("private", ["bob-doc"], principal="alice"))
        self.assertEqual(dense.filters, [])
        self.assertEqual(lexical.filters, [])

        asyncio.run(service.retrieve("private", principal="alice"))
        self.assertEqual(dense.filters, [["alice-doc"]])
        self.assertEqual(lexical.filters, [["alice-doc"]])

        dense.filters.clear()
        asyncio.run(service.retrieve("private", principal=None))
        self.assertEqual(dense.filters, [])

    def test_agent_search_uses_the_same_owner_scoped_retrieval(self):
        dense, lexical = CaptureRetriever(), CaptureRetriever()
        retrieval = RetrievalService(settings(), dense=dense, lexical=lexical, reranker=FakeReranker(),
                                     counter=TokenCounter(None))
        services = make_services(retrieval=retrieval)
        ctx = CapabilityContext("alice", "trace", services, EvidenceStore(), frozenset({"read"}))

        result = asyncio.run(call("search_knowledge_base", {"query": "private"}, ctx))
        self.assertEqual(result.status, "empty")
        self.assertEqual(dense.filters, [["alice-doc"]])
        self.assertEqual(lexical.filters, [["alice-doc"]])

    def test_chat_search_uses_authenticated_owner(self):
        dense, lexical = CaptureRetriever(), CaptureRetriever()
        retrieval = RetrievalService(settings(), dense=dense, lexical=lexical, reranker=FakeReranker(),
                                     counter=TokenCounter(None))
        store = SqliteConversationStore(Path(registry.DB_PATH).parent / "conversations.db")
        orchestrator = ChatOrchestrator(retrieval, isettings(), store, counter=TokenCounter(None))

        response = asyncio.run(orchestrator.run(ChatRequest(message="private"), "alice"))
        self.assertTrue(response.metadata.abstained)
        self.assertEqual(dense.filters, [["alice-doc"]])
        self.assertEqual(lexical.filters, [["alice-doc"]])

    def test_document_status_and_pages_are_owner_only(self):
        registry.save_pages("alice-doc", [{
            "page_no": 1, "page_type": "digital", "char_count": 4, "image_count": 0,
            "large_image_count": 0, "ocr_used": 0, "reason": "text",
        }])
        app.dependency_overrides.clear()
        from app.api.chat import current_user
        app.dependency_overrides[current_user] = lambda: "alice"
        self.addCleanup(app.dependency_overrides.clear)
        client = TestClient(app)

        status = client.get("/documents/alice-doc")
        self.assertEqual(status.status_code, 200)
        self.assertNotIn("owner_id", status.json())
        self.assertEqual(client.get("/documents/bob-doc").status_code, 404)
        self.assertEqual(client.get("/documents/legacy-doc").status_code, 404)
        self.assertEqual(client.get("/documents/alice-doc/pages").status_code, 200)
        self.assertEqual(client.get("/documents/bob-doc/pages").status_code, 404)

    def test_legacy_assignment_requires_exact_operator_confirmed_count(self):
        with self.assertRaises(ValueError):
            registry.assign_unowned_documents("alice", 2)
        self.assertEqual(registry.list_document_ids_for_owner("alice"), ["alice-doc"])
        self.assertEqual(registry.assign_unowned_documents("alice", 1), 1)
        self.assertEqual(registry.list_document_ids_for_owner("alice"), ["alice-doc", "legacy-doc"])

    def test_duplicate_uploads_are_reused_only_within_the_same_owner(self):
        self.assertIsNone(registry.register_upload("alice-copy", "a.pdf", "/a", "shared-hash", "alice"))
        self.assertIsNone(registry.register_upload("bob-copy", "b.pdf", "/b", "shared-hash", "bob"))
        self.assertEqual(registry.get_document_for_owner("alice-copy", "bob"), None)
        self.assertEqual(registry.get_document_for_owner("bob-copy", "bob")["document_id"], "bob-copy")
