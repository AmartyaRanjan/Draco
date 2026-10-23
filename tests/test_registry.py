import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.db import registry


class RegistryTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = patch.object(registry, "DB_PATH", Path(tmp.name) / "documents.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        registry.init_db()

    def upload(self, document_id, content_hash="h1"):
        return registry.register_upload(document_id, f"{document_id}.pdf", f"/tmp/{document_id}.pdf", content_hash)


class DuplicateTests(RegistryTestCase):
    def test_new_content_is_registered(self):
        self.assertIsNone(self.upload("a"))
        self.assertEqual(registry.get_document("a")["content_hash"], "h1")

    def test_identical_content_returns_existing_document(self):
        self.upload("a")
        registry.update_status("a", "queued", job_id="job-1")

        existing = self.upload("b")

        self.assertEqual(existing["document_id"], "a")
        self.assertEqual(existing["job_id"], "job-1")
        self.assertIsNone(registry.get_document("b"))

    def test_different_content_is_not_a_duplicate(self):
        self.upload("a", "h1")
        self.assertIsNone(self.upload("b", "h2"))

    def test_completed_documents_still_block_duplicates(self):
        self.upload("a")
        registry.update_status("a", "completed")
        self.assertEqual(self.upload("b")["document_id"], "a")

    def test_failed_document_can_be_uploaded_again(self):
        self.upload("a")
        registry.update_status("a", registry.FAILED_STATUS, error="boom")
        self.assertIsNone(self.upload("b"))

    def test_concurrent_identical_uploads_register_exactly_once(self):
        results = []

        def attempt(i):
            results.append(self.upload(f"doc{i}"))

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(sum(r is None for r in results), 1)
        with closing(registry.get_connection()) as conn:
            count = conn.execute("SELECT COUNT(*) FROM documents WHERE content_hash = 'h1'").fetchone()[0]
        self.assertEqual(count, 1)


class OwnershipTests(RegistryTestCase):
    def test_owner_queries_fail_closed_and_client_ids_only_narrow(self):
        registry.create_document("alice-doc", "a.pdf", "/a", "alice")
        registry.create_document("bob-doc", "b.pdf", "/b", "bob")
        registry.create_document("legacy-doc", "l.pdf", "/l")
        for document_id in ("alice-doc", "bob-doc", "legacy-doc"):
            registry.update_status(document_id, "completed")

        self.assertEqual(registry.list_document_ids_for_owner("alice"), ["alice-doc"])
        self.assertEqual(registry.list_document_ids_for_owner("alice", ["bob-doc", "alice-doc", "legacy-doc"]), ["alice-doc"])
        self.assertEqual(registry.list_document_ids_for_owner(None), [])
        self.assertEqual(registry.list_documents_for_owner("alice"), [
            {"document_id": "alice-doc", "filename": "a.pdf", "status": "completed"},
        ])
        self.assertEqual(registry.list_documents_for_owner(None), [])
        self.assertIsNone(registry.get_document_for_owner("bob-doc", "alice"))
        self.assertEqual(registry.get_pages_for_owner("legacy-doc", "alice"), [])
        page = {"page_no": 1, "page_type": "digital", "char_count": 1, "image_count": 0,
                "large_image_count": 0, "ocr_used": 0, "reason": "text"}
        registry.save_pages("bob-doc", [page])
        self.assertEqual(registry.get_pages_for_owner("bob-doc", "alice"), [])
        self.assertEqual(registry.get_pages_for_owner("bob-doc", "bob")[0]["page_no"], 1)

    def test_same_content_is_not_deduplicated_across_owners(self):
        self.assertIsNone(registry.register_upload("alice-doc", "a.pdf", "/a", "same", "alice"))
        self.assertIsNone(registry.register_upload("bob-doc", "b.pdf", "/b", "same", "bob"))
        self.assertEqual(registry.get_document_for_owner("bob-doc", "bob")["document_id"], "bob-doc")
        self.assertIsNone(registry.get_document_for_owner("bob-doc", "alice"))

    def test_legacy_assignment_requires_expected_count(self):
        registry.create_document("legacy-doc", "l.pdf", "/l")
        with self.assertRaises(ValueError):
            registry.assign_unowned_documents("alice", 0)
        self.assertEqual(registry.assign_unowned_documents("alice", 1), 1)
        registry.update_status("legacy-doc", "completed")
        self.assertEqual(registry.list_document_ids_for_owner("alice"), ["legacy-doc"])

    def test_retrieval_scope_excludes_incomplete_documents(self):
        registry.create_document("uploaded", "u.pdf", "/u", "alice")
        registry.create_document("failed", "f.pdf", "/f", "alice")
        registry.create_document("completed", "c.pdf", "/c", "alice")
        registry.update_status("failed", "failed")
        registry.update_status("completed", "completed")
        self.assertEqual(registry.list_document_ids_for_owner("alice"), ["completed"])

    def test_retrieval_scope_is_bounded(self):
        with patch.object(registry, "MAX_RETRIEVAL_DOCUMENTS", 2):
            for document_id in ("a", "b", "c"):
                registry.create_document(document_id, "x.pdf", "/x", "alice")
                registry.update_status(document_id, "completed")
            with self.assertRaises(registry.DocumentScopeTooLarge):
                registry.list_document_ids_for_owner("alice")


class MigrationTests(unittest.TestCase):
    def test_init_db_adds_low_yield_columns_to_existing_pages_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "documents.db"
            with closing(sqlite3.connect(db_path)) as conn:
                with conn:
                    conn.execute(
                        """
                    CREATE TABLE document_pages (
                        document_id TEXT NOT NULL, page_no INTEGER NOT NULL,
                        page_type TEXT NOT NULL, char_count INTEGER NOT NULL,
                        image_count INTEGER NOT NULL, large_image_count INTEGER NOT NULL,
                        ocr_used INTEGER NOT NULL, reason TEXT NOT NULL,
                        PRIMARY KEY (document_id, page_no))
                    """
                    )
                    conn.execute(
                        "INSERT INTO document_pages VALUES ('old', 1, 'digital', 5, 0, 0, 0, 'r')"
                    )

            with patch.object(registry, "DB_PATH", db_path):
                registry.init_db()
                registry.init_db()  # idempotent
                page = registry.get_pages("old")[0]

        self.assertIsNone(page["extracted_chars"])
        self.assertEqual(page["low_yield"], 0)

    def test_init_db_adds_content_hash_to_existing_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "documents.db"
            with closing(sqlite3.connect(db_path)) as conn:
                with conn:
                    conn.execute(
                        """
                    CREATE TABLE documents (
                        document_id TEXT PRIMARY KEY, filename TEXT NOT NULL,
                        file_path TEXT NOT NULL, status TEXT NOT NULL,
                        job_id TEXT, error TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)
                    """
                    )
                    conn.execute("INSERT INTO documents VALUES ('old', 'f.md', '/f.md', 'completed', NULL, NULL, 'now')")

            with patch.object(registry, "DB_PATH", db_path):
                registry.init_db()
                registry.init_db()  # idempotent

                old = registry.get_document("old")
                self.assertIsNone(old["content_hash"])  # old rows are kept, just unhashed
                self.assertIsNone(old["owner_id"])  # legacy rows are never assigned an arbitrary owner
                self.assertIsNone(registry.register_upload("new", "n.pdf", "/n.pdf", "h"))


class PageTests(RegistryTestCase):
    def page(self, page_no, page_type="digital", ocr_used=0):
        return {
            "page_no": page_no,
            "page_type": page_type,
            "char_count": 120,
            "image_count": 0,
            "large_image_count": 0,
            "ocr_used": ocr_used,
            "reason": "usable text layer",
        }

    def test_pages_round_trip_in_order(self):
        self.upload("a")
        registry.save_pages("a", [self.page(2, "scanned", 1), self.page(1)])

        pages = registry.get_pages("a")
        self.assertEqual([p["page_no"] for p in pages], [1, 2])
        self.assertEqual([p["page_type"] for p in pages], ["digital", "scanned"])
        self.assertEqual([p["ocr_used"] for p in pages], [0, 1])

    def test_low_yield_fields_round_trip_and_default(self):
        self.upload("a")
        flagged = {**self.page(1), "extracted_chars": 12, "low_yield": 1}
        registry.save_pages("a", [flagged, self.page(2)])

        first, second = registry.get_pages("a")
        self.assertEqual((first["extracted_chars"], first["low_yield"]), (12, 1))
        self.assertEqual((second["extracted_chars"], second["low_yield"]), (None, 0))

    def test_saving_again_replaces_previous_pages(self):
        self.upload("a")
        registry.save_pages("a", [self.page(1), self.page(2)])
        registry.save_pages("a", [self.page(1, "scanned", 1)])

        pages = registry.get_pages("a")
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page_type"], "scanned")

    def test_pages_are_kept_per_document(self):
        self.upload("a", "h1")
        self.upload("b", "h2")
        registry.save_pages("a", [self.page(1)])
        self.assertEqual(registry.get_pages("b"), [])


if __name__ == "__main__":
    unittest.main()
