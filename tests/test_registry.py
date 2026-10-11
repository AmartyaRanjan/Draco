import sqlite3
import tempfile
import threading
import unittest
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
        with registry.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM documents WHERE content_hash = 'h1'").fetchone()[0]
        self.assertEqual(count, 1)


class MigrationTests(unittest.TestCase):
    def test_init_db_adds_content_hash_to_existing_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "documents.db"
            with sqlite3.connect(db_path) as conn:
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
