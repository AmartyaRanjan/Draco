import asyncio
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException, UploadFile

from app.api import routes
from app.db import registry


PDF = b"%PDF-1.7\n"  # uploads are sniffed for the PDF header


class UploadRouteTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

        self.queue = MagicMock()
        self.queue.enqueue.side_effect = lambda *a, **k: SimpleNamespace(id=f"job-{self.queue.enqueue.call_count}")

        for patcher in (
            patch.object(registry, "DB_PATH", self.root / "documents.db"),
            patch.object(routes, "UPLOAD_DIR", self.root / "uploads"),
            patch.object(routes, "ingestion_queue", self.queue),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        registry.init_db()

    def upload(self, data: bytes, filename="a.pdf"):
        file = UploadFile(file=io.BytesIO(PDF + data), filename=filename)
        return asyncio.run(routes.upload_document(file, "alice"))

    def stored_files(self):
        return sorted(p.name for p in (self.root / "uploads").iterdir())

    def test_new_upload_is_queued_and_hashed(self):
        response = self.upload(b"hello")

        self.assertEqual(response["status"], "queued")
        self.assertFalse(response["duplicate"])
        document = registry.get_document(response["document_id"])
        self.assertEqual(document["content_hash"], hashlib.sha256(PDF + b"hello").hexdigest())
        self.assertEqual(self.queue.enqueue.call_count, 1)

    def test_duplicate_upload_returns_existing_document_without_new_job(self):
        first = self.upload(b"same bytes", "first.pdf")
        second = self.upload(b"same bytes", "renamed.pdf")

        self.assertEqual(second.status_code, 200)
        body = second.body.decode()
        self.assertIn(first["document_id"], body)
        self.assertIn('"duplicate":true', body)
        self.assertEqual(self.queue.enqueue.call_count, 1)
        self.assertEqual(len(self.stored_files()), 1)  # the extra copy was discarded

    def test_different_content_is_queued_separately(self):
        self.upload(b"one")
        self.upload(b"two")
        self.assertEqual(self.queue.enqueue.call_count, 2)

    def test_failed_enqueue_marks_document_failed_so_retry_is_not_a_duplicate(self):
        self.queue.enqueue.side_effect = RuntimeError("redis down")
        with self.assertRaises(HTTPException) as ctx:
            self.upload(b"payload")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(self.stored_files(), [])

        self.queue.enqueue.side_effect = lambda *a, **k: SimpleNamespace(id="job-retry")
        retry = self.upload(b"payload")
        self.assertEqual(retry["status"], "queued")
        self.assertFalse(retry["duplicate"])

    def test_non_pdf_is_rejected_before_queueing(self):
        for name, data in (("notes.txt", PDF + b"x"), ("fake.pdf", b"MZ\x90\x00 not a pdf")):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(routes.upload_document(UploadFile(file=io.BytesIO(data), filename=name)))
            self.assertEqual(ctx.exception.status_code, 415)
        self.assertEqual(self.queue.enqueue.call_count, 0)
        self.assertEqual(self.stored_files(), [])

    def test_pages_endpoint(self):
        response = self.upload(b"x")
        document_id = response["document_id"]
        registry.save_pages(
            document_id,
            [{"page_no": 1, "page_type": "digital", "char_count": 5, "image_count": 0,
              "large_image_count": 0, "ocr_used": 0, "reason": "usable text layer"}],
        )
        self.assertEqual(routes.document_pages(document_id, "alice")[0]["page_type"], "digital")
        with self.assertRaises(HTTPException):
            routes.document_pages("missing", "alice")


if __name__ == "__main__":
    unittest.main()
