import asyncio
import hashlib
import io
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from rq.exceptions import DuplicateJobError

from app.api import routes
from app.db import registry


PDF = b"%PDF-1.7\n"  # uploads are sniffed for the PDF header


class FakeJob:
    def __init__(self, job_id):
        self.id = job_id
        self.status = "queued"

    def get_status(self, refresh=True):
        return self.status


class FakeQueue:
    def __init__(self):
        self.jobs = {}
        self.enqueue_calls = 0
        self.fail_next = False

    def enqueue(self, *args, job_id, **kwargs):
        self.enqueue_calls += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("redis down")
        if job_id in self.jobs:
            raise DuplicateJobError(job_id)
        job = FakeJob(job_id)
        self.jobs[job_id] = job
        return job

    def fetch_job(self, job_id):
        return self.jobs.get(job_id)


class UploadRouteTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

        self.queue = FakeQueue()

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
        self.assertEqual(self.queue.enqueue_calls, 1)
        self.assertEqual(registry.get_ingestion_outbox(response["document_id"])["state"], "queued")

    def test_duplicate_upload_returns_existing_document_without_new_job(self):
        first = self.upload(b"same bytes", "first.pdf")
        second = self.upload(b"same bytes", "renamed.pdf")

        self.assertEqual(second.status_code, 200)
        body = second.body.decode()
        self.assertIn(first["document_id"], body)
        self.assertIn('"duplicate":true', body)
        self.assertEqual(self.queue.enqueue_calls, 1)
        self.assertEqual(len(self.stored_files()), 1)  # the extra copy was discarded

    def test_different_content_is_queued_separately(self):
        self.upload(b"one")
        self.upload(b"two")
        self.assertEqual(self.queue.enqueue_calls, 2)

    def test_failed_enqueue_preserves_persisted_upload_and_retry_recovers_it(self):
        self.queue.fail_next = True
        with self.assertRaises(HTTPException) as ctx:
            self.upload(b"payload")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(len(self.stored_files()), 1)
        document = registry.get_document(next(iter(
            row["document_id"] for row in registry.list_documents_for_owner("alice")
        )))
        self.assertEqual(document["status"], "uploaded")
        self.assertEqual(registry.get_ingestion_outbox(document["document_id"])["state"], "pending")

        with closing(registry.get_connection()) as conn, conn:
            conn.execute("UPDATE ingestion_outbox SET next_attempt_at=0")
        self.queue.fail_next = False
        retry = self.upload(b"payload")
        self.assertEqual(retry.status_code, 200)
        self.assertIn('"duplicate":true', retry.body.decode())
        self.assertEqual(len(self.stored_files()), 1)
        self.assertEqual(registry.get_ingestion_outbox(document["document_id"])["state"], "queued")

    def test_non_pdf_is_rejected_before_queueing(self):
        for name, data in (("notes.txt", PDF + b"x"), ("fake.pdf", b"MZ\x90\x00 not a pdf")):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(routes.upload_document(UploadFile(file=io.BytesIO(data), filename=name)))
            self.assertEqual(ctx.exception.status_code, 415)
        self.assertEqual(self.queue.enqueue_calls, 0)
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
