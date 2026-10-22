"""Regression tests: upload must reject filenames that sanitize to empty.

upload_document() strips any client-supplied directory via Path(name).name,
but only checked the raw filename for emptiness. A name like "/" or "."
passes that check and sanitizes to "", creating a storage file named
"{uuid}_" and a registry row with an empty filename. Such names must be
rejected with 400 like a missing filename.
"""
import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException, UploadFile

from app.api import routes
from app.db import registry


class EmptySafeFilenameTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.queue = MagicMock()
        self.queue.enqueue.side_effect = lambda *a, **k: SimpleNamespace(id="job-1")
        for patcher in (
            patch.object(registry, "DB_PATH", self.root / "documents.db"),
            patch.object(routes, "UPLOAD_DIR", self.root / "uploads"),
            patch.object(routes, "ingestion_queue", self.queue),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        registry.init_db()

    def upload(self, filename):
        file = UploadFile(file=io.BytesIO(b"%PDF-1.7 hello"), filename=filename)
        return asyncio.run(routes.upload_document(file, "alice"))

    def test_path_only_filename_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            self.upload("/")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(self.queue.enqueue.call_count, 0)
        # No storage file and no registry row were created.
        uploads = self.root / "uploads"
        self.assertFalse(uploads.exists() and any(uploads.iterdir()))

    def test_dot_filename_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            self.upload(".")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(self.queue.enqueue.call_count, 0)

    def test_normal_filename_still_accepted(self):
        response = self.upload("a.pdf")
        self.assertEqual(response["status"], "queued")
        self.assertEqual(response["filename"], "a.pdf")
        self.assertEqual(self.queue.enqueue.call_count, 1)


if __name__ == "__main__":
    unittest.main()
