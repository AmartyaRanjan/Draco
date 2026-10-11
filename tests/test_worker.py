import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.db import registry
from app.ingestion import worker


class ProcessDocumentTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = patch.object(registry, "DB_PATH", Path(tmp.name) / "documents.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        registry.init_db()
        registry.register_upload("doc", "f", "/f", "hash")

        # The worker checks that the uploaded file exists.
        self.pdf = Path(tmp.name) / "report.PDF"
        self.pdf.write_bytes(b"%PDF-1.4")
        self.notes = Path(tmp.name) / "notes.md"
        self.notes.write_text("hello")

        self.convert_pdf = MagicMock()
        self.convert_pdf.return_value.page_records.return_value = [
            {"page_no": 1, "page_type": "digital", "char_count": 5, "image_count": 0,
             "large_image_count": 0, "ocr_used": 0, "reason": "usable text layer"}
        ]
        self.ingest = MagicMock(return_value={"chunk_count": 3})

        # The worker imports convert and pipeline lazily, so neither Docling
        # nor the embedding model is needed here.
        patcher = patch.dict(
            sys.modules,
            {
                "app.ingestion.convert": SimpleNamespace(convert_pdf=self.convert_pdf),
                "app.ingestion.pipeline": SimpleNamespace(
                    ingest_converted_document=self.ingest
                ),
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pdf_is_converted_ingested_and_completed(self):
        result = worker.process_document("doc", str(self.pdf))

        self.convert_pdf.assert_called_once_with(str(self.pdf))
        self.ingest.assert_called_once()
        self.assertEqual(self.ingest.call_args.kwargs["document_version"], "hash")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["chunk_count"], 3)
        self.assertEqual(registry.get_document("doc")["status"], "completed")
        self.assertEqual(registry.get_pages("doc")[0]["page_type"], "digital")

    def test_non_pdf_is_rejected_without_conversion(self):
        with self.assertRaises(NotImplementedError):
            worker.process_document("doc", str(self.notes))

        self.convert_pdf.assert_not_called()
        self.assertEqual(registry.get_document("doc")["status"], registry.FAILED_STATUS)

    def test_missing_file_marks_document_failed(self):
        with self.assertRaises(FileNotFoundError):
            worker.process_document("doc", "/uploads/missing.pdf")

        self.convert_pdf.assert_not_called()
        self.assertEqual(registry.get_document("doc")["status"], registry.FAILED_STATUS)

    def test_conversion_failure_marks_document_failed(self):
        self.convert_pdf.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            worker.process_document("doc", str(self.pdf))

        document = registry.get_document("doc")
        self.assertEqual(document["status"], registry.FAILED_STATUS)
        self.assertEqual(document["error"], "boom")

    def test_ingestion_failure_marks_document_failed(self):
        self.ingest.side_effect = RuntimeError("store down")
        with self.assertRaises(RuntimeError):
            worker.process_document("doc", str(self.pdf))

        document = registry.get_document("doc")
        self.assertEqual(document["status"], registry.FAILED_STATUS)
        self.assertEqual(document["error"], "store down")


if __name__ == "__main__":
    unittest.main()
