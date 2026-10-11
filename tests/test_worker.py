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

        self.convert_pdf = MagicMock()
        self.convert_pdf.return_value.page_records.return_value = [
            {"page_no": 1, "page_type": "digital", "char_count": 5, "image_count": 0,
             "large_image_count": 0, "ocr_used": 0, "reason": "usable text layer"}
        ]
        # worker imports convert lazily, so Docling is not needed here.
        fake_module = SimpleNamespace(convert_pdf=self.convert_pdf)
        patcher = patch.dict(sys.modules, {"app.ingestion.convert": fake_module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pdf_is_converted_and_page_data_saved(self):
        result = worker.process_document("doc", "/uploads/report.PDF")

        self.convert_pdf.assert_called_once_with("/uploads/report.PDF")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(registry.get_pages("doc")[0]["page_type"], "digital")

    def test_non_pdf_skips_conversion(self):
        worker.process_document("doc", "/uploads/notes.md")
        self.convert_pdf.assert_not_called()
        self.assertEqual(registry.get_document("doc")["status"], "completed")

    def test_conversion_failure_marks_document_failed(self):
        self.convert_pdf.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            worker.process_document("doc", "/uploads/report.pdf")

        document = registry.get_document("doc")
        self.assertEqual(document["status"], registry.FAILED_STATUS)
        self.assertEqual(document["error"], "boom")


if __name__ == "__main__":
    unittest.main()
