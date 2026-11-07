import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.db import registry
from app.ingestion import worker


class _WorkerFixture(unittest.TestCase):
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
        self.notes.write_text("# Notes\n\nhello")
        self.unsupported = Path(tmp.name) / "sheet.xlsx"
        self.unsupported.write_bytes(b"PK")
        self.dir = Path(tmp.name)

        self.convert_pdf = MagicMock()
        self.convert_pdf.return_value.page_records.return_value = [
            {"page_no": 1, "page_type": "digital", "char_count": 5, "image_count": 0,
             "large_image_count": 0, "ocr_used": 0, "reason": "usable text layer"}
        ]
        self.ingest = MagicMock(return_value={"chunk_count": 3})
        self.ingest_text = MagicMock(return_value={"chunk_count": 2})
        self.convert_office = MagicMock(return_value="docling-doc")

        # The worker imports convert and pipeline lazily, so neither Docling
        # nor the embedding model is needed here.
        patcher = patch.dict(
            sys.modules,
            {
                "app.ingestion.convert": SimpleNamespace(
                    convert_pdf=self.convert_pdf, convert_office=self.convert_office
                ),
                "app.ingestion.pipeline": SimpleNamespace(
                    ingest_converted_document=self.ingest,
                    ingest_text_document=self.ingest_text,
                ),
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class ProcessDocumentTests(_WorkerFixture):
    def test_pdf_is_converted_ingested_and_completed(self):
        result = worker.process_document("doc", str(self.pdf))

        self.convert_pdf.assert_called_once_with(str(self.pdf))
        self.ingest.assert_called_once()
        self.assertEqual(self.ingest.call_args.kwargs["document_version"], "hash")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["chunk_count"], 3)
        self.assertEqual(registry.get_document("doc")["status"], "completed")
        self.assertEqual(registry.get_pages("doc")[0]["page_type"], "digital")

    def test_unsupported_format_is_rejected_without_conversion(self):
        with self.assertRaises(NotImplementedError):
            worker.process_document("doc", str(self.unsupported))

        self.convert_pdf.assert_not_called()
        self.convert_office.assert_not_called()
        self.ingest_text.assert_not_called()
        self.assertEqual(registry.get_document("doc")["status"], registry.FAILED_STATUS)

    def test_markdown_uses_the_lightweight_path_without_docling(self):
        result = worker.process_document("doc", str(self.notes))

        self.convert_pdf.assert_not_called()
        self.convert_office.assert_not_called()
        kw = self.ingest_text.call_args.kwargs
        self.assertEqual((kw["fmt"], kw["document_version"], kw["text"]), ("markdown", "hash", "# Notes\n\nhello"))
        self.assertEqual(kw["title"], "f")  # registry filename
        self.assertEqual(result["chunk_count"], 2)
        self.assertEqual(registry.get_document("doc")["status"], "completed")

    def test_txt_uses_the_text_path(self):
        path = self.dir / "a.TXT"
        path.write_text("plain words")
        worker.process_document("doc", str(path))
        self.assertEqual(self.ingest_text.call_args.kwargs["fmt"], "text")

    def test_html_goes_through_docling_office_conversion(self):
        path = self.dir / "page.html"
        path.write_text("<html><body><h1>T</h1><p>x</p></body></html>")
        worker.process_document("doc", str(path))
        self.convert_office.assert_called_once_with(str(path))
        self.assertEqual(self.ingest.call_args.kwargs["document"], "docling-doc")
        self.convert_pdf.assert_not_called()

    def test_docx_is_checked_before_conversion(self):
        path = self.dir / "bad.docx"
        path.write_bytes(b"not a zip")
        with self.assertRaises(ValueError):
            worker.process_document("doc", str(path))
        self.convert_office.assert_not_called()
        self.assertEqual(registry.get_document("doc")["status"], registry.FAILED_STATUS)

    def test_valid_docx_is_converted(self):
        import zipfile
        path = self.dir / "ok.docx"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", "<Types/>")
            z.writestr("word/document.xml", "<w:document>" + "x" * 100 + "</w:document>")
        worker.process_document("doc", str(path))
        self.convert_office.assert_called_once_with(str(path))

    def test_oversized_text_file_is_rejected(self):
        from app.ingestion import formats
        with patch.object(formats, "MAX_TEXT_BYTES", 4):
            with self.assertRaises(formats.IngestionPolicyError):
                worker.process_document("doc", str(self.notes))
        self.ingest_text.assert_not_called()
        self.assertEqual(registry.get_document("doc")["status"], registry.FAILED_STATUS)

    def test_pdf_preflight_still_runs_in_convert_pdf(self):
        # The PDF branch hands the file to convert_pdf, which owns the structural preflight;
        # a policy rejection from it fails the document instead of falling back.
        self.convert_pdf.side_effect = ValueError("PDF has 5000 pages; the limit is 1000")
        with self.assertRaises(ValueError):
            worker.process_document("doc", str(self.pdf))
        self.ingest.assert_not_called()
        self.assertEqual(registry.get_document("doc")["status"], registry.FAILED_STATUS)


class ReindexDocumentTests(_WorkerFixture):
    def setUp(self):
        super().setUp()
        registry.register_upload("done", "Report_2025.md", str(self.notes), "h2", owner_id="alice")
        registry.update_status("done", "completed")

    def test_reindex_uses_a_fresh_version_and_keeps_status(self):
        out = worker.reindex_document("done")
        kw = self.ingest_text.call_args.kwargs
        self.assertTrue(kw["document_version"].startswith("h2:reindex-"))
        self.assertEqual(kw["title"], "Report 2025")
        self.assertEqual(out["document_version"], kw["document_version"])
        self.assertEqual(registry.get_document("done")["status"], "completed")

    def test_reindex_failure_leaves_document_completed(self):
        self.ingest_text.side_effect = RuntimeError("store down")
        with self.assertRaises(RuntimeError):
            worker.reindex_document("done")
        doc = registry.get_document("done")
        self.assertEqual(doc["status"], "completed")
        self.assertIsNone(doc["error"])

    def test_reindex_refuses_documents_that_are_not_completed(self):
        with self.assertRaises(ValueError):
            worker.reindex_document("doc")  # still queued
        with self.assertRaises(KeyError):
            worker.reindex_document("missing")

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
