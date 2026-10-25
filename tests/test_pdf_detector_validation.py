"""Regression tests: PDF detector sampling config must be validated.

DetectorConfig(max_pages=0) (or negative) yields zero sampled pages. The
empty inspection then crashes downstream: PdfInspection.digital_ratio divides
by zero, convert_pdf divides by len(page_ocr), and all_digital is vacuously
True for zero pages. Sampling limits must be rejected early and empty
inspections must not crash.

Policy limits (bytes, document pages, images per page) must raise
PdfPolicyError, which convert_pdf re-raises instead of "converting anyway
with OCR" — the fallback that made the limits bypassable.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ingestion import pdf_detector
from app.ingestion.pdf_detector import (
    DetectorConfig,
    PdfDetectionError,
    PdfInspection,
    PdfPolicyError,
    _sample_indices,
    inspect_pdf,
)

from tests.pdf_factory import LARGE_IMAGE, write_pdf


class DetectorSamplingValidationTests(unittest.TestCase):
    def test_zero_max_pages_rejected(self):
        with self.assertRaises(ValueError):
            DetectorConfig(max_pages=0)
        with self.assertRaises(ValueError):
            _sample_indices(10, 0)

    def test_negative_max_pages_rejected(self):
        with self.assertRaises(ValueError):
            DetectorConfig(max_pages=-1)
        with self.assertRaises(ValueError):
            _sample_indices(10, -2)

    def test_empty_inspection_does_not_crash(self):
        inspection = PdfInspection(page_count=5, pages=[])
        self.assertEqual(inspection.digital_ratio, 0.0)
        self.assertFalse(inspection.all_digital)
        self.assertTrue(inspection.sampled)

    def test_valid_sampling_still_works(self):
        # Existing behavior preserved: 10 pages sampled to 4.
        self.assertEqual(_sample_indices(10, 4), [0, 2, 5, 7])
        DetectorConfig(max_pages=4)

    def test_document_page_limit_is_checked_before_conversion(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_pdf(Path(temp) / "many.pdf", [{"text": "page"}] * 2)
            with self.assertRaises(PdfDetectionError) as ctx:
                inspect_pdf(path, DetectorConfig(max_document_pages=1))
            self.assertIn("page", str(ctx.exception).lower())

    def test_page_limit_raises_pdf_policy_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_pdf(Path(temp) / "many.pdf", [{"text": "page"}] * 2)
            with self.assertRaises(PdfPolicyError) as ctx:
                inspect_pdf(path, DetectorConfig(max_document_pages=1))
            self.assertIsInstance(ctx.exception, PdfDetectionError)
            self.assertIn("limit", str(ctx.exception))

    def test_byte_limit_raises_pdf_policy_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_pdf(Path(temp) / "doc.pdf", [{"text": "page"}])
            with patch.object(pdf_detector, "MAX_PDF_BYTES", 10):
                with self.assertRaises(PdfPolicyError):
                    inspect_pdf(path)

    def test_image_limit_raises_pdf_policy_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_pdf(Path(temp) / "img.pdf", [{"image": LARGE_IMAGE}])
            with self.assertRaises(PdfPolicyError):
                inspect_pdf(path, DetectorConfig(max_images_per_page=0))

    def test_malformed_pdf_still_raises_plain_detection_error(self):
        path = Path("does-not-matter.pdf")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.pdf"
            path.write_bytes(b"not a pdf")
            with self.assertRaises(PdfDetectionError) as ctx:
                inspect_pdf(path)
            self.assertNotIsInstance(ctx.exception, PdfPolicyError)


if __name__ == "__main__":
    unittest.main()
