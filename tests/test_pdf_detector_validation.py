"""Regression tests: PDF detector sampling config must be validated.

DetectorConfig(max_pages=0) (or negative) yields zero sampled pages. The
empty inspection then crashes downstream: PdfInspection.digital_ratio divides
by zero, convert_pdf divides by len(page_ocr), and all_digital is vacuously
True for zero pages. Sampling limits must be rejected early and empty
inspections must not crash.
"""
import unittest
import tempfile
from pathlib import Path

from app.ingestion.pdf_detector import (
    DetectorConfig,
    PdfDetectionError,
    PdfInspection,
    _sample_indices,
    inspect_pdf,
)
from tests.pdf_factory import write_pdf


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


if __name__ == "__main__":
    unittest.main()
