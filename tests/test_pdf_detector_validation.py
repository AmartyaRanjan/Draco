"""Regression tests: PDF detector sampling config must be validated.

DetectorConfig(max_pages=0) (or negative) yields zero sampled pages. The
empty inspection then crashes downstream: PdfInspection.digital_ratio divides
by zero, convert_pdf divides by len(page_ocr), and all_digital is vacuously
True for zero pages. Sampling limits must be rejected early and empty
inspections must not crash.
"""
import unittest

from app.ingestion.pdf_detector import (
    DetectorConfig,
    PdfInspection,
    _sample_indices,
    inspect_pdf,
)


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


if __name__ == "__main__":
    unittest.main()
