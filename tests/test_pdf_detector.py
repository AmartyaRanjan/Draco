import tempfile
import unittest
from pathlib import Path

from app.ingestion.pdf_detector import (
    DetectorConfig,
    PdfDetectionError,
    inspect_pdf,
)
from tests.pdf_factory import LARGE_IMAGE, LONG_TEXT, SMALL_IMAGE, write_pdf

DIGITAL = {"text": LONG_TEXT}
SCANNED = {"image": LARGE_IMAGE}


class PdfDetectorTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def pdf(self, pages, **kwargs):
        return write_pdf(self.dir / "test.pdf", pages, **kwargs)

    def types(self, inspection):
        return [p.page_type for p in inspection.pages]

    def test_digital_pdf(self):
        result = inspect_pdf(self.pdf([DIGITAL] * 3))
        self.assertEqual(self.types(result), ["digital"] * 3)
        self.assertTrue(result.all_digital)
        self.assertEqual(result.digital_ratio, 1.0)
        self.assertEqual(result.page_count, 3)
        self.assertFalse(result.sampled)

    def test_scanned_pdf(self):
        result = inspect_pdf(self.pdf([SCANNED] * 3))
        self.assertEqual(self.types(result), ["scanned"] * 3)
        self.assertFalse(result.all_digital)
        self.assertEqual(result.pages[0].char_count, 0)

    def test_mixed_pdf_is_classified_per_page(self):
        result = inspect_pdf(self.pdf([DIGITAL, SCANNED, DIGITAL]))
        self.assertEqual(self.types(result), ["digital", "scanned", "digital"])
        self.assertEqual(result.count("digital"), 2)
        self.assertEqual(result.count("scanned"), 1)
        self.assertEqual(result.count("ambiguous"), 0)
        self.assertFalse(result.all_digital)
        self.assertAlmostEqual(result.digital_ratio, 2 / 3)

    def test_digital_text_with_large_image_is_ambiguous(self):
        # The detector cannot see what the image contains; it only flags the risk.
        result = inspect_pdf(self.pdf([{"text": LONG_TEXT, "image": LARGE_IMAGE}]))
        self.assertEqual(self.types(result), ["ambiguous"])
        self.assertEqual(result.pages[0].large_image_count, 1)
        self.assertFalse(result.all_digital)

    def test_digital_text_with_small_image_stays_digital(self):
        result = inspect_pdf(self.pdf([{"text": LONG_TEXT, "image": SMALL_IMAGE}]))
        self.assertEqual(self.types(result), ["digital"])
        self.assertEqual(result.pages[0].image_count, 1)

    def test_blank_page_is_ambiguous(self):
        result = inspect_pdf(self.pdf([{}]))
        self.assertEqual(self.types(result), ["ambiguous"])

    def test_short_text_is_ambiguous(self):
        result = inspect_pdf(self.pdf([{"text": "x" * 50}]))
        self.assertEqual(self.types(result), ["ambiguous"])

    def test_thresholds_are_configurable(self):
        path = self.pdf([{"text": "x" * 50}])
        config = DetectorConfig(min_digital_chars=40)
        self.assertEqual(self.types(inspect_pdf(path, config)), ["digital"])

    def test_large_documents_are_sampled_evenly(self):
        path = self.pdf([DIGITAL] * 10)
        result = inspect_pdf(path, DetectorConfig(max_pages=4))
        self.assertTrue(result.sampled)
        self.assertEqual(result.page_count, 10)
        self.assertEqual([p.index for p in result.pages], [0, 2, 5, 7])

    def test_empty_pdf_raises(self):
        with self.assertRaisesRegex(PdfDetectionError, "no pages"):
            inspect_pdf(self.pdf([]))

    def test_malformed_pdf_raises(self):
        path = self.dir / "bad.pdf"
        path.write_bytes(b"this is not a pdf")
        with self.assertRaises(PdfDetectionError):
            inspect_pdf(path)

    def test_missing_file_raises(self):
        with self.assertRaises(PdfDetectionError):
            inspect_pdf(self.dir / "missing.pdf")

    def test_password_protected_pdf_raises(self):
        with self.assertRaisesRegex(PdfDetectionError, "encrypted"):
            inspect_pdf(self.pdf([DIGITAL], password="secret"))

    def test_empty_password_pdf_is_readable(self):
        result = inspect_pdf(self.pdf([DIGITAL], password=""))
        self.assertTrue(result.all_digital)


if __name__ == "__main__":
    unittest.main()
