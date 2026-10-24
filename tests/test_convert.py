import importlib.util
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Docling is heavy; if it is not installed, stub it so routing can still be tested.
DOCLING_INSTALLED = (
    importlib.util.find_spec("docling") is not None
    and importlib.util.find_spec("docling_core") is not None
)
if not DOCLING_INSTALLED:
    for name in (
        "docling",
        "docling.datamodel",
        "docling.datamodel.base_models",
        "docling.datamodel.pipeline_options",
        "docling.document_converter",
        "docling_core",
        "docling_core.types",
        "docling_core.types.doc",
    ):
        sys.modules[name] = MagicMock()

from app.ingestion import convert  # noqa: E402
from tests.pdf_factory import LARGE_IMAGE, LONG_TEXT, write_pdf  # noqa: E402

DIGITAL = {"text": LONG_TEXT}
SCANNED = {"image": LARGE_IMAGE}


class ConvertTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        convert._get_converter.cache_clear()
        self.addCleanup(convert._get_converter.cache_clear)

    def make_pdf(self, pages):
        return write_pdf(self.dir / "test.pdf", pages)


class ConvertRoutingTests(ConvertTestCase):
    """Real detector, mocked Docling: checks which options reach the converter.

    These tests describe document routing, which is no longer the default.
    """

    def setUp(self):
        super().setUp()
        patcher = patch.dict("os.environ", {"PDF_ROUTING": "document"})
        patcher.start()
        self.addCleanup(patcher.stop)
        patchers = {
            name: patch.object(convert, name)
            for name in ("PdfPipelineOptions", "PdfFormatOption", "DocumentConverter")
        }
        self.mocks = {name: p.start() for name, p in patchers.items()}
        for p in patchers.values():
            self.addCleanup(p.stop)
        self.converter = self.mocks["DocumentConverter"].return_value

    def assert_ocr(self, expected: bool):
        options = self.mocks["PdfPipelineOptions"]
        options.assert_called_once()
        kwargs = options.call_args.kwargs
        self.assertIs(kwargs["do_ocr"], expected)
        # OCR routing must not disable the rest of the standard pipeline.
        self.assertIs(kwargs["do_table_structure"], True)
        self.assertIs(kwargs["generate_page_images"], False)

    def test_digital_pdf_disables_ocr(self):
        convert.convert_document(self.make_pdf([DIGITAL] * 2))
        self.assert_ocr(False)

    def test_scanned_pdf_enables_ocr(self):
        convert.convert_document(self.make_pdf([SCANNED]))
        self.assert_ocr(True)

    def test_mixed_pdf_enables_ocr_for_whole_document(self):
        convert.convert_document(self.make_pdf([DIGITAL, SCANNED]))
        self.assert_ocr(True)

    def test_digital_text_with_large_image_enables_ocr(self):
        convert.convert_document(self.make_pdf([{"text": LONG_TEXT, "image": LARGE_IMAGE}]))
        self.assert_ocr(True)

    def test_empty_pdf_falls_back_to_ocr(self):
        with self.assertLogs(convert.logger, logging.WARNING):
            conversion = convert.convert_pdf(self.make_pdf([]))
        self.assert_ocr(True)
        self.assertIsNone(conversion.inspection)
        self.assertEqual(conversion.page_ocr, {})

    def test_malformed_pdf_falls_back_to_ocr(self):
        path = self.dir / "bad.pdf"
        path.write_bytes(b"not a pdf")
        with self.assertLogs(convert.logger, logging.WARNING):
            convert.convert_document(path)
        self.assert_ocr(True)

    def test_detector_error_never_means_digital(self):
        from app.ingestion.pdf_detector import PdfDetectionError

        with patch.object(convert, "inspect_pdf", side_effect=PdfDetectionError("x")):
            convert.convert_document("whatever.pdf")
        self.assert_ocr(True)

    def test_policy_limit_rejection_is_never_converted(self):
        # A structural-limit rejection must propagate, not fall back to OCR:
        # converting the rejected file anyway would defeat the limit.
        from app.ingestion.pdf_detector import PdfPolicyError

        with patch.object(
            convert, "inspect_pdf", side_effect=PdfPolicyError("PDF has 3 pages; limit is 1")
        ):
            with self.assertRaises(PdfPolicyError):
                convert.convert_document("whatever.pdf")
        self.converter.convert.assert_not_called()

    def test_returns_docling_document(self):
        result = convert.convert_document(self.make_pdf([DIGITAL]))
        self.assertIs(result, self.converter.convert.return_value.document)

    def test_conversion_failure_is_logged_and_reraised(self):
        self.converter.convert.side_effect = RuntimeError("boom")
        with self.assertLogs(convert.logger, logging.ERROR):
            with self.assertRaises(RuntimeError):
                convert.convert_document(self.make_pdf([DIGITAL]))

    def test_converter_is_reused_per_ocr_mode(self):
        digital = self.make_pdf([DIGITAL])
        convert.convert_document(digital)
        convert.convert_document(digital)
        self.assertEqual(self.mocks["DocumentConverter"].call_count, 1)

        scanned = self.make_pdf([SCANNED])
        convert.convert_document(scanned)
        convert.convert_document(scanned)
        self.assertEqual(self.mocks["DocumentConverter"].call_count, 2)

    def test_invalid_routing_is_rejected(self):
        with self.assertRaises(ValueError):
            convert.convert_pdf(self.make_pdf([DIGITAL]), routing="bogus")

    def test_page_records_describe_every_page(self):
        conversion = convert.convert_pdf(self.make_pdf([DIGITAL, SCANNED]))
        records = conversion.page_records()
        self.assertEqual([r["page_no"] for r in records], [1, 2])
        self.assertEqual([r["page_type"] for r in records], ["digital", "scanned"])
        # Document routing converted everything with OCR, and the records say so.
        self.assertEqual([r["ocr_used"] for r in records], [1, 1])
        self.assertGreater(records[0]["char_count"], 0)
        self.assertEqual(conversion.routing, "document")


class ConvertPageRoutingTests(ConvertTestCase):
    """Page-level routing with a fake converter per OCR mode."""

    def setUp(self):
        super().setUp()
        self.calls = []  # (do_ocr, page_range)
        self.merged = MagicMock(name="merged")
        patchers = [
            patch.object(convert, "_get_converter", side_effect=self.fake_converter),
            patch.object(convert, "DoclingDocument"),
        ]
        self.mocks = [p.start() for p in patchers]
        for p in patchers:
            self.addCleanup(p.stop)
        # These tests use short PDFs where most pages need OCR.
        patcher = patch.object(convert, "MAX_OCR_FRACTION", 1.0)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.docling_document = self.mocks[1]
        self.docling_document.concatenate.return_value = self.merged
        self.merged.pages = {}
        self.status = convert.ConversionStatus.SUCCESS

    def fake_converter(self, do_ocr):
        def run(path, page_range=None):
            self.calls.append((do_ocr, page_range))
            first, last = page_range or (1, self.total_pages)
            document = MagicMock(name=f"doc{first}-{last}")
            document.pages = {n: object() for n in range(first, last + 1)}
            return SimpleNamespace(status=self.status, document=document)

        return SimpleNamespace(convert=run)

    def convert(self, pages):
        self.total_pages = len(pages)
        self.merged.pages = {n: object() for n in range(1, len(pages) + 1)}
        return convert.convert_pdf(self.make_pdf(pages), routing="page")

    def test_mixed_pdf_converts_runs_with_matching_ocr_mode(self):
        conversion = self.convert([DIGITAL, SCANNED, SCANNED, DIGITAL])

        self.assertEqual(self.calls, [(False, (1, 1)), (True, (2, 3)), (False, (4, 4))])
        self.assertEqual(conversion.routing, "page")
        self.assertEqual(conversion.page_ocr, {1: False, 2: True, 3: True, 4: False})
        self.assertIs(conversion.document, self.merged)
        docs = self.docling_document.concatenate.call_args.args[0]
        self.assertEqual(len(docs), 3)

    def test_merged_document_keeps_origin_of_first_run(self):
        self.convert([DIGITAL, SCANNED])
        first = self.docling_document.concatenate.call_args.args[0][0]
        self.assertIs(self.merged.origin, first.origin)
        self.assertIs(self.merged.name, first.name)

    def test_uniform_pdf_is_converted_whole(self):
        self.convert([DIGITAL, DIGITAL])
        self.assertEqual(self.calls, [(False, None)])
        self.docling_document.concatenate.assert_not_called()

        self.calls.clear()
        self.convert([SCANNED, SCANNED])
        self.assertEqual(self.calls, [(True, None)])

    def test_partial_conversion_falls_back_to_whole_document_ocr(self):
        self.status = convert.ConversionStatus.PARTIAL_SUCCESS
        with self.assertLogs(convert.logger, logging.WARNING):
            conversion = self.convert([DIGITAL, SCANNED])

        self.assertEqual(self.calls[-1], (True, None))
        self.assertEqual(conversion.routing, "document")
        self.assertEqual(conversion.page_ocr, {1: True, 2: True})

    def test_page_number_mismatch_falls_back_to_whole_document_ocr(self):
        self.total_pages = 2
        self.merged.pages = {1: object()}  # a page went missing in the merge
        with self.assertLogs(convert.logger, logging.WARNING):
            conversion = convert.convert_pdf(
                self.make_pdf([DIGITAL, SCANNED]), routing="page"
            )
        self.assertEqual(self.calls[-1], (True, None))
        self.assertEqual(conversion.routing, "document")

    def test_mostly_ocr_pdf_falls_back_to_document_routing(self):
        with patch.object(convert, "MAX_OCR_FRACTION", 0.25):
            conversion = self.convert([DIGITAL, SCANNED, SCANNED, DIGITAL])

        self.assertEqual(conversion.routing, "document")
        self.assertEqual(self.calls, [(True, None)])
        self.assertEqual(set(conversion.page_ocr.values()), {True})

    def test_mostly_digital_pdf_keeps_page_routing(self):
        with patch.object(convert, "MAX_OCR_FRACTION", 0.25):
            conversion = self.convert([DIGITAL] * 7 + [SCANNED])

        self.assertEqual(conversion.routing, "page")

    def test_routing_defaults_to_page_and_reads_environment(self):
        self.total_pages = 2
        pdf = self.make_pdf([DIGITAL, SCANNED])
        self.merged.pages = {1: object(), 2: object()}

        with patch.dict("os.environ", clear=False) as env:
            env.pop("PDF_ROUTING", None)
            self.assertEqual(convert.convert_pdf(pdf).routing, "page")

        with patch.dict("os.environ", {"PDF_ROUTING": "document"}):
            self.assertEqual(convert.convert_pdf(pdf).routing, "document")


class BridgeShortGapsTests(unittest.TestCase):
    @staticmethod
    def pages(pattern):
        return {n: ch == "o" for n, ch in enumerate(pattern, 1)}

    def bridge(self, pattern, max_gap):
        result = convert._bridge_short_gaps(self.pages(pattern), max_gap)
        return "".join("o" if result[n] else "d" for n in sorted(result))

    def test_short_gap_between_ocr_runs_is_filled(self):
        self.assertEqual(self.bridge("oddo", 2), "oooo")

    def test_gap_longer_than_the_limit_is_kept(self):
        self.assertEqual(self.bridge("odddo", 2), "odddo")

    def test_leading_and_trailing_digital_pages_are_not_ocrd(self):
        self.assertEqual(self.bridge("ddodd", 5), "ddodd")

    def test_all_digital_and_all_ocr_are_unchanged(self):
        self.assertEqual(self.bridge("dddd", 5), "dddd")
        self.assertEqual(self.bridge("oooo", 5), "oooo")

    def test_several_gaps_merge_into_one_run(self):
        bridged = self.bridge("ododdo", 2)
        self.assertEqual(bridged, "oooooo")
        self.assertEqual(len(convert._ocr_runs(self.pages(bridged))), 1)


class ExtractedCharsTests(unittest.TestCase):
    @staticmethod
    def prov(page_no):
        return [SimpleNamespace(page_no=page_no)]

    def document(self):
        text = lambda t, page, parent=None: SimpleNamespace(
            text=t, prov=self.prov(page), parent=parent and SimpleNamespace(cref=parent)
        )
        cell = lambda t: SimpleNamespace(text=t)
        return SimpleNamespace(
            texts=[
                text("a" * 60, 1),
                text("  b  ", 2),
                text("c" * 10, 2),
                # Chart labels nested under a picture are never chunked.
                text("d" * 200, 2, parent="#/pictures/0"),
                text("e" * 20, 4, parent="#/groups/0"),
            ],
            tables=[
                SimpleNamespace(
                    prov=self.prov(3),
                    data=SimpleNamespace(table_cells=[cell("x" * 40), cell("y" * 30)]),
                )
            ],
        )

    def test_counts_indexable_text_and_table_cells_per_page(self):
        self.assertEqual(
            convert.extracted_chars_by_page(self.document()), {1: 60, 2: 11, 3: 70, 4: 20}
        )

    def test_page_records_flag_low_yield_pages(self):
        pages = [
            SimpleNamespace(index=i, page_type="digital", char_count=1, image_count=0,
                            large_image_count=0, reason="r")
            for i in range(4)
        ]
        conversion = convert.PdfConversion(
            self.document(), "page", {n: False for n in range(1, 5)},
            SimpleNamespace(pages=pages),
        )

        records = conversion.page_records()

        self.assertEqual([r["extracted_chars"] for r in records], [60, 11, 70, 20])
        self.assertEqual([r["low_yield"] for r in records], [0, 1, 0, 1])


@unittest.skipUnless(DOCLING_INSTALLED, "docling is not installed")
class RealDoclingTests(unittest.TestCase):
    def test_options_are_accepted_by_installed_docling(self):
        # Constructed against the real class, so a renamed option would fail here.
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        for do_ocr in (True, False):
            options = convert.build_pdf_pipeline_options(do_ocr)
            self.assertIsInstance(options, PdfPipelineOptions)
            self.assertIs(options.do_ocr, do_ocr)
            self.assertTrue(options.do_table_structure)
            self.assertFalse(options.generate_page_images)
            self.assertFalse(options.generate_picture_images)

    def test_merge_of_page_runs_restores_page_numbers_and_origin(self):
        from docling_core.types.doc import BoundingBox, DocItemLabel, Size
        from docling_core.types.doc.document import (
            DoclingDocument,
            DocumentOrigin,
            ProvenanceItem,
        )

        def run(first, last):
            doc = DoclingDocument(
                name="report",
                origin=DocumentOrigin(
                    mimetype="application/pdf", binary_hash=1, filename="report.pdf"
                ),
            )
            for n in range(first, last + 1):
                doc.add_page(page_no=n, size=Size(width=612, height=792))
                doc.add_text(
                    label=DocItemLabel.PARAGRAPH,
                    text=f"page {n}",
                    prov=ProvenanceItem(
                        page_no=n,
                        bbox=BoundingBox(l=0, t=10, r=100, b=0),
                        charspan=(0, 6),
                    ),
                )
            return SimpleNamespace(status=convert.ConversionStatus.SUCCESS, document=doc)

        results = {(1, 1): run(1, 1), (2, 3): run(2, 3), (4, 4): run(4, 4)}
        converter = SimpleNamespace(
            convert=lambda path, page_range=None: results[page_range]
        )
        page_ocr = {1: False, 2: True, 3: True, 4: False}

        with patch.object(convert, "_get_converter", return_value=converter):
            merged = convert._convert_by_page("ignored.pdf", page_ocr)

        self.assertEqual(sorted(merged.pages), [1, 2, 3, 4])
        self.assertEqual(
            [(t.text, t.prov[0].page_no) for t in merged.texts],
            [(f"page {n}", n) for n in range(1, 5)],
        )
        self.assertEqual(merged.origin.filename, "report.pdf")


if __name__ == "__main__":
    unittest.main()
