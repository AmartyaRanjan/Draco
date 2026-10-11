import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling_core.types.doc import DoclingDocument

from app.ingestion.pdf_detector import PdfDetectionError, PdfInspection, inspect_pdf

logger = logging.getLogger(__name__)

# "document": one conversion; OCR is skipped only if every page is digital.
# "page": digital pages skip OCR even in mixed PDFs (see _convert_by_page).
ROUTING_MODES = ("document", "page")


@dataclass(frozen=True)
class PdfConversion:
    document: DoclingDocument
    routing: str  # routing actually used: "document" or "page"
    # OCR decision per page, keyed by 1-based page number. Empty if the PDF could
    # not be inspected (the whole document was converted with OCR).
    page_ocr: dict[int, bool]
    inspection: PdfInspection | None  # None if detection failed

    def page_records(self) -> list[dict]:
        """Page-level data for storage; empty if the PDF could not be inspected."""
        if self.inspection is None:
            return []
        return [
            {
                "page_no": p.index + 1,
                "page_type": p.page_type,
                "char_count": p.char_count,
                "image_count": p.image_count,
                "large_image_count": p.large_image_count,
                "ocr_used": int(self.page_ocr[p.index + 1]),
                "reason": p.reason,
            }
            for p in self.inspection.pages
        ]


def build_pdf_pipeline_options(do_ocr: bool) -> PdfPipelineOptions:
    # do_ocr only toggles the OCR stage; layout analysis and table structure
    # still run in Docling's standard pipeline.
    return PdfPipelineOptions(
        do_ocr=do_ocr,
        do_table_structure=True,
        generate_page_images=False,
    )


@lru_cache(maxsize=2)
def _get_converter(do_ocr: bool) -> DocumentConverter:
    # Docling caches its loaded models on the converter instance, so reuse one
    # per OCR mode instead of reloading models for every document.
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=build_pdf_pipeline_options(do_ocr)
            )
        }
    )


def _convert(pdf_path: str | Path, do_ocr: bool, page_range: tuple[int, int] | None = None):
    kwargs = {"page_range": page_range} if page_range else {}
    try:
        return _get_converter(do_ocr).convert(str(pdf_path), **kwargs)
    except Exception:
        logger.exception(
            "Docling conversion failed (OCR %s, pages %s)",
            "enabled" if do_ocr else "disabled",
            page_range or "all",
        )
        raise


def _ocr_runs(page_ocr: dict[int, bool]) -> list[tuple[bool, int, int]]:
    """Group consecutive pages with the same OCR decision into (do_ocr, first, last)."""
    runs = []
    for do_ocr, group in groupby(sorted(page_ocr.items()), key=lambda item: item[1]):
        pages = [page_no for page_no, _ in group]
        runs.append((do_ocr, pages[0], pages[-1]))
    return runs


def _convert_by_page(pdf_path: str | Path, page_ocr: dict[int, bool]) -> DoclingDocument:
    """Convert each run of same-decision pages separately and merge the results.

    Raises ValueError if the merged document does not cover exactly the expected
    pages, so the caller can fall back to a whole-document conversion.
    """
    runs = _ocr_runs(page_ocr)
    docs = []
    for do_ocr, first, last in runs:
        result = _convert(pdf_path, do_ocr, page_range=(first, last))
        if result.status != ConversionStatus.SUCCESS:
            raise ValueError(f"pages {first}-{last} converted with status {result.status}")
        if len(result.document.pages) != last - first + 1:
            raise ValueError(f"pages {first}-{last} produced {len(result.document.pages)} pages")
        docs.append(result.document)

    # concatenate() rebases each document's pages to follow the previous one, so
    # contiguous in-order runs that cover every page come out numbered 1..N. It
    # also drops `origin`, which downstream code may use for the source filename.
    merged = DoclingDocument.concatenate(docs)
    merged.origin = docs[0].origin
    merged.name = docs[0].name

    if sorted(merged.pages) != sorted(page_ocr):
        raise ValueError("merged page numbers do not match the source PDF")
    return merged


def convert_pdf(pdf_path: str | Path, routing: str | None = None) -> PdfConversion:
    """Convert a PDF with Docling's standard pipeline and report per-page OCR decisions.

    Any page that is not clearly digital is converted with OCR. If the PDF cannot
    be inspected, the whole document is converted with OCR.
    """
    routing = routing or os.getenv("PDF_ROUTING", "document")
    if routing not in ROUTING_MODES:
        raise ValueError(f"routing must be one of {ROUTING_MODES}, got {routing!r}")

    try:
        inspection = inspect_pdf(pdf_path)
    except PdfDetectionError as exc:
        logger.warning("PDF detection failed (%s); falling back to OCR", exc)
        result = _convert(pdf_path, True)
        return PdfConversion(result.document, "document", {}, None)

    page_ocr = {p.index + 1: p.page_type != "digital" for p in inspection.pages}

    if routing == "page" and len(set(page_ocr.values())) > 1:
        logger.info(
            "Converting by page: %d page(s) with OCR, %d without",
            sum(page_ocr.values()),
            len(page_ocr) - sum(page_ocr.values()),
        )
        try:
            return PdfConversion(_convert_by_page(pdf_path, page_ocr), "page", page_ocr, inspection)
        except Exception as exc:
            logger.warning("Page-level conversion failed (%s); retrying whole document with OCR", exc)
            result = _convert(pdf_path, True)
            return PdfConversion(result.document, "document", dict.fromkeys(page_ocr, True), inspection)

    do_ocr = any(page_ocr.values())
    logger.info("Converting whole document with OCR %s", "enabled" if do_ocr else "disabled")
    result = _convert(pdf_path, do_ocr)
    return PdfConversion(result.document, "document", dict.fromkeys(page_ocr, do_ocr), inspection)


def convert_document(pdf_path: str | Path):
    """Return Docling's structured document for the PDF."""
    return convert_pdf(pdf_path).document
