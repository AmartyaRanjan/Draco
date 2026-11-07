"""Supported upload formats and their pre-conversion limits (no Docling import here).

PDF keeps its own structural preflight (app.ingestion.pdf_detector: bytes, pages, images per
page). The other formats are checked here before any parser runs:

  .md/.markdown/.txt  INGEST_MAX_TEXT_BYTES (default 10 MiB); must decode as UTF-8 (BOM allowed)
  .html/.htm          INGEST_MAX_HTML_BYTES (default 20 MiB)
  .docx               INGEST_MAX_DOCX_BYTES (default 50 MiB) on disk, and, because a .docx is a
                      ZIP archive, INGEST_MAX_DOCX_UNCOMPRESSED_BYTES (default 200 MiB) total
                      uncompressed size, INGEST_MAX_DOCX_ENTRIES (default 5000) members and a
                      compression ratio above INGEST_MAX_DOCX_RATIO (default 100) is refused
                      (zip-bomb guard). Only the ZIP central directory is read.

A limit violation raises IngestionPolicyError (a ValueError): the document fails, it is never
converted "anyway" by a fallback path.
"""
import os
import zipfile
from pathlib import Path

PDF_SUFFIXES = frozenset({".pdf"})
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
TEXT_SUFFIXES = frozenset({".txt"})
HTML_SUFFIXES = frozenset({".html", ".htm"})
DOCX_SUFFIXES = frozenset({".docx"})
SUPPORTED_SUFFIXES = PDF_SUFFIXES | MARKDOWN_SUFFIXES | TEXT_SUFFIXES | HTML_SUFFIXES | DOCX_SUFFIXES

MAX_TEXT_BYTES = int(os.getenv("INGEST_MAX_TEXT_BYTES", str(10 * 1024 * 1024)))
MAX_HTML_BYTES = int(os.getenv("INGEST_MAX_HTML_BYTES", str(20 * 1024 * 1024)))
MAX_DOCX_BYTES = int(os.getenv("INGEST_MAX_DOCX_BYTES", str(50 * 1024 * 1024)))
MAX_DOCX_UNCOMPRESSED_BYTES = int(os.getenv("INGEST_MAX_DOCX_UNCOMPRESSED_BYTES", str(200 * 1024 * 1024)))
MAX_DOCX_ENTRIES = int(os.getenv("INGEST_MAX_DOCX_ENTRIES", "5000"))
MAX_DOCX_RATIO = float(os.getenv("INGEST_MAX_DOCX_RATIO", "100"))


class IngestionPolicyError(ValueError):
    """The file exceeds a configured limit or is not what its extension claims."""


def detect_format(path: str | Path) -> str:
    """'pdf' | 'markdown' | 'text' | 'html' | 'docx'; NotImplementedError for anything else."""
    suffix = Path(path).suffix.lower()
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in HTML_SUFFIXES:
        return "html"
    if suffix in DOCX_SUFFIXES:
        return "docx"
    raise NotImplementedError(
        f"Unsupported file format: {suffix or '(none)'}. "
        f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
    )


def _check_size(path: Path, limit: int, label: str) -> int:
    size = path.stat().st_size
    if size == 0:
        raise IngestionPolicyError(f"{label} file is empty")
    if size > limit:
        raise IngestionPolicyError(f"{label} file is {size} bytes; the limit is {limit}")
    return size


def read_text_file(path: str | Path, limit: int | None = None) -> str:
    """Size-checked UTF-8 read for .md/.txt (a UTF-8 BOM is accepted and dropped)."""
    path = Path(path)
    _check_size(path, limit or MAX_TEXT_BYTES, "Text")
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise IngestionPolicyError("Text file contains NUL bytes (binary content?)")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestionPolicyError("Text file is not valid UTF-8") from exc
    if not text.strip():
        raise IngestionPolicyError("Text file has no content")
    return text


def check_html(path: str | Path, limit: int | None = None) -> None:
    _check_size(Path(path), limit or MAX_HTML_BYTES, "HTML")


def check_docx(path: str | Path, limit: int | None = None, max_uncompressed: int | None = None,
               max_entries: int | None = None, max_ratio: float | None = None) -> None:
    path = Path(path)
    size = _check_size(path, limit or MAX_DOCX_BYTES, "DOCX")
    max_uncompressed = max_uncompressed or MAX_DOCX_UNCOMPRESSED_BYTES
    max_entries = max_entries or MAX_DOCX_ENTRIES
    max_ratio = max_ratio or MAX_DOCX_RATIO
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            names = {i.filename for i in infos}
    except zipfile.BadZipFile as exc:
        raise IngestionPolicyError("DOCX file is not a valid ZIP/OOXML package") from exc
    if "word/document.xml" not in names or "[Content_Types].xml" not in names:
        raise IngestionPolicyError("DOCX package has no word/document.xml")
    if len(infos) > max_entries:
        raise IngestionPolicyError(f"DOCX has {len(infos)} archive members; the limit is {max_entries}")
    total = sum(i.file_size for i in infos)
    if total > max_uncompressed:
        raise IngestionPolicyError(f"DOCX expands to {total} bytes; the limit is {max_uncompressed}")
    if total / max(1, size) > max_ratio:
        raise IngestionPolicyError(f"DOCX compression ratio {total / size:.0f} exceeds {max_ratio:.0f}")
