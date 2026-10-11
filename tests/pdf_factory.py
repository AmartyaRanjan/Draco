"""Builds tiny PDFs for tests using pypdf's low-level API (no extra dependencies)."""

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

SMALL_IMAGE = (32, 32)  # e.g. a logo
LARGE_IMAGE = (800, 1000)  # e.g. a full-page scan

LONG_TEXT = "The quick brown fox jumps over the lazy dog. " * 6


def add_page(writer: PdfWriter, text: str | None = None, image: tuple[int, int] | None = None):
    page = writer.add_blank_page(612, 792)
    resources = DictionaryObject()
    ops = []

    if text:
        font = writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
        )
        resources[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font})
        ops.append(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET")

    if image:
        width, height = image
        stream = DecodedStreamObject()
        stream.set_data(bytes(width * height))  # blank grayscale pixels
        stream.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Image"),
                NameObject("/Width"): NumberObject(width),
                NameObject("/Height"): NumberObject(height),
                NameObject("/ColorSpace"): NameObject("/DeviceGray"),
                NameObject("/BitsPerComponent"): NumberObject(8),
            }
        )
        resources[NameObject("/XObject")] = DictionaryObject(
            {NameObject("/Im1"): writer._add_object(stream)}
        )
        ops.append("q 612 0 0 792 0 0 cm /Im1 Do Q")

    content = DecodedStreamObject()
    content.set_data("\n".join(ops).encode())
    page[NameObject("/Contents")] = writer._add_object(content)
    page[NameObject("/Resources")] = resources


def write_pdf(path: Path, pages: list[dict], password: str | None = None) -> Path:
    """pages: list of add_page kwargs, e.g. [{"text": "..."}, {"image": LARGE_IMAGE}]."""
    writer = PdfWriter()
    for kwargs in pages:
        add_page(writer, **kwargs)
    if password is not None:
        writer.encrypt(user_password=password)
    with open(path, "wb") as f:
        writer.write(f)
    return path
