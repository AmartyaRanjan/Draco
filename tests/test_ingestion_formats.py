"""Multi-format ingestion: format detection, pre-conversion limits, and the Docling-free Markdown/text chunker."""
import tempfile
import unittest
import zipfile
from pathlib import Path

from app.ingestion import formats
from app.ingestion.text_chunk import iter_text_chunks, markdown_blocks


def words(text):
    return len(text.split())


class FormatDetectionTests(unittest.TestCase):
    def test_supported_suffixes(self):
        cases = {"a.pdf": "pdf", "a.PDF": "pdf", "a.md": "markdown", "a.markdown": "markdown", "a.txt": "text",
                 "a.html": "html", "a.HTM": "html", "a.docx": "docx"}
        for name, fmt in cases.items():
            self.assertEqual(formats.detect_format(name), fmt, name)

    def test_unsupported_suffix_raises_not_implemented(self):
        for name in ("a.doc", "a.xlsx", "a", "a.exe"):
            with self.assertRaises(NotImplementedError):
                formats.detect_format(name)


class LimitTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _docx(self, name="d.docx", body=b"<w:document/>", extra=None, members=None):
        path = self.dir / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for member in members if members is not None else ("[Content_Types].xml", "word/document.xml"):
                z.writestr(member, body if member == "word/document.xml" else b"<Types/>")
            for k, v in (extra or {}).items():
                z.writestr(k, v)
        return path

    def test_text_file_utf8_bom_and_size(self):
        p = self.dir / "a.md"
        p.write_bytes("﻿# Título\n\nhé".encode())
        self.assertEqual(formats.read_text_file(p), "# Título\n\nhé")
        with self.assertRaises(formats.IngestionPolicyError):
            formats.read_text_file(p, limit=3)

    def test_text_file_rejects_binary_non_utf8_and_empty(self):
        for raw in (b"a\x00b", b"\xff\xfe\xfa", b"   \n"):
            p = self.dir / "x.txt"
            p.write_bytes(raw)
            with self.assertRaises(formats.IngestionPolicyError):
                formats.read_text_file(p)
        (self.dir / "e.txt").write_bytes(b"")
        with self.assertRaises(formats.IngestionPolicyError):
            formats.read_text_file(self.dir / "e.txt")

    def test_html_size_limit(self):
        p = self.dir / "a.html"
        p.write_text("<p>hello</p>")
        formats.check_html(p)
        with self.assertRaises(formats.IngestionPolicyError):
            formats.check_html(p, limit=5)

    def test_docx_valid_package_passes(self):
        formats.check_docx(self._docx(body=b"<w:document>" + b"x" * 50 + b"</w:document>"))

    def test_docx_not_a_zip_or_missing_document_xml(self):
        bad = self.dir / "bad.docx"
        bad.write_bytes(b"plain")
        with self.assertRaises(formats.IngestionPolicyError):
            formats.check_docx(bad)
        with self.assertRaises(formats.IngestionPolicyError):
            formats.check_docx(self._docx(members=("[Content_Types].xml",)))

    def test_docx_zip_bomb_guards(self):
        bomb = self._docx(body=b"\x00" * 2_000_000)  # compresses ~1000:1
        with self.assertRaises(formats.IngestionPolicyError) as ctx:
            formats.check_docx(bomb)
        self.assertIn("ratio", str(ctx.exception))
        with self.assertRaises(formats.IngestionPolicyError):
            formats.check_docx(self._docx(body=b"x" * 5000), max_uncompressed=1000, max_ratio=1e9)
        many = self._docx(extra={f"word/media/{i}.bin": b"y" for i in range(20)})
        with self.assertRaises(formats.IngestionPolicyError):
            formats.check_docx(many, max_entries=10)
        with self.assertRaises(formats.IngestionPolicyError):
            formats.check_docx(self._docx(), limit=10)


class MarkdownChunkTests(unittest.TestCase):
    def chunks(self, text, fmt="markdown", max_tokens=60):
        return list(iter_text_chunks(text, "doc", "v1", fmt=fmt, count_tokens=words, max_tokens=max_tokens))

    def test_headings_build_section_paths_and_code_fences_are_not_headings(self):
        md = ("---\ntitle: x\n---\n# Guide\n\nIntro paragraph with enough words to stand on its own here.\n\n"
              "## Install\n\n```bash\n# not a heading\npip install x\n```\n\n"
              "Setext Title\n============\n\nBody under setext with enough words to stand alone too.\n")
        blocks = list(markdown_blocks(md))
        self.assertEqual(blocks[0].section, ("Guide",))
        self.assertEqual(blocks[1].section, ("Guide", "Install"))
        self.assertIn("# not a heading", blocks[1].text)
        self.assertEqual(blocks[-1].section, ("Setext Title",))
        self.assertNotIn("title: x", " ".join(b.text for b in blocks))

    def test_chunks_carry_section_path_and_heading_prefix_and_no_pages(self):
        md = ("# Results\n\n## Revenue\n\nRevenue grew by 12 percent to 4.1 million in the fiscal year 2025 overall.\n\n"
              "# Team\n\nThe team has nine engineers and two designers working across three time zones.\n")
        out = self.chunks(md)
        self.assertEqual([c.section_path for c in out], ["Results > Revenue", "Team"])
        self.assertTrue(out[0].text.startswith("Results\nRevenue\nRevenue grew"))
        self.assertEqual([c.chunk_index for c in out], [0, 1])
        self.assertTrue(all(c.page_start is None and c.page_end is None for c in out))
        self.assertIsNone(out[0].context_header)  # headers are applied by the pipeline

    def test_chunks_respect_the_token_budget_and_split_oversized_paragraphs(self):
        para = " ".join(f"Sentence number {i} has five words." for i in range(60))
        out = self.chunks(f"# H\n\n{para}\n", max_tokens=40)
        self.assertGreater(len(out), 5)
        self.assertTrue(all(words(c.text) <= 40 for c in out))
        self.assertIn("Sentence number 59", out[-1].text)

    def test_small_sections_do_not_become_tiny_chunks(self):
        md = "# A\n\nok\n\n# B\n\n" + "word " * 20
        out = self.chunks(md)
        self.assertEqual(len(out), 1)
        self.assertIn("ok", out[0].text)
        self.assertEqual(out[0].section_path, "B")

    def test_plain_text_has_no_sections_and_ids_are_stable(self):
        text = "First paragraph here.\n\n# not a heading in txt\n\nSecond."
        a, b = self.chunks(text, fmt="text"), self.chunks(text, fmt="text")
        self.assertEqual([c.chunk_id for c in a], [c.chunk_id for c in b])
        self.assertTrue(all(c.section_path is None for c in a))
        self.assertIn("# not a heading in txt", a[0].text)

    def test_unknown_text_format_rejected(self):
        with self.assertRaises(ValueError):
            self.chunks("x", fmt="rst")


if __name__ == "__main__":
    unittest.main()
