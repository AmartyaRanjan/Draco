import importlib
import unittest

try:
    chunk = importlib.import_module("app.ingestion.chunk")
except ImportError:  # Docling is not installed
    chunk = None


def raw(text, start=1, end=None, section=None):
    return chunk._RawChunk(text, start, end if end is not None else start, section)


def words(text):
    return len(text.split())


@unittest.skipIf(chunk is None, "Docling is not installed")
class MergeSmallChunksTests(unittest.TestCase):
    BIG = "word " * 20  # well above the minimum size
    PLACEHOLDER = "<!-- formula-not-decoded -->"

    def merge(self, chunks, max_tokens=1000):
        return list(chunk.merge_small_chunks(iter(chunks), words, max_tokens))

    def test_small_chunk_joins_the_previous_chunk(self):
        result = self.merge([raw(self.BIG, 1), raw("Side by side boxplot\nX", 2), raw(self.BIG, 3)])

        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].text.endswith("Side by side boxplot\nX"))
        self.assertEqual((result[0].page_start, result[0].page_end), (1, 2))
        self.assertEqual(result[1].text, self.BIG)

    def test_placeholders_are_dropped_and_empty_remainder_adds_nothing(self):
        result = self.merge([raw(self.BIG, 1), raw(f"Example\n{self.PLACEHOLDER}", 2)])

        self.assertEqual(len(result), 1)
        self.assertNotIn("formula-not-decoded", result[0].text)
        self.assertTrue(result[0].text.endswith("Example"))

        result = self.merge([raw(self.BIG, 1), raw(self.PLACEHOLDER, 2)])
        self.assertEqual(result[0].text, self.BIG)
        self.assertEqual(result[0].page_end, 2)

    def test_leading_small_chunk_joins_the_next_chunk(self):
        result = self.merge([raw("Recap", 1, section="Intro"), raw(self.BIG, 2, section="Body")])

        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].text.startswith("Recap"))
        self.assertEqual((result[0].page_start, result[0].page_end), (1, 2))
        self.assertEqual(result[0].section_path, "Body")

    def test_consecutive_small_chunks_all_fold_in(self):
        result = self.merge([raw(self.BIG, 1), raw("a", 2), raw("b", 3), raw("c", 4)])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, f"{self.BIG}\na\nb\nc")

    def test_merge_that_would_exceed_the_token_limit_is_skipped(self):
        full = "word " * 20
        result = self.merge([raw(full, 1), raw("tiny", 2), raw(full, 3)], max_tokens=20)

        self.assertEqual([c.text for c in result], [full, "tiny", full])

    def test_normal_chunks_pass_through_unchanged(self):
        chunks = [raw(self.BIG, 1), raw(self.BIG, 2)]
        self.assertEqual([c.text for c in self.merge(chunks)], [self.BIG, self.BIG])

    def test_single_small_chunk_is_kept(self):
        self.assertEqual([c.text for c in self.merge([raw("only")])], ["only"])

    def test_empty_input(self):
        self.assertEqual(self.merge([]), [])


if __name__ == "__main__":
    unittest.main()
