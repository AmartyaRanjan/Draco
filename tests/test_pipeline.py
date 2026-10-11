import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.ingestion.schema import ChunkRecord


def _chunk(index):
    return ChunkRecord(
        chunk_id=f"c{index}",
        document_id="doc",
        document_version="2",
        chunk_index=index,
        text=f"text {index}",
    )


class IngestConvertedDocumentTests(unittest.TestCase):
    def setUp(self):
        # Replace the heavy modules (Docling, sentence-transformers, Qdrant and
        # OpenSearch clients) before the pipeline is imported.
        self.manager = MagicMock()
        self.chunks = [_chunk(i) for i in range(5)]
        self.manager.iter_chunks.side_effect = lambda **_: iter(self.chunks)
        self.manager.embed_chunks.side_effect = lambda batch: [[0.0]] * len(batch)

        fakes = {
            "app.ingestion.chunk": SimpleNamespace(iter_chunks=self.manager.iter_chunks),
            "app.ingestion.embed": SimpleNamespace(embed_chunks=self.manager.embed_chunks),
            "app.db.qdrant": SimpleNamespace(
                upsert_chunks=self.manager.upsert_chunks,
                delete_stale_versions=self.manager.delete_stale_vectors,
            ),
            "app.db.opensearch": SimpleNamespace(
                index_chunks=self.manager.index_chunks,
                delete_stale_versions=self.manager.delete_stale_text,
            ),
        }
        patcher = patch.dict(sys.modules, fakes)
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("app.ingestion.pipeline", None)
        self.addCleanup(sys.modules.pop, "app.ingestion.pipeline", None)

        # import_module, not `from app.ingestion import pipeline`, which would
        # return a stale attribute left on the package by an earlier import.
        pipeline = importlib.import_module("app.ingestion.pipeline")
        self.pipeline = pipeline
        patcher = patch.object(pipeline, "CHUNK_BATCH_SIZE", 2)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _names(self):
        wanted = ("upsert_chunks", "index_chunks", "delete_stale_vectors", "delete_stale_text")
        return [c[0] for c in self.manager.mock_calls if c[0] in wanted]

    def test_chunks_are_written_in_bounded_batches(self):
        result = self.pipeline.ingest_converted_document("doc", object(), "2")

        self.assertEqual(result["chunk_count"], 5)
        self.assertEqual(
            [len(c.args[0]) for c in self.manager.upsert_chunks.call_args_list], [2, 2, 1]
        )

    def test_superseded_versions_are_deleted_after_all_batches(self):
        self.pipeline.ingest_converted_document("doc", object(), "2")

        self.manager.delete_stale_vectors.assert_called_once_with("doc", "2")
        self.manager.delete_stale_text.assert_called_once_with("doc", "2")
        self.assertEqual(
            self._names(),
            ["upsert_chunks", "index_chunks"] * 3
            + ["delete_stale_vectors", "delete_stale_text"],
        )

    def test_failed_write_keeps_the_previous_version(self):
        self.manager.index_chunks.side_effect = [None, RuntimeError("opensearch down")]

        with self.assertRaises(RuntimeError):
            self.pipeline.ingest_converted_document("doc", object(), "2")

        self.manager.delete_stale_vectors.assert_not_called()
        self.manager.delete_stale_text.assert_not_called()

    def test_empty_document_is_an_error_and_deletes_nothing(self):
        self.chunks.clear()

        with self.assertRaises(ValueError):
            self.pipeline.ingest_converted_document("doc", object(), "2")

        self.manager.delete_stale_vectors.assert_not_called()

    def test_embedding_count_mismatch_is_an_error(self):
        self.manager.embed_chunks.side_effect = lambda batch: [[0.0]]

        with self.assertRaises(RuntimeError):
            self.pipeline.ingest_converted_document("doc", object(), "2")

        self.manager.upsert_chunks.assert_not_called()


class ChunkerCacheTests(unittest.TestCase):
    def test_chunker_is_built_once(self):
        try:
            chunk = importlib.import_module("app.ingestion.chunk")
        except ImportError:
            self.skipTest("Docling is not installed")

        chunk.build_chunker.cache_clear()
        self.addCleanup(chunk.build_chunker.cache_clear)
        with patch.object(chunk, "HuggingFaceTokenizer") as tokenizer, patch.object(
            chunk, "HybridChunker"
        ) as hybrid:
            first = chunk.build_chunker()
            second = chunk.build_chunker()

        self.assertIs(first, second)
        tokenizer.from_pretrained.assert_called_once()
        hybrid.assert_called_once()


if __name__ == "__main__":
    unittest.main()
