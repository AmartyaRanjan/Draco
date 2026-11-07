"""Contextual chunk headers: header + text is embedded and BM25-indexed; the displayed text is unchanged."""
import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.ingestion import headers
from app.ingestion.schema import ChunkRecord


def _chunk(i, section="Results > Revenue", text=None):
    return ChunkRecord(chunk_id=f"c{i}", document_id="doc", document_version="v", chunk_index=i,
                       text=text or f"CGPA: 8.{i}", section_path=section)


class HeaderBuilderTests(unittest.TestCase):
    def test_header_names_document_and_section(self):
        self.assertEqual(headers.build_context_header("Annual report", "Results > Revenue"),
                         "Document: Annual report\nSection: Results > Revenue")
        self.assertEqual(headers.build_context_header("T", None), "Document: T")
        self.assertIsNone(headers.build_context_header(None, " "))

    def test_header_is_bounded(self):
        h = headers.build_context_header("t" * 1000, "s " * 1000, max_chars=200)
        self.assertLessEqual(len(h), 230)

    def test_title_from_filename(self):
        self.assertEqual(headers.title_from_filename("Jane_Doe__CV.pdf"), "Jane Doe CV")
        self.assertIsNone(headers.title_from_filename(None))

    def test_index_text_vs_display_text(self):
        c = _chunk(1)
        self.assertEqual(c.index_text(), "CGPA: 8.1")
        c.context_header = "Document: Jane Doe CV\nSection: Education"
        self.assertEqual(c.index_text(), "Document: Jane Doe CV\nSection: Education\n\nCGPA: 8.1")
        self.assertEqual(c.text, "CGPA: 8.1")
        self.assertEqual(c.to_payload()["context_header"], c.context_header)

    def test_setting_default_on_and_switchable(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("INGESTION_CONTEXT_HEADERS", None)
            self.assertTrue(headers.headers_enabled())
        with patch.dict("os.environ", {"INGESTION_CONTEXT_HEADERS": "false"}):
            self.assertFalse(headers.headers_enabled())
            out = list(headers.apply_context_headers([_chunk(1)], "T"))
            self.assertIsNone(out[0].context_header)
            self.assertEqual(headers.chunk_token_budget(512), 512)

    def test_chunk_budget_reserves_room_for_the_header(self):
        self.assertEqual(headers.chunk_token_budget(512, enabled=True), 512 - headers.HEADER_RESERVE_TOKENS)
        self.assertEqual(headers.chunk_token_budget(512, enabled=False), 512)


class PipelineHeaderTests(unittest.TestCase):
    def setUp(self):
        self.m = MagicMock()
        self.chunks = [_chunk(i) for i in range(3)]
        self.m.iter_chunks.side_effect = lambda **_: iter(self.chunks)
        self.embedded = []

        def embed(batch):
            self.embedded.extend(c.index_text() for c in batch)
            return [[0.0]] * len(batch)

        fakes = {
            "app.ingestion.chunk": SimpleNamespace(iter_chunks=self.m.iter_chunks),
            "app.ingestion.embed": SimpleNamespace(embed_chunks=embed),
            "app.db.qdrant": SimpleNamespace(upsert_chunks=self.m.upsert, delete_stale_versions=self.m.dv),
            "app.db.opensearch": SimpleNamespace(index_chunks=self.m.index, delete_stale_versions=self.m.dt),
        }
        p = patch.dict(sys.modules, fakes)
        p.start()
        self.addCleanup(p.stop)
        sys.modules.pop("app.ingestion.pipeline", None)
        self.addCleanup(sys.modules.pop, "app.ingestion.pipeline", None)
        self.pipeline = importlib.import_module("app.ingestion.pipeline")

    def test_headers_reach_embedding_and_bm25_but_not_displayed_text(self):
        with patch.dict("os.environ", {"INGESTION_CONTEXT_HEADERS": "true"}):
            self.pipeline.ingest_converted_document("doc", object(), "v", title="Jane Doe CV")
        header = "Document: Jane Doe CV\nSection: Results > Revenue"
        self.assertTrue(all(t.startswith(header + "\n\n") for t in self.embedded))
        qdrant_records = self.m.upsert.call_args.args[0]
        os_records = self.m.index.call_args.args[0]
        self.assertEqual(qdrant_records[0]["text"], "CGPA: 8.0")
        self.assertNotIn("search_text", qdrant_records[0])
        self.assertEqual(os_records[0]["text"], "CGPA: 8.0")
        self.assertEqual(os_records[0]["search_text"], f"{header}\n\nCGPA: 8.0")
        self.assertEqual(os_records[0]["context_header"], header)

    def test_headers_off_indexes_plain_text(self):
        with patch.dict("os.environ", {"INGESTION_CONTEXT_HEADERS": "false"}):
            self.pipeline.ingest_converted_document("doc", object(), "v", title="Jane Doe CV")
        self.assertEqual(self.embedded, ["CGPA: 8.0", "CGPA: 8.1", "CGPA: 8.2"])
        self.assertNotIn("search_text", self.m.index.call_args.args[0][0])

    def test_text_document_path_uses_the_same_indexing(self):
        with patch.dict("os.environ", {"INGESTION_CONTEXT_HEADERS": "true"}), \
                patch("app.ingestion.text_chunk.default_token_counter", return_value=lambda t: len(t.split())):
            out = self.pipeline.ingest_text_document("doc", "# Edu\n\nCGPA: 9.1 out of 10, first class.", "markdown",
                                                     "v", title="Jane Doe CV")
        self.assertEqual(out["chunk_count"], 1)
        rec = self.m.index.call_args.args[0][0]
        self.assertEqual(rec["section_path"], "Edu")
        self.assertTrue(rec["search_text"].startswith("Document: Jane Doe CV\nSection: Edu\n\n"))
        self.m.dv.assert_called_once_with("doc", "v")


class StoreAndQueryFieldTests(unittest.TestCase):
    def test_embed_chunks_embeds_header_plus_text(self):
        from app.ingestion import embed
        c = _chunk(1)
        c.context_header = "Document: X"
        with patch.object(embed, "embed_texts", return_value=[[0.0]]) as et:
            embed.embed_chunks([c])
        et.assert_called_once_with(["Document: X\n\nCGPA: 8.1"])

    def test_lexical_retriever_searches_text_and_search_text_but_returns_display_fields(self):
        from app.retrieval import retrievers
        client = MagicMock()
        client.search.return_value = {"hits": {"hits": []}}
        with patch.object(retrievers.os_db, "get_opensearch_client", return_value=client):
            retrievers.LexicalRetriever(5, 1)._search("q", ["d1"])
        body = client.search.call_args.kwargs["body"]
        self.assertEqual(body["query"]["bool"]["must"][0]["multi_match"]["fields"],
                         ["text", "search_text", "section_path^0.5"])
        self.assertNotIn("search_text", body["_source"])
        self.assertEqual(body["query"]["bool"]["filter"], [{"terms": {"document_id": ["d1"]}}])

    def test_opensearch_mapping_declares_context_fields(self):
        from app.db import opensearch
        client = MagicMock()
        client.indices.exists.return_value = False
        with patch.object(opensearch, "get_opensearch_client", return_value=client):
            opensearch.ensure_index()
        props = client.indices.create.call_args.kwargs["body"]["mappings"]["properties"]
        self.assertEqual(props["search_text"], {"type": "text"})

    def test_existing_index_gets_context_fields_added_once(self):
        from app.db import opensearch
        client = MagicMock()
        client.indices.exists.return_value = True
        with patch.object(opensearch, "get_opensearch_client", return_value=client), \
                patch.object(opensearch, "_context_fields_ready", False):
            opensearch.ensure_index()
            opensearch.ensure_index()
        client.indices.put_mapping.assert_called_once()
        client.indices.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
