"""Long-term memory: per-user items in Qdrant (dense) + OpenSearch (BM25), recalled with RRF.

Stores are mocked the way the other tests mock them (no service is contacted)."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.db import memory_store
from app.db.memory_store import MemoryIndex, MemoryStoreConfigError, item_key
from app.inference.memory import HybridMemory, MemoryItem, NullMemory, build_memory_provider, rrf_fuse

from tests.test_inference import Harness, isettings


def _point(mid, user, text, score):
    return SimpleNamespace(payload={"memory_id": mid, "user_id": user, "text": text, "kind": "fact"}, score=score)


def _hit(mid, user, text, score):
    return {"_source": {"memory_id": mid, "user_id": user, "text": text, "kind": "fact"}, "_score": score}


def _index(dense_points=(), bm25_hits=()):
    q, o = MagicMock(name="qdrant"), MagicMock(name="opensearch")
    q.collection_exists.return_value = True
    q.query_points.return_value = SimpleNamespace(points=list(dense_points))
    q.count.return_value = SimpleNamespace(count=0)
    q.scroll.return_value = ([], None)
    o.indices.exists.return_value = True
    o.search.return_value = {"hits": {"hits": list(bm25_hits)}}
    return MemoryIndex(qdrant=q, opensearch=o, vector_size=3), q, o


class MemoryIndexTests(unittest.TestCase):
    def test_names_must_differ_from_document_stores(self):
        with self.assertRaises(MemoryStoreConfigError):
            MemoryIndex(qdrant=MagicMock(), opensearch=MagicMock(), collection=memory_store.qd_db.COLLECTION_NAME)
        with self.assertRaises(MemoryStoreConfigError):
            MemoryIndex(qdrant=MagicMock(), opensearch=MagicMock(), index=memory_store.os_db.INDEX_NAME)
        self.assertNotEqual(memory_store.MEMORY_COLLECTION, memory_store.qd_db.COLLECTION_NAME)
        self.assertNotEqual(memory_store.MEMORY_INDEX, memory_store.os_db.INDEX_NAME)

    def test_every_query_carries_the_owner_filter(self):
        idx, q, o = _index()
        idx.dense_search("alice", [0.1, 0.2, 0.3], 5)
        idx.lexical_search("alice", "hindi", 5)
        idx.list_items("alice")
        idx.count("alice")
        idx.delete("alice", "m1")
        idx.delete_user("alice")
        flt = q.query_points.call_args.kwargs["query_filter"]
        self.assertEqual([(c.key, c.match.value) for c in flt.must], [("user_id", "alice")])
        self.assertEqual(q.query_points.call_args.kwargs["collection_name"], idx.collection)
        self.assertEqual(q.scroll.call_args.kwargs["scroll_filter"].must[0].match.value, "alice")
        self.assertEqual(q.count.call_args.kwargs["count_filter"].must[0].match.value, "alice")
        body = o.search.call_args.kwargs["body"]
        self.assertEqual(body["query"]["bool"]["filter"], [{"term": {"user_id": "alice"}}])
        self.assertEqual(o.search.call_args.kwargs["index"], idx.index)
        del_filters = [c.kwargs["points_selector"].filter.must for c in q.delete.call_args_list]
        self.assertEqual([(c.key, c.match.value) for c in del_filters[0]], [("user_id", "alice"), ("memory_id", "m1")])
        self.assertEqual([(c.key, c.match.value) for c in del_filters[1]], [("user_id", "alice")])
        os_del = [c.kwargs["body"]["query"]["bool"]["filter"] for c in o.delete_by_query.call_args_list]
        self.assertEqual(os_del[0], [{"term": {"user_id": "alice"}}, {"term": {"memory_id": "m1"}}])

    def test_missing_owner_is_refused_before_any_store_call(self):
        idx, q, o = _index()
        for bad in (None, "", "  "):
            with self.assertRaises(PermissionError):
                idx.dense_search(bad, [0, 0, 0], 5)
            with self.assertRaises(PermissionError):
                idx.lexical_search(bad, "x", 5)
        q.query_points.assert_not_called()
        o.search.assert_not_called()

    def test_foreign_rows_returned_by_a_store_are_dropped(self):
        idx, _, _ = _index([_point("m1", "bob", "bob secret", .9)], [_hit("m2", "bob", "bob secret", 3.0)])
        self.assertEqual(idx.dense_search("alice", [0, 0, 0], 5), [])
        self.assertEqual(idx.lexical_search("alice", "secret", 5), [])

    def test_upsert_writes_both_stores_with_owner_bound_ids(self):
        idx, q, o = _index()
        idx.upsert("alice", "m1", "answer in Hindi", [0.1, 0.2, 0.3], kind="preference", created_at="t")
        point = q.upsert.call_args.kwargs["points"][0]
        self.assertEqual(point.id, item_key("alice", "m1"))
        self.assertNotEqual(item_key("alice", "m1"), item_key("bob", "m1"))
        self.assertEqual(point.payload["user_id"], "alice")
        self.assertEqual(o.index.call_args.kwargs["id"], item_key("alice", "m1"))
        self.assertEqual(o.index.call_args.kwargs["index"], idx.index)
        with self.assertRaises(ValueError):
            idx.upsert("alice", "m2", "x", [0.1], kind="fact", created_at="t")

    def test_ensure_creates_collection_index_and_owner_payload_index(self):
        idx, q, o = _index()
        q.collection_exists.return_value = False
        o.indices.exists.return_value = False
        idx.ensure()
        q.create_collection.assert_called_once()
        self.assertIn("user_id", [c.kwargs["field_name"] for c in q.create_payload_index.call_args_list])
        props = o.indices.create.call_args.kwargs["body"]["mappings"]["properties"]
        self.assertEqual(props["user_id"], {"type": "keyword"})


class HybridMemoryTests(unittest.TestCase):
    def setUp(self):
        self.embed = MagicMock(return_value=[0.1, 0.2, 0.3])

    def mem(self, dense=(), bm25=()):
        idx, q, o = _index(dense, bm25)
        return HybridMemory(idx, self.embed, timeout_s=2), idx, q, o

    def test_rrf_fusion_ranks_items_found_by_both_retrievers_first(self):
        dense = [_point("a", "alice", "likes tea", .9), _point("b", "alice", "answer in Hindi", .8)]
        bm25 = [_hit("b", "alice", "answer in Hindi", 5.0), _hit("c", "alice", "lives in Pune", 2.0)]
        m, *_ = self.mem(dense, bm25)
        out = asyncio.run(m.search("alice", "which language should answers use", 5))
        self.assertEqual([i.memory_id for i in out], ["b", "a", "c"])
        self.assertAlmostEqual(out[0].score, 1 / 62 + 1 / 61, places=6)
        self.assertIsInstance(out[0], MemoryItem)

    def test_limit_and_empty_query(self):
        dense = [_point(str(i), "alice", f"m{i}", 1 - i / 10) for i in range(6)]
        m, *_ = self.mem(dense)
        self.assertEqual(len(asyncio.run(m.search("alice", "q", 2))), 2)
        self.assertEqual(asyncio.run(m.search("alice", "   ", 2)), [])

    def test_one_store_down_degrades_both_down_raises(self):
        m, idx, q, o = self.mem([_point("a", "alice", "x", .9)])
        o.search.side_effect = RuntimeError("opensearch down")
        self.assertEqual([i.memory_id for i in asyncio.run(m.search("alice", "q", 5))], ["a"])
        q.query_points.side_effect = RuntimeError("qdrant down")
        with self.assertRaises(RuntimeError):
            asyncio.run(m.search("alice", "q", 5))

    def test_search_is_user_scoped(self):
        m, idx, q, o = self.mem()
        asyncio.run(m.search("bob", "q", 5))
        self.assertEqual(q.query_points.call_args.kwargs["query_filter"].must[0].match.value, "bob")
        self.assertEqual(o.search.call_args.kwargs["body"]["query"]["bool"]["filter"], [{"term": {"user_id": "bob"}}])
        with self.assertRaises(PermissionError):
            asyncio.run(m.search("", "q", 5))

    def test_add_validates_and_writes(self):
        m, idx, q, o = self.mem()
        item = asyncio.run(m.add("alice", "  Answer in   Hindi ", kind="preference"))
        self.assertEqual((item.text, item.kind), ("Answer in Hindi", "preference"))
        self.embed.assert_called_with("Answer in Hindi")
        self.assertEqual(q.upsert.call_args.kwargs["points"][0].payload["user_id"], "alice")
        for bad in ("", "x" * 5000):
            with self.assertRaises(ValueError):
                asyncio.run(m.add("alice", bad))
        with self.assertRaises(ValueError):
            asyncio.run(m.add("alice", "x", kind="secret"))
        q.count.return_value = SimpleNamespace(count=m.max_items_per_user)
        with self.assertRaises(ValueError):
            asyncio.run(m.add("alice", "one more"))

    def test_rrf_fuse_is_deterministic_on_ties(self):
        fused = rrf_fuse([{"memory_id": "b"}], [{"memory_id": "a"}], k=60)
        self.assertEqual([h["memory_id"] for h, _ in fused], ["a", "b"])


class ProviderSelectionTests(Harness):
    def test_null_is_default_hybrid_by_setting(self):
        self.assertIsInstance(build_memory_provider(isettings()), NullMemory)
        self.assertIsInstance(build_memory_provider(isettings(memory_backend="null")), NullMemory)
        with unittest.mock.patch("app.db.memory_store.MemoryIndex") as mi:
            self.assertIsInstance(build_memory_provider(isettings(memory_backend="hybrid")), HybridMemory)
        mi.assert_called_once()
        with self.assertRaises(ValueError):
            isettings(memory_backend="redis").validate()

    def test_orchestrator_defaults_to_null_and_passes_user_to_memory(self):
        o = self.orch([])
        self.assertIsInstance(o.memory, NullMemory)
        idx, q, _ = _index([_point("p", "alice", "Prefers short answers", .9)])
        m = HybridMemory(idx, MagicMock(return_value=[0.1, 0.2, 0.3]), timeout_s=2)
        from tests.test_inference import ai
        o = self.orch([ai("ok [S1]")], memory=m)
        r = self.run_(o)
        self.assertEqual(q.query_points.call_args.kwargs["query_filter"].must[0].match.value, "alice")
        system = self.log[0]["messages"][0].content
        self.assertIn("Prefers short answers", system)
        self.assertEqual(r.answer, "ok [S1]")


if __name__ == "__main__":
    unittest.main()
