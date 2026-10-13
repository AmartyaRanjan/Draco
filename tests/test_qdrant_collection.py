"""Regression tests: Qdrant payload indexes must exist even for old collections.

ensure_collection() returned early when the collection already existed,
skipping the document_id/document_version payload indexes that document-
scoped filtering (retriever filters, delete_stale_versions) depends on.
Collections created before the indexes were added stay unindexed: filtered
search/cleanup is slow or rejected under strict mode. Indexes must be ensured
on both paths and be idempotent when they already exist.
"""
import sys
import unittest
from unittest.mock import MagicMock

for _name in ("qdrant_client", "qdrant_client.models"):
    sys.modules.setdefault(_name, MagicMock())

import app.db.qdrant as qd  # noqa: E402


class EnsureCollectionIndexTests(unittest.TestCase):
    def setUp(self):
        self._prev = qd._client
        self.addCleanup(setattr, qd, "_client", self._prev)

    def test_existing_collection_still_creates_indexes(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        qd._client = client
        qd.ensure_collection()
        client.create_collection.assert_not_called()
        self.assertEqual(client.create_payload_index.call_count, 2)
        fields = sorted(
            c.kwargs.get("field_name") for c in client.create_payload_index.call_args_list
        )
        self.assertEqual(fields, ["document_id", "document_version"])

    def test_missing_collection_creates_collection_and_indexes(self):
        client = MagicMock()
        client.collection_exists.return_value = False
        qd._client = client
        qd.ensure_collection()
        client.create_collection.assert_called_once()
        self.assertEqual(client.create_payload_index.call_count, 2)

    def test_existing_index_is_tolerated(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.create_payload_index.side_effect = Exception(
            "index already exists for field document_id"
        )
        qd._client = client
        # Must not raise for idempotent re-creation.
        qd.ensure_collection()
        self.assertEqual(client.create_payload_index.call_count, 2)

    def test_unrelated_index_error_propagates(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.create_payload_index.side_effect = RuntimeError("qdrant down")
        qd._client = client
        with self.assertRaises(RuntimeError):
            qd.ensure_collection()


if __name__ == "__main__":
    unittest.main()
