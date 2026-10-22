"""Regression tests for owner-scoped document listing in the inference tool."""
import asyncio
import unittest
from unittest.mock import patch

from app.inference import tools as tools_mod


class ListToolConnectionTests(unittest.TestCase):
    def test_list_documents_scopes_to_authenticated_user(self):
        with patch.object(tools_mod.registry, "list_documents_for_owner", return_value=[]) as listing:
            text, summary = asyncio.run(
                tools_mod._list(
                    tools_mod.ListArgs(),
                    tools_mod.ToolContext(user_id="alice", search=None),
                )
            )
        listing.assert_called_once_with("alice", 50)
        self.assertEqual((text, summary), ("No documents.", "Listed 0 documents"))

    def test_list_documents_formats_owner_scoped_rows(self):
        rows = [
            {"document_id": "d1", "filename": "a.pdf", "status": "completed"},
            {"document_id": "d2", "filename": "b.pdf", "status": "queued"},
        ]
        with patch.object(tools_mod.registry, "list_documents_for_owner", return_value=rows):
            text, summary = asyncio.run(
                tools_mod._list(
                    tools_mod.ListArgs(),
                    tools_mod.ToolContext(user_id="alice", search=None),
                )
            )
        self.assertIn("d1 | a.pdf | completed", text)
        self.assertIn("d2 | b.pdf | queued", text)
        self.assertEqual(summary, "Listed 2 documents")


if __name__ == "__main__":
    unittest.main()
