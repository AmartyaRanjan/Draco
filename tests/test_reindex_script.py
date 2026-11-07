"""scripts/reindex.py selection and dry-run behaviour (never touches a store: the worker is mocked)."""
import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.db import registry

_SPEC = importlib.util.spec_from_file_location(
    "reindex_script", Path(__file__).resolve().parents[1] / "scripts" / "reindex.py")
reindex = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reindex)


class ReindexScriptTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = Path(tmp.name) / "documents.db"
        p = patch.object(registry, "DB_PATH", self.db)
        p.start()
        self.addCleanup(p.stop)
        registry.init_db()
        for doc, owner, status in (("a", "alice", "completed"), ("b", "bob", "completed"), ("c", "alice", "failed")):
            registry.create_document(doc, f"{doc}.md", f"/{doc}.md", owner)
            registry.update_status(doc, status)
        self.reindex_document = MagicMock(side_effect=lambda d: {"chunk_count": 1})
        p = patch.dict(sys.modules, {"app.ingestion.worker": SimpleNamespace(reindex_document=self.reindex_document)})
        p.start()
        self.addCleanup(p.stop)

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = reindex.main([*argv, "--database", str(self.db)])
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_by_default(self):
        code, out, _ = self.run_main("--all")
        self.assertEqual(code, 0)
        self.assertIn("Dry run: 2 document(s)", out)
        self.reindex_document.assert_not_called()

    def test_only_completed_documents_and_owner_scope(self):
        self.run_main("--owner", "alice", "--yes")
        self.assertEqual([c.args[0] for c in self.reindex_document.call_args_list], ["a"])
        self.reindex_document.reset_mock()
        self.run_main("--document-id", "c", "--document-id", "b", "--yes")
        self.assertEqual([c.args[0] for c in self.reindex_document.call_args_list], ["b"])

    def test_failures_are_reported_and_set_exit_code(self):
        self.reindex_document.side_effect = RuntimeError("boom")
        code, out, err = self.run_main("--all", "--yes")
        self.assertEqual(code, 1)
        self.assertIn("FAIL", err)
        self.assertIn("0 of 2", out)


if __name__ == "__main__":
    unittest.main()
