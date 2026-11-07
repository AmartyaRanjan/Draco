"""/memory routes: user scoping, validation, disabled backend. The index is an in-memory fake."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api import chat as chat_api
from app.inference.memory import HybridMemory, NullMemory
from app.main import app
from fastapi.testclient import TestClient

from tests.agent_helpers import isettings

KEYS = {"k-alice": "alice", "k-bob": "bob"}


class FakeIndex:
    def __init__(self):
        self.rows: dict[tuple[str, str], dict] = {}

    def count(self, user):
        return sum(1 for (u, _) in self.rows if u == user)

    def upsert(self, user, memory_id, text, vec, kind="fact", created_at=""):
        self.rows[(user, memory_id)] = {"memory_id": memory_id, "text": text, "kind": kind}

    def delete(self, user, memory_id):
        self.rows.pop((user, memory_id), None)

    def delete_user(self, user):
        for k in [k for k in self.rows if k[0] == user]:
            del self.rows[k]

    def list_items(self, user, limit):
        return [r for (u, _), r in self.rows.items() if u == user][:limit]


def hdr(user):
    return {"X-API-Key": f"k-{user}"}


class MemoryApiTests(unittest.TestCase):
    def setUp(self):
        self.index = FakeIndex()
        memory = HybridMemory(self.index, lambda text: [0.1, 0.2, 0.3], timeout_s=2, max_items_per_user=3)
        app.dependency_overrides[chat_api.get_orchestrator] = lambda: SimpleNamespace(memory=memory)
        self.addCleanup(app.dependency_overrides.clear)
        p = patch("app.api.chat.get_inference_settings", return_value=isettings(api_keys=KEYS))
        p.start()
        self.addCleanup(p.stop)
        self.c = TestClient(app)

    def add(self, user, text, kind="fact"):
        return self.c.post("/memory", json={"text": text, "kind": kind}, headers=hdr(user))

    def test_add_list_delete_roundtrip(self):
        r = self.add("alice", "  answers in   Hindi ", "preference")
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json()["text"], "answers in Hindi")  # whitespace normalised
        items = self.c.get("/memory", headers=hdr("alice")).json()["items"]
        self.assertEqual([i["text"] for i in items], ["answers in Hindi"])
        self.assertEqual(self.c.delete(f"/memory/{r.json()['memory_id']}", headers=hdr("alice")).status_code, 204)
        self.assertEqual(self.c.get("/memory", headers=hdr("alice")).json()["items"], [])

    def test_users_cannot_see_or_delete_each_others_items(self):
        mid = self.add("alice", "lives in Pune").json()["memory_id"]
        self.assertEqual(self.c.get("/memory", headers=hdr("bob")).json()["items"], [])
        self.c.delete(f"/memory/{mid}", headers=hdr("bob"))  # bob's delete is scoped to bob: a no-op
        self.assertEqual(len(self.c.get("/memory", headers=hdr("alice")).json()["items"]), 1)

    def test_erase_removes_only_the_callers_items(self):
        self.add("alice", "one")
        self.add("bob", "two")
        self.assertEqual(self.c.delete("/memory", headers=hdr("alice")).status_code, 204)
        self.assertEqual(self.c.get("/memory", headers=hdr("alice")).json()["items"], [])
        self.assertEqual(len(self.c.get("/memory", headers=hdr("bob")).json()["items"]), 1)

    def test_validation_errors(self):
        self.assertEqual(self.add("alice", "x", "secret").status_code, 422)   # kind
        self.assertEqual(self.add("alice", "   ").status_code, 422)            # blank after normalising
        self.assertEqual(self.add("alice", "y" * 1001).status_code, 422)       # too long
        r = self.c.post("/memory", json={"text": "z", "owner": "bob"}, headers=hdr("alice"))
        self.assertEqual(r.status_code, 422)                                   # extra fields refused
        for t in ("a", "b", "c"):
            self.assertEqual(self.add("alice", t).status_code, 201)
        self.assertEqual(self.add("alice", "d").status_code, 422)              # per-user limit
        self.assertEqual(self.add("bob", "d").status_code, 201)                # limit is per user

    def test_requires_a_valid_key(self):
        self.assertEqual(self.c.get("/memory").status_code, 401)
        self.assertEqual(self.c.get("/memory", headers={"X-API-Key": "nope"}).status_code, 401)

    def test_disabled_backend_answers_501(self):
        app.dependency_overrides[chat_api.get_orchestrator] = lambda: SimpleNamespace(memory=NullMemory())
        self.assertEqual(self.c.get("/memory", headers=hdr("alice")).status_code, 501)
        self.assertEqual(self.add("alice", "x").status_code, 501)


if __name__ == "__main__":
    unittest.main()
