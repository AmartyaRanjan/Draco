"""Conversation + long-term memory interfaces and a SQLite store.

The planned PostgreSQL / Qdrant+OpenSearch memory layer plugs in by implementing
these Protocols; the orchestrator depends only on them. Rolling summaries are an
addition to, never a replacement for, the stored messages.
"""
import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.db import registry


@dataclass
class ConversationContext:
    summary: str | None
    recent: list[dict]  # [{"role": "user"|"assistant", "content": str}], oldest first


@dataclass
class MemoryItem:
    memory_id: str
    text: str


class ConversationStore(Protocol):
    def load(self, user_id: str, conversation_id: str, limit: int) -> ConversationContext: ...
    def save_turn(self, user_id: str, conversation_id: str, user_msg: str, response_json: str,
                  answer: str, tool_outputs: dict) -> None: ...


class MemoryProvider(Protocol):
    async def search(self, user_id: str, query: str, limit: int) -> list[MemoryItem]: ...


class NullMemory:
    """Default until the long-term memory service exists."""

    async def search(self, user_id, query, limit):
        return []


class SqliteConversationStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else registry.DB_PATH.with_name("conversations.db")  # never the ingestion registry DB
        self._lock, self._ready = threading.Lock(), False

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        if not self._ready:
            with self._lock:
                c.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                  conversation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, summary TEXT,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
                  role TEXT NOT NULL, content TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS inference_runs (
                  run_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, user_id TEXT NOT NULL,
                  response_json TEXT NOT NULL, tool_outputs_json TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);""")
                self._ready = True
        return c

    def owner(self, conversation_id: str) -> str | None:
        with self._conn() as c:
            r = c.execute("SELECT user_id FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
        return r["user_id"] if r else None

    def load(self, user_id, conversation_id, limit):
        with self._conn() as c:
            conv = c.execute("SELECT user_id, summary FROM conversations WHERE conversation_id=?",
                             (conversation_id,)).fetchone()
            if conv is None:
                return ConversationContext(None, [])
            if conv["user_id"] != user_id:  # never read another user's conversation
                raise PermissionError("conversation belongs to another user")
            rows = c.execute("SELECT role, content FROM messages WHERE conversation_id=? "
                             "ORDER BY id DESC LIMIT ?", (conversation_id, limit)).fetchall()
        return ConversationContext(conv["summary"], [dict(r) for r in reversed(rows)])

    def save_turn(self, user_id, conversation_id, user_msg, response_json, answer, tool_outputs):
        run_id = json.loads(response_json)["metadata"]["request_id"]
        with self._conn() as c:  # one transaction: all or nothing
            c.execute("INSERT OR IGNORE INTO conversations (conversation_id, user_id) VALUES (?, ?)",
                      (conversation_id, user_id))
            c.executemany("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                          [(conversation_id, "user", user_msg), (conversation_id, "assistant", answer)])
            c.execute("INSERT INTO inference_runs (run_id, conversation_id, user_id, response_json, "
                      "tool_outputs_json) VALUES (?, ?, ?, ?, ?)",
                      (run_id, conversation_id, user_id, response_json, json.dumps(tool_outputs)))


def new_conversation_id() -> str:
    return str(uuid.uuid4())
