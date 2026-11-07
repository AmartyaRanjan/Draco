"""Conversation + long-term memory interfaces, a SQLite conversation store, a hybrid long-term
memory provider, and rolling-summary generation.

The orchestrator depends only on the Protocols. Rolling summaries are an addition to, never a
replacement for, the stored messages: every message stays in `messages`; the summary only
replaces the *covered* messages in the prompt (`summary_upto` marks the last covered message id).
"""
import asyncio
import json
import logging
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from app.db import registry

logger = logging.getLogger("rag.inference")


@dataclass
class ConversationContext:
    summary: str | None
    recent: list[dict]  # [{"role": "user"|"assistant", "content": str}], oldest first


@dataclass
class MemoryItem:
    memory_id: str
    text: str
    kind: str = "fact"
    score: float | None = None  # RRF score from the hybrid provider


@dataclass
class SummaryState:
    summary: str | None
    summary_upto: int  # id of the last message the summary covers (0: none)
    messages: list[dict] = field(default_factory=list)  # uncovered: [{"id", "role", "content"}], oldest first


class ConversationStore(Protocol):
    def load(self, user_id: str, conversation_id: str, limit: int) -> ConversationContext: ...
    def save_turn(self, user_id: str, conversation_id: str, user_msg: str, response_json: str,
                  answer: str, tool_outputs: dict) -> None: ...


class SummarizableStore(Protocol):
    """Optional store capability used by RollingSummarizer."""
    def summary_state(self, user_id: str, conversation_id: str) -> SummaryState | None: ...
    def save_summary(self, user_id: str, conversation_id: str, summary: str, upto_message_id: int,
                     expected_upto: int) -> bool: ...


class MemoryProvider(Protocol):
    async def search(self, user_id: str, query: str, limit: int) -> list[MemoryItem]: ...


class NullMemory:
    """Default: no long-term memory (MEMORY_BACKEND=null)."""

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
                c.execute("PRAGMA journal_mode=WAL")
                c.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                  conversation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, summary TEXT,
                  summary_upto INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
                  role TEXT NOT NULL, content TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS inference_runs (
                  run_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, user_id TEXT NOT NULL,
                  response_json TEXT NOT NULL, tool_outputs_json TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);""")
                cols = {r[1] for r in c.execute("PRAGMA table_info(conversations)").fetchall()}
                if "summary_upto" not in cols:  # databases created before rolling summaries
                    c.execute("ALTER TABLE conversations ADD COLUMN summary_upto INTEGER NOT NULL DEFAULT 0")
                    c.commit()
                self._ready = True
        return c

    def owner(self, conversation_id: str) -> str | None:
        # closing() is required: sqlite3.Connection as a context manager only
        # commits/rolls back, it never closes.
        with closing(self._conn()) as c:
            r = c.execute("SELECT user_id FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
            return r["user_id"] if r else None

    @staticmethod
    def _upto(conv) -> int:
        return int(conv["summary_upto"] or 0) if "summary_upto" in conv.keys() else 0

    def load(self, user_id, conversation_id, limit):
        with closing(self._conn()) as c:
            conv = c.execute("SELECT * FROM conversations WHERE conversation_id=?",
                             (conversation_id,)).fetchone()
            if conv is None:
                return ConversationContext(None, [])
            if conv["user_id"] != user_id:  # never read another user's conversation
                raise PermissionError("conversation belongs to another user")
            # Messages already folded into the summary are represented by it, not repeated verbatim.
            rows = c.execute("SELECT role, content FROM messages WHERE conversation_id=? AND id>? "
                             "ORDER BY id DESC LIMIT ?", (conversation_id, self._upto(conv), limit)).fetchall()
            return ConversationContext(conv["summary"], [dict(r) for r in reversed(rows)])

    def save_turn(self, user_id, conversation_id, user_msg, response_json, answer, tool_outputs):
        run_id = json.loads(response_json)["metadata"]["request_id"]
        with closing(self._conn()) as c:
            with c:  # one transaction: all or nothing
                c.execute("INSERT OR IGNORE INTO conversations (conversation_id, user_id) VALUES (?, ?)",
                          (conversation_id, user_id))
                c.executemany("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                              [(conversation_id, "user", user_msg), (conversation_id, "assistant", answer)])
                c.execute("INSERT INTO inference_runs (run_id, conversation_id, user_id, response_json, "
                          "tool_outputs_json) VALUES (?, ?, ?, ?, ?)",
                          (run_id, conversation_id, user_id, response_json, json.dumps(tool_outputs)))

    # ---- rolling summary support ------------------------------------------------------
    def summary_state(self, user_id, conversation_id) -> SummaryState | None:
        with closing(self._conn()) as c:
            conv = c.execute("SELECT * FROM conversations WHERE conversation_id=?",
                             (conversation_id,)).fetchone()
            if conv is None:
                return None
            if conv["user_id"] != user_id:
                raise PermissionError("conversation belongs to another user")
            upto = self._upto(conv)
            rows = c.execute("SELECT id, role, content FROM messages WHERE conversation_id=? AND id>? ORDER BY id",
                             (conversation_id, upto)).fetchall()
            return SummaryState(conv["summary"], upto, [dict(r) for r in rows])

    def save_summary(self, user_id, conversation_id, summary, upto_message_id, expected_upto) -> bool:
        """Compare-and-set on summary_upto, so two concurrent summarisers cannot regress each other."""
        with closing(self._conn()) as c:
            with c:
                cur = c.execute("UPDATE conversations SET summary=?, summary_upto=? WHERE conversation_id=? "
                                "AND user_id=? AND summary_upto=? AND ?>summary_upto",
                                (summary, upto_message_id, conversation_id, user_id, expected_upto, upto_message_id))
                return cur.rowcount == 1


def new_conversation_id() -> str:
    return str(uuid.uuid4())


# ---- long-term memory: Qdrant (dense) + OpenSearch (BM25), RRF fused ----------------------------
def rrf_fuse(dense: list[dict], lexical: list[dict], k: int = 60) -> list[tuple[dict, float]]:
    """Reciprocal Rank Fusion over memory hits keyed by memory_id (1-based ranks, inputs pre-sorted).
    Ties break on the best single-source rank, then memory_id (deterministic)."""
    merged: dict[str, dict] = {}
    for source, hits in (("dense", dense), ("bm25", lexical)):
        for rank, h in enumerate(hits, 1):
            mid = h.get("memory_id")
            if not mid:
                continue
            e = merged.setdefault(mid, {"hit": h, "ranks": {}})
            e["ranks"].setdefault(source, rank)
    scored = [(e["hit"], sum(1.0 / (k + r) for r in e["ranks"].values()), min(e["ranks"].values()), mid)
              for mid, e in merged.items()]
    scored.sort(key=lambda t: (-t[1], t[2], t[3]))
    return [(h, s) for h, s, _, _ in scored]


def _default_embed(text: str) -> list[float]:
    # Lazy: loads BGE-M3 on first use only, and shares the retrieval GPU lock.
    from app.retrieval.retrievers import _embed_query
    return _embed_query(text)


class HybridMemory:
    """Long-term, strictly user-scoped memory items (preferences and facts the user asked to keep).

    Every operation requires the authenticated user_id; MemoryIndex puts the owner filter on every
    store query. Recall = dense + BM25 in parallel, fused with RRF; one store failing degrades to the
    other, both failing raises (the orchestrator treats memory as optional context and logs it)."""

    def __init__(self, index=None, embed: Callable[[str], list[float]] | None = None, *, rrf_k: int = 60,
                 candidates: int = 20, timeout_s: float = 5.0, max_text_chars: int = 1000,
                 max_items_per_user: int = 500):
        if index is None:
            from app.db.memory_store import MemoryIndex
            index = MemoryIndex()
        self.index, self.embed = index, embed or _default_embed
        self.rrf_k, self.candidates, self.timeout_s = rrf_k, candidates, timeout_s
        self.max_text_chars, self.max_items_per_user = max_text_chars, max_items_per_user

    async def _blocking(self, fn, *args, **kwargs):
        from app.core.blocking import run_blocking
        return await run_blocking(fn, *args, timeout=self.timeout_s, **kwargs)

    async def search(self, user_id, query, limit):
        query = re.sub(r"\s+", " ", query or "").strip()
        if not query or limit < 1:
            return []

        async def dense():
            vec = await self._blocking(self.embed, query)
            return await self._blocking(self.index.dense_search, user_id, vec, self.candidates, self.timeout_s)

        results = await asyncio.gather(
            dense(), self._blocking(self.index.lexical_search, user_id, query, self.candidates, self.timeout_s),
            return_exceptions=True)
        ok = []
        for name, r in zip(("dense", "bm25"), results, strict=True):
            if isinstance(r, asyncio.CancelledError):
                raise r
            if isinstance(r, PermissionError):
                raise r
            if isinstance(r, BaseException):
                logger.warning("memory_retriever_failed", extra={"retriever": name, "error": type(r).__name__})
                ok.append(None)
            else:
                ok.append(r)
        if ok[0] is None and ok[1] is None:
            raise RuntimeError("both memory retrievers failed")
        fused = rrf_fuse(ok[0] or [], ok[1] or [], self.rrf_k)
        return [MemoryItem(h["memory_id"], h.get("text", ""), h.get("kind") or "fact", round(score, 6))
                for h, score in fused[:limit] if h.get("user_id") == user_id]

    async def add(self, user_id: str, text: str, kind: str = "fact") -> MemoryItem:
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            raise ValueError("memory text is empty")
        if len(text) > self.max_text_chars:
            raise ValueError(f"memory text exceeds {self.max_text_chars} characters")
        if kind not in ("fact", "preference"):
            raise ValueError("kind must be 'fact' or 'preference'")
        if await self._blocking(self.index.count, user_id) >= self.max_items_per_user:
            raise ValueError("memory limit reached for this user")
        item = MemoryItem(str(uuid.uuid4()), text, kind)
        vec = await self._blocking(self.embed, text)
        await self._blocking(self.index.upsert, user_id, item.memory_id, text, vec,
                             kind=kind, created_at=datetime.now(UTC).isoformat())
        return item

    async def delete(self, user_id: str, memory_id: str) -> None:
        await self._blocking(self.index.delete, user_id, memory_id)

    async def list(self, user_id: str, limit: int = 100) -> list[MemoryItem]:
        rows = await self._blocking(self.index.list_items, user_id, limit)
        return [MemoryItem(r["memory_id"], r.get("text", ""), r.get("kind") or "fact") for r in rows]


def build_memory_provider(settings) -> MemoryProvider:
    """MEMORY_BACKEND=null (default) -> NullMemory; hybrid -> HybridMemory over Qdrant + OpenSearch."""
    backend = getattr(settings, "memory_backend", "null")
    if backend == "hybrid":
        return HybridMemory()
    return NullMemory()


# ---- rolling summary --------------------------------------------------------------------------
_THINK = re.compile(r"<think>.*?</think>", re.S)


class RollingSummarizer:
    """Folds older turns into `conversations.summary` once unsummarised history exceeds a token threshold.

    Runs after a turn is persisted (background task), never on the request's critical path. The newest
    messages that fit `history_tokens` stay verbatim; everything older is summarised together with the
    previous summary, in batches of `summary_input_tokens`. Any failure leaves the previous state intact."""

    def __init__(self, store, settings, llm_factory, counter):
        self.store, self.s, self.llm_factory, self.counter = store, settings, llm_factory, counter
        self._tasks: set[asyncio.Task] = set()
        self._running: set[tuple[str, str]] = set()

    def _truncate(self, text: str, tokens: int) -> str:
        if self.counter.count(text) <= tokens:
            return text
        # Character cut proportional to the token overrun; good enough for a summary input.
        ratio = tokens / max(1, self.counter.count(text))
        return text[: max(1, int(len(text) * ratio * 0.95))] + " …"

    async def _summarise(self, previous: str | None, batch: list[dict]) -> str | None:
        from langchain_core.messages import HumanMessage, SystemMessage

        from app.inference import prompts

        transcript = "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in batch)
        body = (f"PREVIOUS SUMMARY:\n{previous}\n\n" if previous else "") + f"NEW TURNS:\n{transcript}\n\nUPDATED SUMMARY:"
        llm = self.llm_factory(self.s, thinking=False, max_tokens=self.s.summary_max_tokens, streaming=False)
        msg = await asyncio.wait_for(llm.ainvoke([SystemMessage(prompts.SUMMARY_SYSTEM), HumanMessage(body)]),
                                     self.s.timeout_s)
        text = _THINK.sub("", msg.content if isinstance(msg.content, str) else "").strip()
        return text or None

    async def maybe_summarize(self, user_id: str, conversation_id: str) -> bool:
        """True if a new summary was stored."""
        get_state = getattr(self.store, "summary_state", None)
        if get_state is None or not hasattr(self.store, "save_summary"):
            return False
        state = await asyncio.to_thread(get_state, user_id, conversation_id)
        if state is None or not state.messages:
            return False
        sizes = [self.counter.count(m["content"]) for m in state.messages]
        if sum(sizes) <= self.s.summary_trigger_tokens:
            return False
        keep, room = 0, self.s.history_tokens  # newest whole messages that the prompt shows verbatim
        for n in reversed(sizes):
            if n > room:
                break
            room -= n
            keep += 1
        older = state.messages[: len(state.messages) - keep]
        if not older:
            return False
        per_msg = max(64, self.s.summary_input_tokens // 2)
        summary, batch, used = state.summary, [], 0
        for m in older:
            m = {**m, "content": self._truncate(m["content"], per_msg)}
            n = self.counter.count(m["content"])
            if batch and used + n > self.s.summary_input_tokens:
                summary = await self._summarise(summary, batch)
                if summary is None:
                    return False
                batch, used = [], 0
            batch.append(m)
            used += n
        if batch:
            summary = await self._summarise(summary, batch)
            if summary is None:
                return False
        saved = await asyncio.to_thread(self.store.save_summary, user_id, conversation_id, summary,
                                        int(older[-1]["id"]), state.summary_upto)
        logger.info("conversation_summarized", extra={"messages": len(older), "saved": saved})
        return saved

    async def _guarded(self, key):
        try:
            await self.maybe_summarize(*key)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("summary_failed", extra={"error": type(exc).__name__})
        finally:
            self._running.discard(key)

    def schedule(self, user_id: str, conversation_id: str) -> asyncio.Task | None:
        """Start a background summary for this conversation unless one is already running."""
        key = (user_id, conversation_id)
        if key in self._running:
            return None
        self._running.add(key)
        task = asyncio.get_running_loop().create_task(self._guarded(key))
        self._tasks.add(task)  # strong reference until done
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self) -> None:
        """Await pending summaries (shutdown, tests)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
