"""Durable agent state (SQLite, own file; the ingestion registry DB is never touched).

Holds web-source candidates, approval records, ingestion jobs, feedback and artifact metadata.
Approval rows are only ever written by `approve_source`, which the HTTP layer calls for an
authenticated user; no model-callable capability can reach it.
"""
import json
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

from app.agent.contracts import new_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, url TEXT NOT NULL, normalized_url TEXT NOT NULL,
  url_hash TEXT NOT NULL, title TEXT, content_hash TEXT NOT NULL, content_text TEXT NOT NULL,
  reason TEXT, preview_json TEXT NOT NULL, trust_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending_approval',   -- pending_approval|approved|rejected|ingesting|ingested|failed|expired
  created_at REAL NOT NULL, expires_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_sources_user_url ON sources(user_id, url_hash);
CREATE TABLE IF NOT EXISTS approvals (
  approval_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(source_id),
  user_id TEXT NOT NULL, url_hash TEXT NOT NULL, content_hash TEXT NOT NULL,
  decision TEXT NOT NULL, decided_at REAL NOT NULL, expires_at REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
CREATE UNIQUE INDEX IF NOT EXISTS idx_approvals_source_user ON approvals(source_id, user_id);
CREATE TABLE IF NOT EXISTS ingestion_jobs (
  job_id TEXT PRIMARY KEY, source_id TEXT NOT NULL UNIQUE REFERENCES sources(source_id),
  idempotency_key TEXT NOT NULL UNIQUE, document_id TEXT, rq_job_id TEXT,
  status TEXT NOT NULL, error TEXT, lease_until REAL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS feedback (
  feedback_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, target_type TEXT NOT NULL, target_id TEXT NOT NULL,
  rating INTEGER NOT NULL, comment TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, fmt TEXT NOT NULL,
  filename TEXT NOT NULL, storage_key TEXT NOT NULL, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
  status TEXT NOT NULL, meta_json TEXT, created_at REAL NOT NULL, expires_at REAL NOT NULL);
"""


class AgentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock, self._ready = threading.Lock(), False

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        if not self._ready:
            with self._lock:
                c.execute("PRAGMA journal_mode=WAL")
                c.executescript(SCHEMA)
                columns = {row["name"] for row in c.execute("PRAGMA table_info(ingestion_jobs)")}
                if "lease_until" not in columns:
                    c.execute("ALTER TABLE ingestion_jobs ADD COLUMN lease_until REAL")
                c.execute(
                    "UPDATE ingestion_jobs SET lease_until=updated_at+300 "
                    "WHERE status='claimed' AND lease_until IS NULL"
                )
                c.commit()
                self._ready = True
        return c

    def _one(self, sql, args=()):
        with closing(self._conn()) as c:
            r = c.execute(sql, args).fetchone()
        return dict(r) if r else None

    def _all(self, sql, args=()):
        with closing(self._conn()) as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def _write(self, sql, args=()):
        with closing(self._conn()) as c, c:
            return c.execute(sql, args).rowcount

    # ---- sources / approvals / jobs ---------------------------------------------------------
    def create_source(self, *, user_id, url, normalized_url, url_hash, title, content_hash, content_text,
                      reason, preview: dict, trust: dict, ttl_hours: float) -> dict:
        sid, now = new_id(), time.time()
        self._write("INSERT INTO sources (source_id,user_id,url,normalized_url,url_hash,title,content_hash,content_text,"
                    "reason,preview_json,trust_json,created_at,expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sid, user_id, url, normalized_url, url_hash, title, content_hash, content_text, reason,
                     json.dumps(preview), json.dumps(trust), now, now + ttl_hours * 3600))
        return self.get_source(sid)

    def get_source(self, source_id: str, user_id: str | None = None) -> dict | None:
        r = self._one("SELECT * FROM sources WHERE source_id=?", (source_id,))
        if r is None or (user_id is not None and r["user_id"] != user_id):
            return None
        r["preview"], r["trust"] = json.loads(r.pop("preview_json")), json.loads(r.pop("trust_json"))
        return r

    def sources_for_url(self, user_id: str, url_hash: str) -> list[dict]:
        return self._all("SELECT source_id,status,created_at FROM sources WHERE user_id=? AND url_hash=? "
                         "ORDER BY created_at DESC", (user_id, url_hash))

    def set_source_status(self, source_id: str, status: str):
        self._write("UPDATE sources SET status=? WHERE source_id=?", (status, source_id))

    def approve_source(self, source_id: str, user_id: str, *, approve: bool = True, ttl_hours: float = 24) -> dict:
        """Record the user's explicit decision for exactly this source (url + content hash).
        Called only from the authenticated HTTP layer."""
        src = self.get_source(source_id, user_id)
        if src is None:
            raise KeyError("source not found")
        if src["status"] not in ("pending_approval", "approved", "rejected"):
            raise ValueError(f"source is {src['status']}")
        if src["expires_at"] < time.time():
            self.set_source_status(source_id, "expired")
            raise ValueError("proposal expired")
        now = time.time()
        with closing(self._conn()) as c, c:
            c.execute("INSERT INTO approvals (approval_id,source_id,user_id,url_hash,content_hash,decision,decided_at,expires_at) "
                      "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(source_id,user_id) DO UPDATE SET decision=excluded.decision,"
                      "decided_at=excluded.decided_at,expires_at=excluded.expires_at,revoked=0",
                      (new_id(), source_id, user_id, src["url_hash"], src["content_hash"],
                       "approved" if approve else "rejected", now, now + ttl_hours * 3600))
            c.execute("UPDATE sources SET status=? WHERE source_id=?", ("approved" if approve else "rejected", source_id))
        return self.get_approval(source_id, user_id)

    def get_approval(self, source_id: str, user_id: str) -> dict | None:
        return self._one("SELECT * FROM approvals WHERE source_id=? AND user_id=?", (source_id, user_id))

    def valid_approval(self, source_id: str, user_id: str) -> dict | None:
        """An approval counts only if it is an 'approved' decision, unexpired, unrevoked and bound to
        the same url hash and content snapshot the user was shown."""
        src, ap = self.get_source(source_id, user_id), self.get_approval(source_id, user_id)
        if not src or not ap or ap["decision"] != "approved" or ap["revoked"] or ap["expires_at"] < time.time():
            return None
        if ap["url_hash"] != src["url_hash"] or ap["content_hash"] != src["content_hash"]:
            return None
        return ap

    def claim_job(self, source_id: str, idempotency_key: str) -> tuple[dict, bool]:
        """Create a job or atomically reclaim a claim whose renewable lease expired."""
        now = time.time()
        with closing(self._conn()) as c:
            c.execute("BEGIN IMMEDIATE")
            job = c.execute("SELECT * FROM ingestion_jobs WHERE source_id=?", (source_id,)).fetchone()
            if job is None:
                c.execute("INSERT INTO ingestion_jobs (job_id,source_id,idempotency_key,status,created_at,updated_at) "
                          "VALUES (?,?,?,?,?,?)", (new_id(), source_id, idempotency_key, "claimed", now, now))
                c.execute("UPDATE ingestion_jobs SET lease_until=? WHERE source_id=?", (now + 300, source_id))
                c.commit()
                return self._one("SELECT * FROM ingestion_jobs WHERE source_id=?", (source_id,)), True
            if job["status"] == "claimed" and (job["lease_until"] is None or job["lease_until"] <= now):
                changed = c.execute(
                    "UPDATE ingestion_jobs SET lease_until=?, error=NULL, updated_at=? "
                    "WHERE job_id=? AND status='claimed' AND (lease_until IS NULL OR lease_until<=?)",
                    (now + 300, now, job["job_id"], now),
                ).rowcount
                c.commit()
                return self._one("SELECT * FROM ingestion_jobs WHERE source_id=?", (source_id,)), changed == 1
            c.rollback()
        return self._one("SELECT * FROM ingestion_jobs WHERE source_id=?", (source_id,)), False

    def update_job(self, job_id: str, **fields):
        allowed = {"document_id", "rq_job_id", "status", "error", "lease_until"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if sets.get("status") != "claimed" and "lease_until" not in sets:
            sets["lease_until"] = None
        if sets:
            self._write(f"UPDATE ingestion_jobs SET {','.join(k + '=?' for k in sets)},updated_at=? WHERE job_id=?",
                        (*sets.values(), time.time(), job_id))

    def renew_claim(self, job_id: str, lease_seconds: float = 300) -> bool:
        now = time.time()
        return self._write(
            "UPDATE ingestion_jobs SET lease_until=?, updated_at=? "
            "WHERE job_id=? AND status='claimed' AND lease_until>?",
            (now + lease_seconds, now, job_id, now),
        ) == 1

    def job_for_source(self, source_id: str) -> dict | None:
        return self._one("SELECT * FROM ingestion_jobs WHERE source_id=?", (source_id,))

    # ---- feedback ---------------------------------------------------------------------------
    def add_feedback(self, user_id, target_type, target_id, rating, comment) -> str:
        fid = new_id()
        self._write("INSERT INTO feedback VALUES (?,?,?,?,?,?,?)",
                    (fid, user_id, target_type, target_id, rating, comment, time.time()))
        return fid

    def purge_feedback(self, retention_days: int) -> int:
        return self._write("DELETE FROM feedback WHERE created_at < ?", (time.time() - retention_days * 86400,))

    def delete_user_feedback(self, user_id: str) -> int:
        return self._write("DELETE FROM feedback WHERE user_id=?", (user_id,))

    def feedback_for(self, user_id: str, target_id: str) -> list[dict]:
        return self._all("SELECT * FROM feedback WHERE user_id=? AND target_id=?", (user_id, target_id))

    # ---- artifacts --------------------------------------------------------------------------
    def add_artifact(self, *, user_id, kind, fmt, filename, storage_key, size_bytes, sha256, ttl_days, meta=None,
                     status="ready") -> dict:
        aid, now = new_id(), time.time()
        self._write("INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (aid, user_id, kind, fmt, filename, storage_key, size_bytes, sha256, status,
                     json.dumps(meta or {}), now, now + ttl_days * 86400))
        return self.get_artifact(aid)

    def get_artifact(self, artifact_id: str, user_id: str | None = None) -> dict | None:
        r = self._one("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,))
        if r is None or (user_id is not None and r["user_id"] != user_id):
            return None
        r["meta"] = json.loads(r.pop("meta_json") or "{}")
        return r

    def expired_artifacts(self) -> list[dict]:
        return self._all("SELECT artifact_id, storage_key FROM artifacts WHERE expires_at < ? AND status != 'deleted'",
                         (time.time(),))

    def mark_artifact(self, artifact_id: str, status: str):
        self._write("UPDATE artifacts SET status=? WHERE artifact_id=?", (status, artifact_id))

    def reclaim_failed_job(self, job_id: str) -> bool:
        """Atomically move a failed job back to 'claimed' so exactly one caller resubmits it."""
        now = time.time()
        return self._write(
            "UPDATE ingestion_jobs SET status='claimed', error=NULL, lease_until=?, updated_at=? "
            "WHERE job_id=? AND status='failed'",
            (now + 300, now, job_id),
        ) == 1
