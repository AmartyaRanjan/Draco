"""Periodic cleanup (run from a scheduler: deploy/maintenance/ has a systemd timer and a compose service).

  python scripts/agent_maintenance.py [--dry-run] [--conversation-days N] [--agent-state-days N]

What it removes:
  * expired agent artifacts (files + rows) and feedback older than FEEDBACK_RETENTION_DAYS   (as before)
  * conversation history: whole conversations (messages, inference runs, the conversation row) whose most recent
    activity is older than CONVERSATION_RETENTION_DAYS (default 90; 0 keeps history forever)
  * agent state older than AGENT_STATE_RETENTION_DAYS (default 30; 0 keeps it forever):
      - agent_executions not in flight (status not running/resuming) last updated before the cutoff
      - LangGraph checkpoint threads (AGENT_CHECKPOINT_PATH) whose newest checkpoint is older than the cutoff,
        except threads of executions still running/resuming
      - unapproved web sources (pending_approval/rejected/expired) past their expiry + cutoff with no ingestion job,
        and their approval rows

--dry-run reports what would be deleted and changes nothing. Output is one JSON object on stdout.
Exit code is non-zero if any step failed (the other steps still run).
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LIVE = ("running", "resuming")
_CKPT_TABLES = ("graph_task_writes", "graph_checkpoint_channels", "graph_channel_blobs", "graph_checkpoints")


def _tables(c: sqlite3.Connection) -> set[str]:
    return {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _connect(path: str | Path) -> sqlite3.Connection | None:
    """Open an existing database only: maintenance must never create an empty store."""
    if not path or not Path(path).is_file():
        return None
    return sqlite3.connect(path, timeout=30)


def purge_conversations(path: str | Path, days: int, dry_run: bool = False) -> dict:
    out = {"path": str(path), "conversations": 0, "messages": 0, "inference_runs": 0}
    if days <= 0:
        return {**out, "skipped": "retention disabled"}
    c = _connect(path)
    if c is None:
        return {**out, "skipped": "database not found"}
    with closing(c):
        if not {"conversations", "messages", "inference_runs"} <= _tables(c):
            return {**out, "skipped": "schema not initialised"}
        # SQLite CURRENT_TIMESTAMP is UTC 'YYYY-MM-DD HH:MM:SS', so string comparison with datetime() is exact.
        stale = [r[0] for r in c.execute(
            "SELECT c.conversation_id FROM conversations c WHERE max(c.created_at, "
            " COALESCE((SELECT max(m.created_at) FROM messages m WHERE m.conversation_id=c.conversation_id), ''), "
            " COALESCE((SELECT max(r.created_at) FROM inference_runs r WHERE r.conversation_id=c.conversation_id), '')"
            ") < datetime('now', ?)", (f"-{int(days)} days",))]
        if not stale:
            return out
        with c:
            for i in range(0, len(stale), 500):
                ids = stale[i:i + 500]
                q = ",".join("?" * len(ids))
                counts = {t: c.execute(f"SELECT count(*) FROM {t} WHERE conversation_id IN ({q})", ids).fetchone()[0]
                          for t in ("conversations", "messages", "inference_runs")}
                for t, n in counts.items():
                    out[t] += n
                if not dry_run:
                    for t in ("messages", "inference_runs", "conversations"):
                        c.execute(f"DELETE FROM {t} WHERE conversation_id IN ({q})", ids)
    return out


def purge_agent_state(store_path: str | Path, checkpoint_path: str | Path | None, days: int,
                      dry_run: bool = False) -> dict:
    out = {"executions": 0, "checkpoint_threads": 0, "sources": 0, "approvals": 0}
    if days <= 0:
        return {**out, "skipped": "retention disabled"}
    cutoff = time.time() - days * 86400
    live: set[str] = set()
    c = _connect(store_path)
    if c is None:
        out["store_skipped"] = "database not found"
    else:
        with closing(c), c:
            tables = _tables(c)
            if "agent_executions" in tables:
                live = {r[0] for r in c.execute(
                    "SELECT execution_id FROM agent_executions WHERE status IN (?,?)", _LIVE)}
                where = "status NOT IN (?,?) AND updated_at < ?"
                out["executions"] = c.execute(f"SELECT count(*) FROM agent_executions WHERE {where}",
                                              (*_LIVE, cutoff)).fetchone()[0]
                if not dry_run:
                    c.execute(f"DELETE FROM agent_executions WHERE {where}", (*_LIVE, cutoff))
            if {"sources", "approvals", "ingestion_jobs"} <= tables:
                sel = ("SELECT source_id FROM sources WHERE status IN ('pending_approval','rejected','expired') "
                       "AND expires_at < ? AND source_id NOT IN (SELECT source_id FROM ingestion_jobs)")
                ids = [r[0] for r in c.execute(sel, (cutoff,))]
                for i in range(0, len(ids), 500):
                    chunk = ids[i:i + 500]
                    q = ",".join("?" * len(chunk))
                    out["approvals"] += c.execute(f"SELECT count(*) FROM approvals WHERE source_id IN ({q})",
                                                  chunk).fetchone()[0]
                    out["sources"] += len(chunk)
                    if not dry_run:
                        c.execute(f"DELETE FROM approvals WHERE source_id IN ({q})", chunk)
                        c.execute(f"DELETE FROM sources WHERE source_id IN ({q})", chunk)
    k = _connect(checkpoint_path) if checkpoint_path else None
    if k is None:
        out["checkpoints_skipped"] = "database not found"
    else:
        with closing(k), k:
            if set(_CKPT_TABLES) <= _tables(k):
                threads = [r[0] for r in k.execute(
                    "SELECT thread_id FROM graph_checkpoints GROUP BY thread_id HAVING max(created_at) < ?", (cutoff,))
                    if r[0] not in live]
                out["checkpoint_threads"] = len(threads)
                if not dry_run:
                    for t in threads:
                        for table in _CKPT_TABLES:
                            k.execute(f"DELETE FROM {table} WHERE thread_id=?", (t,))
    return out


def _artifacts_and_feedback(dry_run: bool) -> dict:
    from app.agent.artifacts import ArtifactStore
    from app.agent.config import get_agent_settings
    from app.agent.store import AgentStore

    s = get_agent_settings()
    if dry_run and not Path(s.store_path).is_file():
        return {"artifacts_deleted": 0, "feedback_purged": 0, "skipped": "database not found"}
    store = AgentStore(s.store_path)
    if dry_run:
        with closing(sqlite3.connect(s.store_path, timeout=30)) as c:
            fb = c.execute("SELECT count(*) FROM feedback WHERE created_at < ?",
                           (time.time() - s.feedback_retention_days * 86400,)).fetchone()[0]
        return {"artifacts_deleted": len(store.expired_artifacts()), "feedback_purged": fb}
    arts = ArtifactStore(s.artifact_dir, store, max_bytes=s.artifact_max_bytes, ttl_days=s.artifact_ttl_days)
    return {"artifacts_deleted": arts.cleanup_expired(), "feedback_purged": store.purge_feedback(s.feedback_retention_days)}


def _default_paths() -> tuple[str, str, str]:
    from app.agent.config import get_agent_settings
    from app.db import registry

    s = get_agent_settings()
    conv = os.getenv("INFERENCE_STORE_PATH") or str(registry.DB_PATH.with_name("conversations.db"))
    return conv, s.store_path, getattr(s, "checkpoint_path", "") or ""


def _days(name: str, default: int) -> int:
    v = int(os.getenv(name, str(default)))
    if v < 0:
        raise ValueError(f"{name} must be >= 0")
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="report what would be deleted; change nothing")
    ap.add_argument("--conversation-days", type=int, default=None,
                    help="override CONVERSATION_RETENTION_DAYS (default 90, 0 = keep forever)")
    ap.add_argument("--agent-state-days", type=int, default=None,
                    help="override AGENT_STATE_RETENTION_DAYS (default 30, 0 = keep forever)")
    args = ap.parse_args(argv)
    conv_days = args.conversation_days if args.conversation_days is not None else _days("CONVERSATION_RETENTION_DAYS", 90)
    agent_days = args.agent_state_days if args.agent_state_days is not None else _days("AGENT_STATE_RETENTION_DAYS", 30)
    if conv_days < 0 or agent_days < 0:
        ap.error("retention days must be >= 0")

    report: dict = {"dry_run": args.dry_run, "conversation_days": conv_days, "agent_state_days": agent_days}
    failed = False
    conv_path, store_path, ckpt_path = _default_paths()
    steps = (("artifacts_feedback", lambda: _artifacts_and_feedback(args.dry_run)),
             ("conversations", lambda: purge_conversations(conv_path, conv_days, args.dry_run)),
             ("agent_state", lambda: purge_agent_state(store_path, ckpt_path, agent_days, args.dry_run)))
    for name, step in steps:
        try:
            report[name] = step()
        except Exception as exc:  # keep going: one broken store must not stop the others' retention
            failed = True
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    raise SystemExit(main())
