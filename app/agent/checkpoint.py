"""SQLite-backed LangGraph checkpoints for the standalone application."""

from __future__ import annotations

import asyncio
import random
import sqlite3
import threading
import time
import zlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)


class CheckpointCorruptError(RuntimeError):
    """A persisted checkpoint failed its integrity check."""


def _crc(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


class SqliteCheckpointSaver(BaseCheckpointSaver):
    """Persist LangGraph state in SQLite; one short-lived connection per operation."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS graph_checkpoints (
      thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL, checkpoint_id TEXT NOT NULL,
      parent_checkpoint_id TEXT, created_at REAL NOT NULL, expires_at REAL,
      checkpoint_type TEXT NOT NULL, checkpoint_bytes BLOB NOT NULL, checkpoint_crc INTEGER NOT NULL,
      metadata_type TEXT NOT NULL, metadata_bytes BLOB NOT NULL, metadata_crc INTEGER NOT NULL,
      PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id));
    CREATE INDEX IF NOT EXISTS idx_graph_checkpoints_thread
      ON graph_checkpoints(thread_id, checkpoint_ns, checkpoint_id);
    CREATE TABLE IF NOT EXISTS graph_channel_blobs (
      thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL, channel TEXT NOT NULL, version TEXT NOT NULL,
      value_type TEXT NOT NULL, value_bytes BLOB NOT NULL, value_crc INTEGER NOT NULL,
      PRIMARY KEY (thread_id, checkpoint_ns, channel, version));
    CREATE TABLE IF NOT EXISTS graph_task_writes (
      thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL, checkpoint_id TEXT NOT NULL,
      task_id TEXT NOT NULL, idx INTEGER NOT NULL, channel TEXT NOT NULL, task_path TEXT NOT NULL,
      value_type TEXT NOT NULL, value_bytes BLOB NOT NULL, value_crc INTEGER NOT NULL,
      PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx));
    CREATE TABLE IF NOT EXISTS graph_checkpoint_channels (
      thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL, checkpoint_id TEXT NOT NULL,
      channel TEXT NOT NULL, version TEXT NOT NULL,
      PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, channel, version));
    CREATE INDEX IF NOT EXISTS idx_graph_checkpoint_channels_blob
      ON graph_checkpoint_channels(thread_id, checkpoint_ns, channel, version);
    """

    def __init__(self, path: str | Path, ttl_seconds: int | None = None):
        super().__init__()
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError(
                "a durable checkpoint saver requires a file path; use InMemorySaver for ephemeral state"
            )
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connection()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(self._SCHEMA)
            with conn:
                self._backfill_checkpoint_channels(conn)
                self._cleanup_expired(conn, time.time())
        finally:
            conn.close()

    def _connection(self):
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _backfill_checkpoint_channels(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT thread_id, checkpoint_ns, checkpoint_id, checkpoint_type, checkpoint_bytes, "
            "checkpoint_crc, expires_at FROM graph_checkpoints"
        ).fetchall()
        now = time.time()
        for row in rows:
            if row["expires_at"] is not None and row["expires_at"] <= now:
                continue
            data = row["checkpoint_bytes"]
            if _crc(data) != row["checkpoint_crc"]:
                continue
            checkpoint = self.serde.loads_typed((row["checkpoint_type"], data))
            versions = checkpoint.get("channel_versions") or {}
            conn.executemany(
                "INSERT OR IGNORE INTO graph_checkpoint_channels "
                "(thread_id,checkpoint_ns,checkpoint_id,channel,version) VALUES (?,?,?,?,?)",
                [
                    (row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"], channel, version)
                    for channel, version in versions.items()
                ],
            )

    @staticmethod
    def _cleanup_expired(conn: sqlite3.Connection, now: float) -> None:
        expired = "cp.expires_at IS NOT NULL AND cp.expires_at<=?"
        conn.execute(
            "DELETE FROM graph_task_writes WHERE EXISTS ("
            "SELECT 1 FROM graph_checkpoints cp WHERE "
            "cp.thread_id=graph_task_writes.thread_id AND cp.checkpoint_ns=graph_task_writes.checkpoint_ns "
            "AND cp.checkpoint_id=graph_task_writes.checkpoint_id AND " + expired + ")",
            (now,),
        )
        conn.execute(
            "DELETE FROM graph_checkpoint_channels WHERE EXISTS ("
            "SELECT 1 FROM graph_checkpoints cp WHERE "
            "cp.thread_id=graph_checkpoint_channels.thread_id "
            "AND cp.checkpoint_ns=graph_checkpoint_channels.checkpoint_ns "
            "AND cp.checkpoint_id=graph_checkpoint_channels.checkpoint_id AND " + expired + ")",
            (now,),
        )
        conn.execute("DELETE FROM graph_checkpoints WHERE expires_at IS NOT NULL AND expires_at<=?", (now,))
        conn.execute(
            "DELETE FROM graph_channel_blobs WHERE NOT EXISTS ("
            "SELECT 1 FROM graph_checkpoint_channels refs WHERE "
            "refs.thread_id=graph_channel_blobs.thread_id "
            "AND refs.checkpoint_ns=graph_channel_blobs.checkpoint_ns "
            "AND refs.channel=graph_channel_blobs.channel AND refs.version=graph_channel_blobs.version)"
        )

    def _load_tuple(self, conn: sqlite3.Connection, row: sqlite3.Row) -> CheckpointTuple:
        checkpoint_bytes = row["checkpoint_bytes"]
        metadata_bytes = row["metadata_bytes"]
        if _crc(checkpoint_bytes) != row["checkpoint_crc"] or _crc(metadata_bytes) != row["metadata_crc"]:
            raise CheckpointCorruptError(f"checkpoint {row['checkpoint_id']} failed its integrity check")

        checkpoint: Checkpoint = self.serde.loads_typed((row["checkpoint_type"], checkpoint_bytes))
        metadata: CheckpointMetadata = self.serde.loads_typed((row["metadata_type"], metadata_bytes))
        channel_values: dict[str, Any] = {}
        for channel, version in (checkpoint.get("channel_versions") or {}).items():
            blob = conn.execute(
                "SELECT value_type, value_bytes, value_crc FROM graph_channel_blobs "
                "WHERE thread_id=? AND checkpoint_ns=? AND channel=? AND version=?",
                (row["thread_id"], row["checkpoint_ns"], channel, version),
            ).fetchone()
            if blob is None:
                raise CheckpointCorruptError(
                    f"checkpoint {row['checkpoint_id']} is missing channel {channel!r}"
                )
            if _crc(blob["value_bytes"]) != blob["value_crc"]:
                raise CheckpointCorruptError(
                    f"checkpoint {row['checkpoint_id']} channel {channel!r} failed its integrity check"
                )
            if blob["value_type"] != "empty":
                channel_values[channel] = self.serde.loads_typed((blob["value_type"], blob["value_bytes"]))

        writes = conn.execute(
            "SELECT task_id, idx, channel, task_path, value_type, value_bytes, value_crc "
            "FROM graph_task_writes WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
            (row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"]),
        ).fetchall()
        pending_writes = []
        for write in sorted(writes, key=lambda item: (item["task_path"], item["task_id"], item["idx"])):
            if _crc(write["value_bytes"]) != write["value_crc"]:
                raise CheckpointCorruptError(
                    f"checkpoint {row['checkpoint_id']} task write failed its integrity check"
                )
            pending_writes.append(
                (
                    write["task_id"],
                    write["channel"],
                    self.serde.loads_typed((write["value_type"], write["value_bytes"])),
                )
            )

        config = {
            "configurable": {
                "thread_id": row["thread_id"],
                "checkpoint_ns": row["checkpoint_ns"],
                "checkpoint_id": row["checkpoint_id"],
            }
        }
        parent_config = None
        if row["parent_checkpoint_id"]:
            parent_config = {
                "configurable": {
                    "thread_id": row["thread_id"],
                    "checkpoint_ns": row["checkpoint_ns"],
                    "checkpoint_id": row["parent_checkpoint_id"],
                }
            }
        return CheckpointTuple(
            config=config,
            checkpoint={**checkpoint, "channel_values": channel_values},
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        with self._lock:
            conn = self._connection()
            try:
                with conn:
                    self._cleanup_expired(conn, time.time())
                if checkpoint_id:
                    row = conn.execute(
                        "SELECT * FROM graph_checkpoints WHERE thread_id=? AND checkpoint_ns=? "
                        "AND checkpoint_id=? AND (expires_at IS NULL OR expires_at>?)",
                        (thread_id, checkpoint_ns, checkpoint_id, time.time()),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT * FROM graph_checkpoints WHERE thread_id=? AND checkpoint_ns=? "
                        "AND (expires_at IS NULL OR expires_at>?) ORDER BY checkpoint_id DESC LIMIT 1",
                        (thread_id, checkpoint_ns, time.time()),
                    ).fetchone()
                return self._load_tuple(conn, row) if row is not None else None
            finally:
                conn.close()

    def list(
        self,
        config: dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if limit is not None and limit <= 0:
            return iter(())
        configurable = (config or {}).get("configurable", {})
        before_id = get_checkpoint_id(before) if before else None
        clauses = ["(expires_at IS NULL OR expires_at>?)"]
        params: list[Any] = [time.time()]
        if configurable.get("thread_id") is not None:
            clauses.append("thread_id=?")
            params.append(configurable["thread_id"])
        if configurable.get("checkpoint_ns") is not None:
            clauses.append("checkpoint_ns=?")
            params.append(configurable["checkpoint_ns"])
        checkpoint_id = get_checkpoint_id(config) if config else None
        if checkpoint_id:
            clauses.append("checkpoint_id=?")
            params.append(checkpoint_id)
        if before_id:
            clauses.append("checkpoint_id<?")
            params.append(before_id)
        query = "SELECT * FROM graph_checkpoints WHERE " + " AND ".join(clauses)
        query += " ORDER BY checkpoint_id DESC"
        if limit is not None and not filter:
            query += " LIMIT ?"
            params.append(limit)

        with self._lock:
            conn = self._connection()
            try:
                with conn:
                    self._cleanup_expired(conn, time.time())
                rows = conn.execute(query, params).fetchall()
                result = []
                for row in rows:
                    checkpoint_tuple = self._load_tuple(conn, row)
                    if filter and not all(
                        checkpoint_tuple.metadata.get(key) == value for key, value in filter.items()
                    ):
                        continue
                    result.append(checkpoint_tuple)
                    if limit is not None and len(result) >= limit:
                        break
                return iter(result)
            finally:
                conn.close()

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict[str, Any]:
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_without_values = checkpoint.copy()
        values = checkpoint_without_values.pop("channel_values")
        checkpoint_type, checkpoint_bytes = self.serde.dumps_typed(checkpoint_without_values)
        metadata_type, metadata_bytes = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        now = time.time()
        expires_at = now + self.ttl_seconds if self.ttl_seconds is not None else None

        with self._lock:
            conn = self._connection()
            try:
                with conn:
                    self._cleanup_expired(conn, now)
                    for channel, version in new_versions.items():
                        value = values.get(channel)
                        value_type, value_bytes = (
                            ("empty", b"") if channel not in values else self.serde.dumps_typed(value)
                        )
                        conn.execute(
                            "INSERT OR REPLACE INTO graph_channel_blobs "
                            "(thread_id,checkpoint_ns,channel,version,value_type,value_bytes,value_crc) "
                            "VALUES (?,?,?,?,?,?,?)",
                            (
                                thread_id,
                                checkpoint_ns,
                                channel,
                                version,
                                value_type,
                                value_bytes,
                                _crc(value_bytes),
                            ),
                        )
                    conn.execute(
                        "INSERT OR REPLACE INTO graph_checkpoints "
                        "(thread_id,checkpoint_ns,checkpoint_id,parent_checkpoint_id,created_at,expires_at,"
                        "checkpoint_type,checkpoint_bytes,checkpoint_crc,metadata_type,metadata_bytes,metadata_crc) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint["id"],
                            configurable.get("checkpoint_id"),
                            now,
                            expires_at,
                            checkpoint_type,
                            checkpoint_bytes,
                            _crc(checkpoint_bytes),
                            metadata_type,
                            metadata_bytes,
                            _crc(metadata_bytes),
                        ),
                    )
                    conn.execute(
                        "DELETE FROM graph_checkpoint_channels WHERE thread_id=? AND checkpoint_ns=? "
                        "AND checkpoint_id=?",
                        (thread_id, checkpoint_ns, checkpoint["id"]),
                    )
                    conn.executemany(
                        "INSERT INTO graph_checkpoint_channels "
                        "(thread_id,checkpoint_ns,checkpoint_id,channel,version) VALUES (?,?,?,?,?)",
                        [
                            (thread_id, checkpoint_ns, checkpoint["id"], channel, version)
                            for channel, version in (checkpoint.get("channel_versions") or {}).items()
                        ],
                    )
            finally:
                conn.close()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable["checkpoint_id"]
        rows = []
        for index, (channel, value) in enumerate(writes):
            write_index = WRITES_IDX_MAP.get(channel, index)
            value_type, value_bytes = self.serde.dumps_typed(value)
            rows.append(
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    write_index,
                    channel,
                    task_path,
                    value_type,
                    value_bytes,
                    _crc(value_bytes),
                )
            )
        if not rows:
            return
        with self._lock:
            conn = self._connection()
            try:
                with conn:
                    self._cleanup_expired(conn, time.time())
                    for row in rows:
                        if row[4] >= 0:
                            conn.execute(
                                "INSERT OR IGNORE INTO graph_task_writes "
                                "(thread_id,checkpoint_ns,checkpoint_id,task_id,idx,channel,task_path,"
                                "value_type,value_bytes,value_crc) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                row,
                            )
                        else:
                            conn.execute(
                                "INSERT OR REPLACE INTO graph_task_writes "
                                "(thread_id,checkpoint_ns,checkpoint_id,task_id,idx,channel,task_path,"
                                "value_type,value_bytes,value_crc) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                row,
                            )
            finally:
                conn.close()

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            conn = self._connection()
            try:
                with conn:
                    conn.execute("DELETE FROM graph_task_writes WHERE thread_id=?", (thread_id,))
                    conn.execute("DELETE FROM graph_checkpoint_channels WHERE thread_id=?", (thread_id,))
                    conn.execute("DELETE FROM graph_channel_blobs WHERE thread_id=?", (thread_id,))
                    conn.execute("DELETE FROM graph_checkpoints WHERE thread_id=?", (thread_id,))
            finally:
                conn.close()

    def get_next_version(self, current: str | int | None, channel: str) -> str:
        del channel
        if current is None:
            current_version = 0
        elif isinstance(current, int):
            current_version = current
        else:
            current_version = int(str(current).split(".")[0])
        return f"{current_version + 1:032}.{random.random():016}"

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        rows = await asyncio.to_thread(self.list, config, filter=filter, before=before, limit=limit)
        for row in rows:
            yield row

    async def adelete_thread(self, thread_id):
        await asyncio.to_thread(self.delete_thread, thread_id)
