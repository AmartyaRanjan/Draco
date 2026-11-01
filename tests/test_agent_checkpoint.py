import asyncio
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agent.checkpoint import CheckpointCorruptError, SqliteCheckpointSaver


def save_checkpoint(saver, thread_id, checkpoint_id, parent_id=None):
    checkpoint = empty_checkpoint()
    checkpoint["id"] = checkpoint_id
    checkpoint["channel_values"] = {"answer": checkpoint_id}
    checkpoint["channel_versions"] = {"answer": checkpoint_id}
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "", "checkpoint_id": parent_id}}
    return saver.put(config, checkpoint, {"source": "test"}, {"answer": checkpoint_id})


class SqliteCheckpointSaverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "checkpoints.db"
        self.saver = SqliteCheckpointSaver(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_pending_writes_list_and_delete(self):
        config = save_checkpoint(self.saver, "thread-1", "cp-1")
        self.saver.put_writes(config, [("answer", "pending")], "task-1")

        restored = self.saver.get_tuple(config)
        self.assertEqual(restored.checkpoint["channel_values"], {"answer": "cp-1"})
        self.assertEqual(restored.pending_writes, [("task-1", "answer", "pending")])
        self.assertEqual(len(list(self.saver.list({"configurable": {"thread_id": "thread-1"}}))), 1)

        self.saver.delete_thread("thread-1")
        self.assertIsNone(self.saver.get_tuple(config))

    def test_expired_checkpoint_is_not_returned(self):
        config = save_checkpoint(self.saver, "thread-expired", "cp-1")
        self.saver.put_writes(config, [("answer", "pending")], "task-1")
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("UPDATE graph_checkpoints SET expires_at=0 WHERE thread_id=?", ("thread-expired",))
        self.assertIsNone(self.saver.get_tuple(config))
        self.assertEqual(list(self.saver.list({"configurable": {"thread_id": "thread-expired"}})), [])
        with closing(sqlite3.connect(self.path)) as conn:
            for table in (
                "graph_checkpoints",
                "graph_task_writes",
                "graph_checkpoint_channels",
                "graph_channel_blobs",
            ):
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)

    def test_expiry_cleanup_preserves_blobs_referenced_by_live_checkpoints(self):
        expired_config = save_checkpoint(self.saver, "shared-blob", "cp-1")
        checkpoint = empty_checkpoint()
        checkpoint["id"] = "cp-2"
        checkpoint["channel_versions"] = {"answer": "cp-1"}
        live_config = self.saver.put(
            {"configurable": {"thread_id": "shared-blob", "checkpoint_id": "cp-1"}},
            checkpoint,
            {},
            {},
        )
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("UPDATE graph_checkpoints SET expires_at=0 WHERE checkpoint_id='cp-1'")

        self.assertIsNone(self.saver.get_tuple(expired_config))
        restored = self.saver.get_tuple(live_config)
        self.assertEqual(restored.checkpoint["channel_values"], {"answer": "cp-1"})
        with closing(sqlite3.connect(self.path)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM graph_channel_blobs").fetchone()[0], 1)

    def test_corrupted_checkpoint_is_reported(self):
        config = save_checkpoint(self.saver, "thread-corrupt", "cp-1")
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                "UPDATE graph_checkpoints SET checkpoint_bytes=? WHERE thread_id=?",
                (b"corrupt", "thread-corrupt"),
            )
        with self.assertRaises(CheckpointCorruptError):
            self.saver.get_tuple(config)

    def test_async_methods_and_concurrent_savers(self):
        async def exercise_async():
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {"value": 1}
            checkpoint["channel_versions"] = {"value": "1"}
            config = await self.saver.aput(
                {"configurable": {"thread_id": "async-thread", "checkpoint_ns": ""}},
                checkpoint,
                {},
                {"value": "1"},
            )
            await self.saver.aput_writes(config, [("value", 2)], "task")
            restored = await self.saver.aget_tuple(config)
            listed = [
                entry async for entry in self.saver.alist({"configurable": {"thread_id": "async-thread"}})
            ]
            self.assertEqual(restored.pending_writes, [("task", "value", 2)])
            self.assertEqual(len(listed), 1)
            await self.saver.adelete_thread("async-thread")

        asyncio.run(exercise_async())

        savers = [SqliteCheckpointSaver(self.path) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(
                pool.map(
                    lambda item: save_checkpoint(*item),
                    zip(savers, ["shared"] * 4, [f"cp-{i}" for i in range(4)]),
                )
            )
        self.assertEqual(len(list(self.saver.list({"configurable": {"thread_id": "shared"}}))), 4)

    def test_interrupt_resumes_from_a_separate_process(self):
        async def gate(state):
            decision = interrupt({"type": "approval_required"})
            if decision["decision"] == "approve":
                return {"approved": True}
            return {"rejected": True}

        def build():
            graph = StateGraph(dict)
            graph.add_node("approval", gate)
            graph.add_edge(START, "approval")
            graph.add_edge("approval", END)
            return graph.compile(checkpointer=SqliteCheckpointSaver(self.path))

        config = {"configurable": {"thread_id": "resume-thread"}}
        paused = asyncio.run(build().ainvoke({"approved": False}, config=config))
        self.assertTrue(paused["__interrupt__"])

        code = """
import asyncio, json, sys
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from app.agent.checkpoint import SqliteCheckpointSaver

async def gate(state):
    decision = interrupt({"type": "approval_required"})
    return {"approved": decision["decision"] == "approve"}

graph_builder = StateGraph(dict)
graph_builder.add_node("approval", gate)
graph_builder.add_edge(START, "approval")
graph_builder.add_edge("approval", END)
graph = graph_builder.compile(checkpointer=SqliteCheckpointSaver(sys.argv[1]))
result = asyncio.run(graph.ainvoke(
    Command(resume={"decision": "approve"}),
    config={"configurable": {"thread_id": "resume-thread"}},
))
print(json.dumps(result))
"""
        child = subprocess.run(
            [sys.executable, "-c", code, str(self.path)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        resumed = json.loads(child.stdout)
        self.assertTrue(resumed["approved"])
        self.assertNotIn("__interrupt__", resumed)


if __name__ == "__main__":
    unittest.main()
