import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from app.agent.graph import AgentRunner, AgentState
from app.agent.schemas import AgentChatRequest
from app.inference.schemas import InferenceError
from langgraph.graph import END, START, StateGraph

from tests.agent_helpers import make_runner


class AgentResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.checkpoint_path = str(Path(self.tmp.name) / "checkpoints.db")
        self.runner, self.services = make_runner([])
        self.services.settings = replace(self.services.settings, checkpoint_path=self.checkpoint_path)
        self.source = self.services.store.create_source(
            user_id="alice",
            url="https://example.org/source",
            normalized_url="https://example.org/source",
            url_hash="url-hash",
            title="Source",
            content_hash="content-hash",
            content_text="approved text",
            reason="requested by the user",
            preview={"title": "Source", "domain": "example.org"},
            trust={"level": "medium"},
            ttl_hours=1,
        )
        self.protected_calls = []

        async def execute(name, args, ctx, *, purpose="", sink=None):
            self.protected_calls.append(
                (name, args, self.services.store.get_approval(args["source_id"], ctx.user_id))
            )
            return SimpleNamespace(
                status="success",
                data={"summary": "submitted"},
                error=None,
                metadata=SimpleNamespace(extra={}),
            )

        self.runner.reg.execute = execute
        self.runner._build = self.build_graph

    def tearDown(self):
        self.tmp.cleanup()

    def build_graph(self, run):
        if not run.pending:
            run.pending = [
                {
                    "source_id": self.source["source_id"],
                    "title": "Source",
                    "url": "https://example.org/source",
                }
            ]
        graph = StateGraph(AgentState)

        async def gate(state):
            return await self.runner.n_approval_gate(run, state)

        async def finish(state):
            return {"answer": "finished", "status": "ok"}

        graph.add_node("approval_gate", gate)
        graph.add_node("finish", finish)
        graph.add_edge(START, "approval_gate")
        graph.add_edge("approval_gate", "finish")
        graph.add_edge("finish", END)
        from app.agent.checkpoint import SqliteCheckpointSaver

        return graph.compile(checkpointer=SqliteCheckpointSaver(self.checkpoint_path, ttl_seconds=3600))

    def test_invalid_invalid_valid_resumes_only_after_decision_validation(self):
        async def exercise():
            response = await self.runner.run(
                AgentChatRequest(message="save this source"), "alice", tenant_id="tenant-a"
            )
            execution_id = response.metadata["request_id"]
            self.assertEqual(response.status, "awaiting_approval")
            record = await self.runner.get_execution(execution_id, "alice", "tenant-a")
            self.assertEqual(record["status"], "waiting")
            self.assertEqual((record["user_id"], record["tenant_id"]), ("alice", "tenant-a"))
            self.assertEqual(
                self.services.store.get_source(self.source["source_id"])["status"], "pending_approval"
            )

            for invalid in ("approve ", "invalid"):
                with self.assertRaises(InferenceError) as raised:
                    await self.runner.resume(execution_id, invalid, "alice", tenant_id="tenant-a")
                self.assertEqual(raised.exception.status_code, 422)
                record = await self.runner.get_execution(execution_id, "alice", "tenant-a")
                self.assertEqual(record["status"], "waiting")
                self.assertEqual(
                    self.services.store.get_source(self.source["source_id"])["status"], "pending_approval"
                )
                self.assertEqual(self.protected_calls, [])
                restored = self.runner._build(
                    self.runner._create_run_for_resume(
                        "alice",
                        execution_id,
                        "tenant-a",
                        AgentChatRequest.model_validate(record["request"]),
                        record["response"]["pending_approvals"],
                    )
                )
                snapshot = await restored.aget_state({"configurable": {"thread_id": execution_id}})
                self.assertIn("approval_gate", snapshot.next)
                self.assertTrue(any(task.interrupts for task in snapshot.tasks))

            resumed = await self.runner.resume(execution_id, "approve", "alice", tenant_id="tenant-a")
            self.assertEqual(resumed.metadata["approval_decision"], "approve", resumed.model_dump_json())
            self.assertEqual(resumed.status, "ok")
            self.assertIn("Approval recorded", resumed.answer)
            self.assertEqual(self.services.store.get_source(self.source["source_id"])["status"], "approved")
            self.assertEqual(len(self.protected_calls), 1)
            self.assertEqual(self.protected_calls[0][0], "ingest_approved_source")
            self.assertEqual(
                (await self.runner.get_execution(execution_id, "alice", "tenant-a"))["status"],
                "completed",
            )

            with self.assertRaises(InferenceError) as raised:
                await self.runner.resume(execution_id, "approve", "alice", tenant_id="tenant-a")
            self.assertEqual(raised.exception.status_code, 409)
            self.assertEqual(len(self.protected_calls), 1)

        asyncio.run(exercise())

    def test_rejection_persists_decision_without_running_protected_operation(self):
        async def exercise():
            response = await self.runner.run(
                AgentChatRequest(message="save this source"), "alice", tenant_id="tenant-a"
            )
            resumed = await self.runner.resume(
                response.metadata["request_id"], "reject", "alice", tenant_id="tenant-a"
            )
            self.assertEqual(resumed.metadata["approval_decision"], "reject", resumed.model_dump_json())
            self.assertIn("Approval rejected", resumed.answer)
            self.assertEqual(self.services.store.get_source(self.source["source_id"])["status"], "rejected")
            self.assertEqual(self.protected_calls, [])

        asyncio.run(exercise())

    def test_interrupted_checkpoint_recovers_if_execution_status_write_was_lost(self):
        async def exercise():
            response = await self.runner.run(
                AgentChatRequest(message="save this source"), "alice", tenant_id="tenant-a"
            )
            execution_id = response.metadata["request_id"]
            self.services.store._write(
                "UPDATE agent_executions SET status='running', response_json=NULL WHERE execution_id=?",
                (execution_id,),
            )

            resumed = await self.runner.resume(execution_id, "approve", "alice", tenant_id="tenant-a")
            self.assertEqual(resumed.metadata["approval_decision"], "approve")
            self.assertEqual(len(self.protected_calls), 1)
            self.assertEqual(
                (await self.runner.get_execution(execution_id, "alice", "tenant-a"))["status"],
                "completed",
            )

        asyncio.run(exercise())

    def test_stale_resume_claim_recovers_from_persisted_interrupt(self):
        async def exercise():
            response = await self.runner.run(
                AgentChatRequest(message="save this source"), "alice", tenant_id="tenant-a"
            )
            execution_id = response.metadata["request_id"]
            _, claimed = self.services.store.claim_execution_resume(execution_id, "alice", "tenant-a")
            self.assertTrue(claimed)
            self.services.store._write(
                "UPDATE agent_executions SET updated_at=0 WHERE execution_id=?", (execution_id,)
            )

            resumed = await self.runner.resume(execution_id, "approve", "alice", tenant_id="tenant-a")
            self.assertEqual(resumed.metadata["approval_decision"], "approve")
            self.assertEqual(len(self.protected_calls), 1)
            self.assertEqual(
                (await self.runner.get_execution(execution_id, "alice", "tenant-a"))["status"],
                "completed",
            )

        asyncio.run(exercise())

    def test_execution_is_hidden_from_other_user_and_tenant(self):
        async def exercise():
            response = await self.runner.run(
                AgentChatRequest(message="save this source"), "alice", tenant_id="tenant-a"
            )
            execution_id = response.metadata["request_id"]
            for user, tenant in (("bob", "tenant-a"), ("alice", "tenant-b")):
                with self.assertRaises(InferenceError) as raised:
                    await self.runner.get_execution(execution_id, user, tenant)
                self.assertEqual(raised.exception.status_code, 404)
                with self.assertRaises(InferenceError) as raised:
                    await self.runner.resume(execution_id, "approve", user, tenant_id=tenant)
                self.assertEqual(raised.exception.status_code, 404)

        asyncio.run(exercise())

    def test_missing_expired_and_corrupt_checkpoints_fail_closed(self):
        async def exercise():
            for kind in ("missing", "expired", "corrupt"):
                response = await self.runner.run(
                    AgentChatRequest(message=f"save source after {kind} checkpoint"),
                    "alice",
                    tenant_id="tenant-a",
                )
                execution_id = response.metadata["request_id"]
                with closing(sqlite3.connect(self.checkpoint_path)) as conn, conn:
                    if kind == "missing":
                        conn.execute("DELETE FROM graph_checkpoints WHERE thread_id=?", (execution_id,))
                    elif kind == "expired":
                        conn.execute(
                            "UPDATE graph_checkpoints SET expires_at=0 WHERE thread_id=?",
                            (execution_id,),
                        )
                    else:
                        conn.execute(
                            "UPDATE graph_checkpoints SET checkpoint_bytes=? WHERE thread_id=?",
                            (b"corrupt", execution_id),
                        )

                with self.assertRaises(InferenceError) as raised:
                    await self.runner.resume(execution_id, "approve", "alice", tenant_id="tenant-a")
                self.assertIn(raised.exception.status_code, (409, 502))
                record = await self.runner.get_execution(execution_id, "alice", "tenant-a")
                self.assertEqual(record["status"], "failed")
                self.assertEqual(
                    self.services.store.get_source(self.source["source_id"])["status"], "pending_approval"
                )
                self.assertEqual(self.protected_calls, [])

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
