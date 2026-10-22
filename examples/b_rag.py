"""Example B: the existing RetrievalService + generator through the harness: ids, tracing, policy,
failure handling and budgets. Offline fakes replace Qdrant/OpenSearch/the model server."""
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.db import registry

from draco import BudgetLimits, Harness, Policy
from draco.adapters import rag_workload
from examples._fakes import retrieval_service


def show(title, r):
    err = f" error={r.error.code}(retryable={r.error.retryable})" if r.error else ""
    out = r.output or {}
    print(f"{title:<34} status={r.status.value:<17}{err} abstained={out.get('abstained')} degraded={out.get('degraded')}")


async def main():
    with tempfile.TemporaryDirectory() as tmp, patch.object(registry, "DB_PATH", Path(tmp) / "documents.db"):
        registry.init_db()
        registry.create_document("handbook", "handbook.pdf", "/handbook.pdf", "alice")
        h = Harness()
        h.register("rag", rag_workload(retrieval_service()))
        r = await h.run("rag", "How are tools protected?", user_id="alice", session_id="s1")
        show("1. normal", r)
        print("   answer:", r.output["answer"])
        print("   citation:", {k: r.output["citations"][0][k] for k in ("filename", "page_start", "chunk_id")})
        print("   execution_id == retrieval request_id:", r.execution_id == r.output["trace"]["request_id"])
        print("   usage:", r.usage)
        print("   trace:", [e["event"] for e in h.tracer.trace(r.execution_id)])

        show("2. policy denies workload",
             await h.run("rag", "q", user_id="alice", policy=Policy(allowed_agents=frozenset({"other"}))))
        show("3. token budget exhausted", await h.run("rag", "q", user_id="alice", limits=BudgetLimits(max_tokens=10)))
        show("4. empty query (contract error)", await h.run("rag", "", user_id="alice"))

        h2 = Harness()
        h2.register("rag", rag_workload(retrieval_service(lexical_down=True)))
        show("5. BM25 down -> degraded, ok", await h2.run("rag", "q", user_id="alice"))

asyncio.run(main())
