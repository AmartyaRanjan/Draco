"""Expose the existing ingestion worker as a harness workload (no rewrite)."""
from __future__ import annotations

import asyncio
from typing import Any

from draco.core.context import ExecutionContext
from draco.core.errors import ContractError
from draco.core.protocols import Workload


def ingestion_workload(process_document: Any = None) -> Workload:
    """input: {'document_id': str, 'file_path': str}. Runs the (blocking) worker off the event loop.
    Ingestion upserts deterministic chunk ids, so re-running a failed document is safe; wrap larger
    flows in `draco.harness.step` if they add non-idempotent side effects."""
    async def run(ctx: ExecutionContext, inp: Any) -> dict[str, Any]:
        fn = process_document
        if fn is None:
            from app.ingestion.worker import process_document as fn  # heavy import, only when used
        if not isinstance(inp, dict) or not {"document_id", "file_path"} <= inp.keys():
            raise ContractError("input must contain document_id and file_path")
        ctx.check()
        await asyncio.to_thread(fn, inp["document_id"], inp["file_path"])
        return {"document_id": inp["document_id"], "status": "processed"}
    return run
