"""Generic ingestion workload: the harness runs whatever callable the application injects. Parsing,
chunking, embedding and storage stay in that callable; nothing here knows about them."""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from draco.core.context import ExecutionContext
from draco.core.errors import ContractError
from draco.core.protocols import Workload
from draco.security.audit import authorize
from draco.security.authorization import Resource


def ingest_workload(ingest: Callable[..., Any], *, resource_for: Callable[[str], Resource] | None = None) -> Workload:
    """input {'document_id': str, 'file_path': str}. `ingest(document_id, file_path)` may be sync (run in a
    thread) or async. With `resource_for`, `document.ingest` is authorized on the target document before the
    callable runs. Re-running is only safe if `ingest` is idempotent (e.g. deterministic chunk ids)."""
    async def run(ctx: ExecutionContext, inp: Any) -> dict[str, Any]:
        if not isinstance(inp, dict) or not {"document_id", "file_path"} <= inp.keys():
            raise ContractError("input must contain document_id and file_path")
        ctx.check()
        doc = str(inp["document_id"])
        if resource_for is not None:
            await authorize(ctx, "document.ingest", resource_for(doc))
        elif ctx.identity is not None:
            await authorize(ctx, "document.ingest", Resource("document", doc, tenant_id=ctx.tenant_id))
        if inspect.iscoroutinefunction(ingest):
            await ingest(doc, inp["file_path"])
        else:
            await asyncio.to_thread(ingest, doc, inp["file_path"])
        return {"document_id": doc, "status": "processed"}
    return run
