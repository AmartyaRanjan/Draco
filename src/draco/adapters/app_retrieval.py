"""Run the existing `app.retrieval.RetrievalService` (and its generator) through the harness.

The service is not modified and stays usable on its own. The adapter only supplies identity
(`request_id` = execution id), budget/event instrumentation around the model call, typed errors,
and a JSON-friendly output.
"""
from __future__ import annotations

from typing import Any

from draco.adapters.errors import map_app_error
from draco.adapters.retrieval import DocumentResource, Evidence, authorize_evidence, authorize_request
from draco.core.context import ExecutionContext
from draco.core.errors import PolicyViolation
from draco.core.protocols import Workload
from draco.harness.instrument import call_model, emit


def _request(inp: Any) -> tuple[str, list[str] | None]:
    if isinstance(inp, str):
        return inp, None
    if isinstance(inp, dict) and isinstance(inp.get("query"), str):
        return inp["query"], inp.get("document_ids")
    from draco.core.errors import ContractError
    raise ContractError("input must be a query string or {'query': str, 'document_ids': [...]}")


def _items(outcome: Any) -> list[dict[str, Any]]:
    out = []
    for e in outcome.context.items:
        c = e.candidate
        out.append({"label": e.label, "chunk_id": c.chunk_id, "document_id": c.document_id, "filename": e.filename,
                    "page_start": c.page_start, "page_end": c.page_end, "score": c.colbert_score, "text": c.text})
    return out


async def _allowed_items(ctx: ExecutionContext, out: Any, document_resource: DocumentResource | None) -> list[dict[str, Any]]:
    """Items the caller may read; documents are checked after retrieval because the shared index may over-return."""
    items = _items(out)
    kept = await authorize_evidence(ctx, [Evidence(i["document_id"], "") for i in items], document_resource)
    ok = {e.document_id for e in kept}
    return [i for i in items if i["document_id"] in ok]


def retrieval_workload(service: Any, *, document_resource: DocumentResource | None = None) -> Workload:
    """query -> {items, evidence_sufficient, degraded, trace}. No model call is made.
    `document_resource` maps a document id to a `Resource` (owner tenant/ACL) so access is authorized
    before and after retrieval; with an identity-bearing execution and no resolver, document access is denied."""
    async def run(ctx: ExecutionContext, inp: Any) -> dict[str, Any]:
        query, doc_ids = _request(inp)
        ctx.check()
        await authorize_request(ctx, doc_ids, document_resource)
        try:
            out = await service.retrieve(query, doc_ids, request_id=ctx.execution_id, principal=ctx.user_id)
        except Exception as exc:
            mapped = map_app_error(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        items = await _allowed_items(ctx, out, document_resource)
        out.trace.finish("degraded" if out.degraded else "ok")
        await emit(ctx, "retrieval.after", selected=len(items), withheld=len(out.context.items) - len(items),
                   degraded=list(out.degraded), stages={k: v["ms"] for k, v in out.trace.stages.items()})
        sel = out.selection
        return {"query": out.query, "items": items, "degraded": list(out.degraded),
                "evidence_sufficient": bool(sel and sel.evidence_sufficient), "trace": out.trace.as_dict()}
    return run


def rag_workload(service: Any, *, document_resource: DocumentResource | None = None) -> Workload:
    """query -> grounded answer. Retrieval, then the service's own generator as a budgeted, evented
    model call. Abstains (without a model call) when nothing relevant was retrieved."""
    async def run(ctx: ExecutionContext, inp: Any) -> dict[str, Any]:
        query, doc_ids = _request(inp)
        ctx.check()
        await authorize_request(ctx, doc_ids, document_resource)
        try:
            out = await service.retrieve(query, doc_ids, request_id=ctx.execution_id, principal=ctx.user_id)
            if len(await _allowed_items(ctx, out, document_resource)) != len(out.context.items):
                raise PolicyViolation("retrieved evidence includes documents the caller may not read")  # before any generation
            base = {"query": out.query, "degraded": list(out.degraded)}
            if not out.context.items:
                out.trace.set(abstained="no_evidence")
                out.trace.finish("ok")
                return {**base, "answer": None, "abstained": "no_evidence", "citations": [], "trace": out.trace.as_dict()}
            if service.generator is None:
                out.trace.finish("ok")
                return {**base, "answer": None, "abstained": "generator_not_configured", "citations": [],
                        "items": _items(out), "trace": out.trace.as_dict()}
            weak = not (out.selection and out.selection.evidence_sufficient)
            gen = await call_model(ctx, service.generator.generate, out.query, out.context, weak,
                                   provider="app.retrieval.generator", model=getattr(service.s, "llm_model", None) or "",
                                   usage_of=lambda g: (g.usage or {}).get("total_tokens"))
        except Exception as exc:
            mapped = map_app_error(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        out.trace.set(**gen.usage, invalid_citations=gen.invalid_markers)
        out.trace.finish("degraded" if out.degraded else "ok")
        return {**base, "answer": gen.answer, "abstained": None, "citations": gen.citations,
                "invalid_markers": gen.invalid_markers, "evidence_sufficient": not weak, "trace": out.trace.as_dict()}
    return run
