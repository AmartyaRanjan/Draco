"""Provider-neutral retrieval contract with authorization applied before and after the backend call.

Retrieval is optional: nothing in the core imports this module. A backend implements `Retriever`; the
`app` RetrievalService is adapted in `app_retrieval`."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from draco.core.context import ExecutionContext
from draco.core.errors import ContractError, PolicyViolation
from draco.core.protocols import Workload
from draco.harness.instrument import emit
from draco.security.audit import audit, authorize
from draco.security.authorization import Decision, Resource, decide


@dataclass
class Evidence:
    document_id: str
    text: str
    chunk_id: str = ""
    score: float | None = None
    source: str = ""
    locator: dict[str, Any] = field(default_factory=dict)  # page range, offsets ...


@dataclass
class RetrievalResult:
    query: str
    evidence: list[Evidence]
    degraded: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, ctx: ExecutionContext, query: str, document_ids: list[str] | None) -> RetrievalResult: ...


DocumentResource = Callable[[str], Resource]  # document id -> Resource carrying the owning tenant/ACL attributes


async def authorize_request(ctx: ExecutionContext, document_ids: list[str] | None,
                            document_resource: DocumentResource | None) -> None:
    """Before the backend runs. Fails closed in identity-aware (multi-tenant) mode: if documents are
    requested but the application gave no way to resolve their owners, the request is denied."""
    await authorize(ctx, "retrieval.query", Resource("retrieval", tenant_id=ctx.tenant_id))
    for doc in document_ids or []:
        if document_resource is None:
            if ctx.identity is not None:
                raise PolicyViolation("document access cannot be verified: no document_resource configured")
            continue
        await authorize(ctx, "document.read", document_resource(doc))


async def authorize_evidence(ctx: ExecutionContext, evidence: list[Evidence],
                             document_resource: DocumentResource | None) -> list[Evidence]:
    """After the backend ran: drop (and audit) evidence the caller may not read. Backends may over-return
    (shared index), so the result is never trusted to be pre-filtered."""
    if document_resource is None:
        if ctx.identity is not None and evidence:
            raise PolicyViolation("document access cannot be verified: no document_resource configured")
        return evidence
    cache: dict[str, Decision] = {}
    kept = []
    for e in evidence:
        res = document_resource(e.document_id)
        if e.document_id not in cache:
            authorizer = ctx.services.get("authorizer")
            cache[e.document_id] = await decide(authorizer, ctx.identity, "document.read", res)
            await audit(ctx, "document.read", res, cache[e.document_id], stage="post_retrieval")
        if cache[e.document_id].allowed:
            kept.append(e)
    return kept


def retrieval_workload(retriever: Retriever, *, document_resource: DocumentResource | None = None) -> Workload:
    """query string or {'query': str, 'document_ids': [...]} -> {'query', 'evidence', 'degraded'}."""
    async def run(ctx: ExecutionContext, inp: Any) -> dict[str, Any]:
        if isinstance(inp, str):
            query, docs = inp, None
        elif isinstance(inp, dict) and isinstance(inp.get("query"), str):
            query, docs = inp["query"], inp.get("document_ids")
        else:
            raise ContractError("input must be a query string or {'query': str, 'document_ids': [...]}")
        if docs is not None and not (isinstance(docs, list) and all(isinstance(d, str) for d in docs)):
            raise ContractError("document_ids must be a list of strings")
        ctx.check()
        await authorize_request(ctx, docs, document_resource)
        res = await retriever.retrieve(ctx, query, docs)
        evidence = await authorize_evidence(ctx, res.evidence, document_resource)
        await emit(ctx, "retrieval.after", selected=len(evidence), withheld=len(res.evidence) - len(evidence),
                   degraded=list(res.degraded))
        return {"query": res.query, "evidence": [asdict(e) for e in evidence], "degraded": list(res.degraded)}
    return run
