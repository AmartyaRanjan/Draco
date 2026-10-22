import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.chat import heavy_user
from app.core import concurrency
from app.retrieval.context import pages
from app.retrieval.schema import RetrievalError
from app.retrieval.service import MAX_QUERY_CHARS, get_service

logger = logging.getLogger("rag.retrieval")
router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    document_ids: list[str] | None = Field(default=None, max_length=50)
    generate: bool = True
    include_diagnostics: bool = False


@router.post("/query")
async def query(req: QueryRequest, user: str = Depends(heavy_user)):
    svc = get_service()
    try:
        if req.generate:
            async with concurrency.llm_slot():
                out, gen = await svc.answer(req.query, req.document_ids, principal=user)
        else:
            out, gen = await svc.retrieve(req.query, req.document_ids, principal=user), None
            out.trace.finish("degraded" if out.degraded else "ok")
    except HTTPException:
        raise
    except RetrievalError as exc:
        logger.error("query_failed", extra={"stage": exc.stage})
        raise HTTPException(exc.status_code, f"{exc} (stage: {exc.stage})") from exc
    except Exception as exc:
        logger.exception("query_unexpected")
        raise HTTPException(500, "Internal error while answering the query") from exc

    sel = out.selection
    body = {
        "request_id": out.request_id,
        "answer": gen.answer if gen else None,
        "abstained": gen is None,
        "evidence_sufficient": bool(sel and sel.evidence_sufficient),
        "degraded_retrievers": out.degraded,
        "citations": gen.citations if gen else [],
        "sources": [
            {"label": e.label, "chunk_id": e.candidate.chunk_id, "document_id": e.candidate.document_id,
             "filename": e.filename, "pages": pages(e.candidate), "section_path": e.candidate.section_path,
             "selection": e.candidate.selection, "rrf_score": e.candidate.rrf_score,
             "colbert_score": e.candidate.colbert_score,
             "calibrated_probability": e.candidate.calibrated_probability}
            for e in out.context.items
        ],
    }
    if req.include_diagnostics:
        body["diagnostics"] = out.trace.as_dict()
    return body
