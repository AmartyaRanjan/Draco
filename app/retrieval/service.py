"""Retrieval orchestration; usable without FastAPI."""
import asyncio
import logging
import uuid
from dataclasses import dataclass, field

from app.db.registry import DocumentScopeTooLarge, get_document, list_document_ids_for_owner
from app.core.blocking import run_blocking
from app.retrieval import fusion
from app.retrieval.context import BuiltContext, TokenCounter, build_context
from app.retrieval.generator import SYSTEM, WEAK_NOTE
from app.retrieval.rerank import ColbertReranker
from app.retrieval.retrievers import DenseRetriever, LexicalRetriever, prepare_query
from app.retrieval.schema import Candidate, RetrievalError
from app.retrieval.selection import Calibrator, SelectionResult, select
from app.retrieval.settings import RetrievalSettings, get_settings, load_calibration
from app.retrieval.timing import COLD_START_MS, Trace, now_ns

logger = logging.getLogger("rag.retrieval")
MAX_QUERY_CHARS = 2000
GENERATION_OVERHEAD_TOKENS = 200


@dataclass
class RetrievalOutcome:
    request_id: str
    query: str
    selection: SelectionResult | None
    context: BuiltContext
    trace: Trace
    degraded: list[str] = field(default_factory=list)


def resolve_filters(requested_document_ids: list[str] | None, principal=None) -> list[str]:
    """Intersect client filters with the authenticated principal's owned documents."""
    if not isinstance(principal, str) or not principal.strip():
        return []
    try:
        return list_document_ids_for_owner(principal, requested_document_ids)
    except DocumentScopeTooLarge as exc:
        raise RetrievalError(str(exc), "request_validation", 413) from exc


class RetrievalService:
    def __init__(self, settings: RetrievalSettings | None = None, dense=None, lexical=None,
                 reranker=None, counter=None, generator=None, filename_lookup=None):
        s = self.s = settings or get_settings()
        self.dense = dense or DenseRetriever(s.dense_top_k, s.dense_timeout_s)
        self.lexical = lexical or LexicalRetriever(s.bm25_top_k, s.bm25_timeout_s)
        self.reranker = reranker or ColbertReranker(
            s.colbert_model, s.colbert_device, s.colbert_share_backbone,
            s.colbert_batch_size, s.colbert_max_length, s.colbert_max_candidates)
        self.counter = counter or TokenCounter(s.tokenizer)
        self.calibrator = Calibrator(load_calibration(s.calibration_file)) if s.calibration_file else None
        # Answer generation lives in app.inference; /query only generates if one is injected.
        self.generator = generator
        self._filename = filename_lookup or (lambda d: (get_document(d) or {}).get("filename"))

    def filename_for(self, document_id: str) -> str | None:
        return self._filename(document_id)

    async def _hybrid(self, query, doc_ids, trace):
        t0 = now_ns()
        results = await asyncio.gather(
            self.dense.retrieve(query, doc_ids, trace),
            self.lexical.retrieve(query, doc_ids, trace),
            return_exceptions=True)
        trace.record("hybrid_retrieval", t0)  # wall clock, not a sum
        names = ("dense", "bm25")
        degraded, ok = [], []
        for name, r in zip(names, results):
            if isinstance(r, asyncio.CancelledError):
                # Cancellation is not a retriever failure: the caller went away.
                # Re-raise so the request cancels instead of degrading.
                raise r
            if isinstance(r, BaseException):
                logger.error("retriever_failed", extra={"retriever": name, "error": type(r).__name__})
                degraded.append(name)
                ok.append([])
            else:
                ok.append(r)
        if len(degraded) == 2:
            raise RetrievalError("Both retrievers failed", "hybrid_retrieval", 503)
        return ok[0], ok[1], degraded

    async def retrieve(self, query: str, document_ids: list[str] | None = None,
                       request_id: str | None = None, principal=None) -> RetrievalOutcome:
        s, trace = self.s, Trace(request_id or str(uuid.uuid4()))
        trace.set(reranker_model=s.colbert_model, generator_model=s.llm_model)
        try:
            with trace.stage("request_validation"):
                q = prepare_query(query or "")
                if not q or len(q) > MAX_QUERY_CHARS:
                    raise RetrievalError("Query must be 1-2000 characters", "request_validation", 422)
                doc_ids = resolve_filters(document_ids, principal)
            with trace.stage("query_preprocessing"):
                pass  # normalisation done above; stage kept for a stable timing schema
            if not doc_ids:
                trace.set(dense_candidates=0, bm25_candidates=0, overlap=0, unique_candidates=0,
                          conflicts=0, degraded=[], reranked=0, selected=0, passed=0, fallback=0)
                return RetrievalOutcome(trace.request_id, q, None, BuiltContext(), trace, [])
            dense, lex, degraded = await self._hybrid(q, doc_ids, trace)
            with trace.stage("deduplication"):
                unique, stats = fusion.deduplicate(dense, lex)
            with trace.stage("rrf_fusion"):
                fused = fusion.rrf(unique, s.rrf_k)
            trace.set(dense_candidates=len(dense), bm25_candidates=len(lex),
                      overlap=stats["overlap"], unique_candidates=stats["unique"],
                      conflicts=stats["conflicts"], degraded=degraded)
            if not fused:
                trace.set(reranked=0, selected=0, passed=0, fallback=0)
                return RetrievalOutcome(trace.request_id, q, None, BuiltContext(), trace, degraded)
            with trace.stage("colbert_rerank"):
                try:
                    ranked, rstats = await run_blocking(
                        self.reranker.rerank, q, fused, timeout=s.dense_timeout_s
                    )
                except Exception as exc:
                    raise RetrievalError("Reranking failed", "colbert_rerank", 500) from exc
            trace.set(**{"reranked": rstats["reranked"], "rerank_batches": rstats["batches"],
                         "rerank_batch_size": rstats["batch_size"]})
            with trace.stage("selection"):
                sel = select(ranked, s.score_threshold, s.min_selected, s.max_selected, self.calibrator)
            with trace.stage("context_assembly"):
                names = {d: self._filename(d) for d in {c.document_id for c in sel.selected}}
                reserved = (self.counter.count(SYSTEM + WEAK_NOTE + q) + GENERATION_OVERHEAD_TOKENS
                            + s.llm_max_output_tokens)
                ctx = build_context(sel.selected, names, s.max_context_tokens, self.counter, reserved)
            trace.set(selected=len(ctx.items), passed=sel.passed, fallback=sel.fallback,
                      context_tokens=ctx.tokens, dropped_for_budget=len(ctx.dropped),
                      threshold_basis=sel.basis, evidence_sufficient=sel.evidence_sufficient,
                      cold_start_ms=dict(COLD_START_MS))
            if ctx.dropped:
                logger.warning("context_budget_drop", extra={"dropped": len(ctx.dropped)})
            return RetrievalOutcome(trace.request_id, q, sel, ctx, trace, degraded)
        except BaseException:
            trace.finish("error")
            raise

    async def answer(self, query: str, document_ids=None, request_id=None, principal=None):
        out = await self.retrieve(query, document_ids, request_id, principal)
        trace, gen = out.trace, None
        try:
            if not out.context.items:
                trace.set(abstained="no_evidence")
            elif self.generator is None:
                trace.set(abstained="generator_not_configured")
            else:
                with trace.stage("llm"):
                    gen = await self.generator.generate(
                        out.query, out.context, weak=not (out.selection and out.selection.evidence_sufficient))
                for label in gen.invalid_markers:  # never let unverifiable references reach the client
                    gen.answer = gen.answer.replace(f"[{label}]", "")
                trace.stages.update({k: {"ms": v, "status": "ok"} for k, v in gen.timings.items()})
                trace.set(**gen.usage, invalid_citations=gen.invalid_markers)
        except BaseException:
            trace.finish("error")
            raise
        trace.finish("degraded" if out.degraded else "ok")
        return out, gen


_service: RetrievalService | None = None


def get_service() -> RetrievalService:
    global _service
    if _service is None:
        _service = RetrievalService()
    return _service
