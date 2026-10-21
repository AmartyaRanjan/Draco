"""Group A - knowledge: private retrieval over the existing hybrid pipeline (dense || BM25 -> exact
chunk_id dedup -> RRF -> ColBERT -> selection). Nothing here re-implements those stages."""
import re

from pydantic import BaseModel, Field

from app.agent.contracts import CapabilityError, Empty, ErrorCode, Outcome
from app.agent.capabilities.common import check_scope, require, safe_id
from app.agent.registry import Capability
from app.retrieval.schema import RetrievalError

SCORE_NOTE = "colbert_score is a raw MaxSim value, not a probability or percentage"


class EvidenceView(BaseModel):
    ref: str
    kind: str
    document_id: str | None = None
    chunk_id: str | None = None
    filename: str | None = None
    pages: str | None = None
    section_path: str | None = None
    colbert_score: float | None = None
    rrf_score: float | None = None
    selection: str | None = None
    text: str


def view(item, max_chars=1800) -> EvidenceView:
    pg = None
    if item.page_start is not None:
        pg = str(item.page_start) if item.page_end in (None, item.page_start) else f"{item.page_start}-{item.page_end}"
    return EvidenceView(ref=item.ref, kind=item.kind, document_id=item.document_id, chunk_id=item.chunk_id,
                        filename=item.filename, pages=pg, section_path=item.section_path,
                        colbert_score=item.colbert_score, rrf_score=item.rrf_score, selection=item.selection,
                        text=item.text[:max_chars])


# ---- search_knowledge_base ----------------------------------------------------------------
class SearchKBIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    document_ids: list[str] | None = Field(default=None, max_length=20)
    max_tokens: int | None = Field(default=None, ge=100, le=6000, description="evidence token budget for this call")


class SearchKBOut(BaseModel):
    items: list[EvidenceView]
    evidence_sufficient: bool = False
    passed_threshold: int = 0
    skipped_for_budget: int = 0
    degraded_retrievers: list[str] = []
    score_semantics: str = SCORE_NOTE
    summary: str = ""


async def search_knowledge_base(a: SearchKBIn, ctx):
    svc = ctx.services
    retrieval = require(svc.retrieval, "Knowledge retrieval")
    scope = a.document_ids
    if scope and ctx.document_scope and not set(scope) <= set(ctx.document_scope):
        raise CapabilityError(ErrorCode.denied, "Document outside the permitted set.")
    scope = scope or ctx.document_scope
    try:
        out = await retrieval.retrieve(a.query, scope, f"{ctx.trace_id}-kb{len(ctx.scratch.setdefault('kb_calls', []))}", ctx.user_id)
    except RetrievalError as exc:
        raise CapabilityError(ErrorCode.upstream_error, f"Retrieval failed at stage {exc.stage}.", retryable=True) from exc
    ctx.scratch["kb_calls"].append(a.query)
    ctx.scratch.setdefault("retrieval_traces", []).append(out.trace.as_dict())
    sel = out.selection
    if not sel or not sel.selected:
        raise Empty({"items": [], "degraded_retrievers": out.degraded, "summary": "0 results"})
    budget = a.max_tokens or ctx.scratch.get("evidence_tokens_left", 3000)
    items, skipped = [], 0
    for c in sel.selected:  # whole chunks, in rerank order, within the budget; never truncated silently
        fn = retrieval.filename_for(c.document_id)
        cost = svc.counter.count(c.text) + 40
        if cost > budget:
            skipped += 1
            continue
        budget -= cost
        items.append(ctx.evidence.add(dict(
            kind="private", text=c.text, document_id=c.document_id, chunk_id=c.chunk_id, filename=fn,
            page_start=c.page_start, page_end=c.page_end, section_path=c.section_path,
            colbert_score=c.colbert_score, rrf_score=c.rrf_score, selection=c.selection)))
    ctx.scratch["evidence_tokens_left"] = budget
    if not items:
        raise CapabilityError(ErrorCode.limit_exceeded, "Evidence found but it does not fit the remaining token budget; narrow the query.")
    res = SearchKBOut(items=[view(i) for i in items], evidence_sufficient=sel.evidence_sufficient,
                      passed_threshold=sel.passed, skipped_for_budget=skipped, degraded_retrievers=out.degraded,
                      summary=f"{len(items)} chunks, {sel.passed} passed threshold")
    return Outcome(res, extra={"retrieval_ms": out.trace.stages.get("total", {}).get("ms"), "degraded": out.degraded})


# ---- get_document_metadata ----------------------------------------------------------------
class DocMetaIn(BaseModel):
    document_id: str = Field(min_length=1, max_length=128)


class DocMetaOut(BaseModel):
    document_id: str
    filename: str
    status: str
    created_at: str | None = None
    content_hash_prefix: str | None = None
    page_count: int = 0
    ocr_page_count: int = 0
    low_yield_pages: list[int] = []
    has_error: bool = False
    chunk_count: int | None = None


async def get_document_metadata(a: DocMetaIn, ctx):
    svc = ctx.services
    lookup = require(svc.document_lookup, "Document registry")
    check_scope(ctx, a.document_id)
    doc = lookup(safe_id(a.document_id), ctx.user_id)
    if doc is None:
        raise CapabilityError(ErrorCode.not_found, "Document not found.")
    pages = (svc.pages_lookup(a.document_id, ctx.user_id) if svc.pages_lookup else []) or []
    n_chunks = await svc.chunk_count(a.document_id) if svc.chunk_count else None
    return DocMetaOut(document_id=doc["document_id"], filename=doc["filename"], status=doc["status"],
                      created_at=doc.get("created_at"), content_hash_prefix=(doc.get("content_hash") or "")[:12] or None,
                      page_count=len(pages), ocr_page_count=sum(1 for p in pages if p.get("ocr_used")),
                      low_yield_pages=[p["page_no"] for p in pages if p.get("low_yield")],
                      has_error=bool(doc.get("error")), chunk_count=n_chunks)  # file_path / raw error text are never exposed


# ---- expand_document_context --------------------------------------------------------------
class ExpandIn(BaseModel):
    document_id: str = Field(min_length=1, max_length=128)
    chunk_id: str = Field(min_length=1, max_length=128)
    before: int = Field(default=1, ge=0, le=3)
    after: int = Field(default=1, ge=0, le=3)


class ExpandOut(BaseModel):
    items: list[EvidenceView]
    summary: str = ""


async def expand_document_context(a: ExpandIn, ctx):
    svc = ctx.services
    fn = require(svc.chunk_neighbors, "Chunk neighbor lookup")
    check_scope(ctx, a.document_id)
    lookup = require(svc.document_lookup, "Document registry")
    if lookup(safe_id(a.document_id), ctx.user_id) is None:
        raise CapabilityError(ErrorCode.not_found, "Document not found.")
    rows = await fn(safe_id(a.document_id), safe_id(a.chunk_id), a.before, a.after)
    if rows is None:
        raise CapabilityError(ErrorCode.not_found, "Chunk not found.")
    doc = lookup(a.document_id, ctx.user_id)
    name = doc.get("filename") if doc else None
    items = [ctx.evidence.add(dict(kind="private", text=r["text"], document_id=r["document_id"], chunk_id=r["chunk_id"],
                                   filename=name, page_start=r.get("page_start"), page_end=r.get("page_end"),
                                   section_path=r.get("section_path"), selection="context_expansion")) for r in rows]
    if not items:
        raise Empty({"items": []})
    return ExpandOut(items=[view(i) for i in items], summary=f"{len(items)} chunks")


# ---- rewrite_search_query (deterministic, no LLM) -------------------------------------------
class RewriteIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    previous_queries: list[str] = Field(default=[], max_length=8)


class RewriteOut(BaseModel):
    variants: list[str]
    strategy: dict[str, str]
    summary: str = ""


_STOP = frozenset("what which who whom whose when where why how is are was were do does did can could should would "
                  "the a an of to in on for and or about please tell me explain describe give show find".split())


def rewrite_variants(query: str, previous: list[str]) -> dict[str, str]:
    q = " ".join(query.split())
    words = re.findall(r"[\w'-]+", q, re.U)
    out: dict[str, str] = {}
    kw = " ".join(w for w in words if w.lower() not in _STOP)
    if kw and kw.lower() != q.lower():
        out[kw] = "keywords_only"
    ids = [w for w in words if re.search(r"\d", w) or (w.isupper() and len(w) > 1) or re.search(r"[A-Za-z]\d|\d[A-Za-z]", w)]
    if ids:
        out[" ".join(ids)] = "identifiers_only"
    parts = [p.strip() for p in re.split(r"\b(?:and|versus|vs\.?|compared? to|then)\b|[;,]", q, flags=re.I) if len(p.split()) >= 2]
    for p in parts[:3]:
        if len(parts) > 1:
            out[p] = "sub_question"
    if len(words) > 6:
        out[" ".join(words[: max(4, len(words) // 2)])] = "shortened"
    seen = {x.lower() for x in previous} | {q.lower()}
    return {k: v for k, v in out.items() if k.lower() not in seen}


async def rewrite_search_query(a: RewriteIn, ctx):
    v = rewrite_variants(a.query, a.previous_queries)
    if not v:
        raise Empty({"variants": [], "strategy": {}, "summary": "no new variants"})
    top = dict(list(v.items())[:4])
    return RewriteOut(variants=list(top), strategy=top, summary=f"{len(top)} variants")


CAPS = [
    Capability("search_knowledge_base", "knowledge", "Hybrid search (dense+BM25, RRF, ColBERT) over the private documents. "
               "Returns labelled evidence chunks with real document_id/chunk_id.", SearchKBIn, SearchKBOut,
               search_knowledge_base, cost="medium", timeout_s=40, max_output_chars=30_000),
    Capability("get_document_metadata", "knowledge", "Metadata for one document (filename, status, page counts).",
               DocMetaIn, DocMetaOut, get_document_metadata),
    Capability("expand_document_context", "knowledge", "Fetch chunks adjacent to a known chunk for more context.",
               ExpandIn, ExpandOut, expand_document_context, cost="medium"),
    Capability("rewrite_search_query", "knowledge", "Deterministic query variants (keywords, identifiers, sub-questions).",
               RewriteIn, RewriteOut, rewrite_search_query),
]
