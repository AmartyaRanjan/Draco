"""Group C - knowledge_management: approval-gated ingestion of web sources.

Authorization is a database record written by the authenticated HTTP layer for one exact source
(url hash + content hash). Capabilities here can *read* approvals, never create them; nothing in the
model's control can set an approval flag. One job per source (unique constraint), reused on retry.
"""
import hashlib
import time
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from app.agent import artifacts as A
from app.agent.capabilities.common import require
from app.agent.contracts import CapabilityError, Empty, ErrorCode, Outcome
from app.agent.registry import Capability
from app.agent.web import normalize_url, url_hash

_HIGH_TRUST = (".gov", ".edu", ".ac.uk", ".int")
_ACTIVE = ("pending_approval", "approved", "ingesting", "ingested")


class ProposeIn(BaseModel):
    page_id: str = Field(min_length=4, max_length=32, description="page_id from fetch_webpage in this run")
    reason: str = Field(min_length=5, max_length=500, description="why this page should be saved")


class ProposalOut(BaseModel):
    source_id: str
    status: str
    approval_required: bool = True
    preview: dict
    trust: dict
    duplicate_of: str | None = None
    how_to_approve: str = ""
    summary: str = ""


def trust_assessment(url: str, page: dict) -> dict:
    host = (urlsplit(url).hostname or "").lower()
    warnings = []
    if urlsplit(url).scheme != "https":
        warnings.append("not served over HTTPS")
    if page.get("injection_suspected"):
        warnings.append("page contains instruction-like text (possible prompt injection)")
    if not page.get("title"):
        warnings.append("page has no title")
    tld = "high" if host.endswith(_HIGH_TRUST) else "unknown"
    level = "low" if warnings else ("medium-high" if tld == "high" else "medium")
    return {"level": level, "domain": host, "domain_class": tld, "https": urlsplit(url).scheme == "https",
            "warnings": warnings, "basis": "heuristic indicators only; not a verification of accuracy"}


async def propose_knowledge_ingestion(a: ProposeIn, ctx):
    svc = ctx.services
    page = ctx.pages.get(a.page_id)
    if page is None:
        raise CapabilityError(ErrorCode.not_found, "Unknown page_id; fetch the page first.")
    final_url = page["final_url"]
    uh = url_hash(final_url)
    for prev in svc.store.sources_for_url(ctx.user_id, uh):  # duplicate URLs / existing source records
        if prev["status"] in _ACTIVE:
            existing = svc.store.get_source(prev["source_id"], ctx.user_id)
            return Outcome(ProposalOut(source_id=existing["source_id"], status=existing["status"],
                                       approval_required=existing["status"] == "pending_approval",
                                       preview=existing["preview"], trust=existing["trust"],
                                       duplicate_of=existing["source_id"],
                                       how_to_approve=f"POST /agent/sources/{existing['source_id']}/approve",
                                       summary="existing source record reused"),
                           status="pending" if existing["status"] == "pending_approval" else "success")
    text = page["text"][:svc.settings.web_snapshot_max_chars]
    if A.unrenderable_ratio(text) > 0.05:
        raise CapabilityError(ErrorCode.invalid_input, "The page uses scripts the PDF-based ingestion path cannot render; it cannot be saved.")
    chash = hashlib.sha256(text.encode()).hexdigest()
    trust = trust_assessment(final_url, page)
    preview = {"title": page["title"] or urlsplit(final_url).hostname, "url": final_url, "domain": trust["domain"],
               "characters": len(text), "excerpt": text[:500], "content_sha256": chash, "fetched_at": page["fetched_at"],
               "will_be_saved_as": "a text snapshot rendered to PDF and indexed through the standard ingestion pipeline",
               "reason": a.reason}
    src = svc.store.create_source(user_id=ctx.user_id, url=final_url, normalized_url=normalize_url(final_url),
                                  url_hash=uh, title=preview["title"], content_hash=chash, content_text=text,
                                  reason=a.reason, preview=preview, trust=trust, ttl_hours=svc.settings.approval_ttl_hours)
    return Outcome(ProposalOut(source_id=src["source_id"], status="pending_approval", preview=preview, trust=trust,
                               how_to_approve=f"POST /agent/sources/{src['source_id']}/approve (authenticated user only)",
                               summary="awaiting explicit user approval"),
                   status="pending", extra={"source_id": src["source_id"]})


class IngestIn(BaseModel):
    source_id: str = Field(min_length=8, max_length=64)


class IngestOut(BaseModel):
    source_id: str
    job_id: str
    document_id: str | None = None
    rq_job_id: str | None = None
    status: str
    reused_job: bool = False
    duplicate_document: bool = False
    summary: str = ""


def _refresh(svc, job: dict, src: dict) -> dict:
    """Mirror the ingestion registry's status (queued/processing/completed/failed) onto our records."""
    if job.get("document_id") and svc.document_lookup:
        doc = svc.document_lookup(job["document_id"], src["user_id"])
        if doc and doc["status"] != job["status"]:
            svc.store.update_job(job["job_id"], status=doc["status"], error=None if doc["status"] != "failed" else "ingestion failed")
            svc.store.set_source_status(src["source_id"], {"completed": "ingested", "failed": "failed"}.get(doc["status"], "ingesting"))
            job = svc.store.job_for_source(src["source_id"])
    return job


async def ingest_approved_source(a: IngestIn, ctx):
    svc = ctx.services
    ingest = require(svc.ingest_pdf, "Ingestion submit")
    src = svc.store.get_source(a.source_id, ctx.user_id)
    if src is None:
        raise CapabilityError(ErrorCode.not_found, "Source not found.")
    if not svc.store.valid_approval(a.source_id, ctx.user_id):
        raise CapabilityError(ErrorCode.approval_required,
                              "No valid user approval for this exact source (missing, rejected, expired, or content changed).")
    key = hashlib.sha256(f"{ctx.user_id}|{src['url_hash']}|{src['content_hash']}".encode()).hexdigest()
    job, created = svc.store.claim_job(a.source_id, key)
    if not created:
        if job["status"] == "failed" and svc.store.reclaim_failed_job(job["job_id"]):
            created = True  # explicit retry of a failed job: same job record, new submission
        else:
            job = _refresh(svc, job, src)
            return Outcome(IngestOut(source_id=a.source_id, job_id=job["job_id"], document_id=job["document_id"],
                                     rq_job_id=job["rq_job_id"], status=job["status"], reused_job=True,
                                     summary="existing ingestion job reused"),
                           status="pending" if job["status"] in ("claimed", "queued", "processing", "uploaded") else "success",
                           extra={"job_id": job["job_id"]})
    title = src["title"] or "web source"
    header = [f"Source URL: {src['url']}", f"Retrieved: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(src['created_at']))} UTC",
              f"Content SHA-256: {src['content_hash']}", f"Approved source_id: {src['source_id']}"]
    import asyncio

    async def renew_claim():
        while True:
            await asyncio.sleep(60)
            if not await asyncio.to_thread(svc.store.renew_claim, job["job_id"]):
                return

    renewal = asyncio.create_task(renew_claim())
    try:
        pdf = await asyncio.to_thread(A.text_to_pdf, title, header, src["content_text"])
        res = await asyncio.to_thread(ingest, ctx.user_id, f"{title[:80]} ({src['preview']['domain']}).pdf", pdf)
    except Exception as exc:
        svc.store.update_job(job["job_id"], status="failed", error=type(exc).__name__)
        svc.store.set_source_status(a.source_id, "failed")
        raise CapabilityError(ErrorCode.upstream_error, "Could not submit the source to the ingestion queue.", retryable=True) from exc
    finally:
        renewal.cancel()
        await asyncio.gather(renewal, return_exceptions=True)
    svc.store.update_job(job["job_id"], document_id=res["document_id"], rq_job_id=res.get("job_id"), status=res["status"])
    svc.store.set_source_status(a.source_id, "ingesting" if res["status"] != "completed" else "ingested")
    return Outcome(IngestOut(source_id=a.source_id, job_id=job["job_id"], document_id=res["document_id"],
                             rq_job_id=res.get("job_id"), status=res["status"], duplicate_document=bool(res.get("duplicate")),
                             summary="ingestion job queued" if not res.get("duplicate") else "identical document already indexed"),
                   status="pending" if res["status"] != "completed" else "success", extra={"job_id": job["job_id"]})


class ProvenanceIn(BaseModel):
    source_id: str | None = Field(default=None, max_length=64)
    document_id: str | None = Field(default=None, max_length=128)
    url: str | None = Field(default=None, max_length=2048)


class ProvenanceOut(BaseModel):
    found: bool
    source: dict | None = None
    approval: dict | None = None
    job: dict | None = None
    document: dict | None = None
    summary: str = ""


async def inspect_source_provenance(a: ProvenanceIn, ctx):
    svc = ctx.services
    if sum(x is not None for x in (a.source_id, a.document_id, a.url)) != 1:
        raise CapabilityError(ErrorCode.invalid_input, "Provide exactly one of source_id, document_id, url.")
    src = None
    if a.source_id:
        src = svc.store.get_source(a.source_id, ctx.user_id)
    elif a.url:
        rows = svc.store.sources_for_url(ctx.user_id, url_hash(a.url))
        src = svc.store.get_source(rows[0]["source_id"], ctx.user_id) if rows else None
    else:
        doc = svc.document_lookup(a.document_id, ctx.user_id) if svc.document_lookup else None
        job = svc.store._one("SELECT source_id FROM ingestion_jobs WHERE document_id=?", (a.document_id,)) if doc else None
        src = svc.store.get_source(job["source_id"], ctx.user_id) if job else None
    doc = svc.document_lookup(a.document_id, ctx.user_id) if (a.document_id and svc.document_lookup) else None
    if src is None:
        if doc:
            return ProvenanceOut(found=True, document={"document_id": doc["document_id"], "filename": doc["filename"],
                                                      "status": doc["status"], "origin": "direct upload (no web source record)"},
                                 summary="document has no web-source record")
        raise Empty({"found": False, "summary": "no source record"})
    job = svc.store.job_for_source(src["source_id"])
    if job:
        job = _refresh(svc, job, src)
        src = svc.store.get_source(src["source_id"], ctx.user_id)  # status may have just been refreshed
    ap = svc.store.get_approval(src["source_id"], ctx.user_id)
    d = svc.document_lookup(job["document_id"], ctx.user_id) if (job and job.get("document_id") and svc.document_lookup) else doc
    return ProvenanceOut(
        found=True,
        source={k: src[k] for k in ("source_id", "url", "title", "status", "content_hash", "reason", "created_at", "expires_at")} | {"trust": src["trust"]},
        approval={k: ap[k] for k in ("decision", "decided_at", "expires_at", "revoked")} if ap else None,
        job={k: job[k] for k in ("job_id", "status", "document_id", "rq_job_id", "created_at", "updated_at")} if job else None,
        document={"document_id": d["document_id"], "filename": d["filename"], "status": d["status"]} if d else None,
        summary=f"source {src['status']}")


CAPS = [
    Capability("propose_knowledge_ingestion", "knowledge_management", "Prepare a fetched web page as a candidate for permanent "
               "saving. Creates a proposal and preview only; nothing is saved until the user approves.", ProposeIn,
               ProposalOut, propose_knowledge_ingestion, permissions=frozenset({"read", "write"}), mutates_state=True,
               requires_approval=True, untrusted_output=False, parallel_safe=False),
    Capability("ingest_approved_source", "knowledge_management", "Submit a user-approved source to the ingestion pipeline "
               "(idempotent: one job per source).", IngestIn, IngestOut, ingest_approved_source,
               permissions=frozenset({"read", "write"}), mutates_state=True, requires_approval=True, cost="medium",
               parallel_safe=False, llm_exposed=False, timeout_s=60),
    Capability("inspect_source_provenance", "knowledge_management", "Show where a saved source came from: approval, job and "
               "document status.", ProvenanceIn, ProvenanceOut, inspect_source_provenance),
]
