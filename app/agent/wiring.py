"""Production wiring of Level-3 services onto the existing stack (OpenSearch, registry, RQ queue,
retrieval service, vLLM). Imported lazily by the API so unit tests never touch real infrastructure."""
import asyncio
import hashlib
import logging
from pathlib import Path
from uuid import uuid4

from app.agent.artifacts import ArtifactStore
from app.agent.capabilities.visualization import build_image_provider
from app.agent.config import AgentSettings, get_agent_settings
from app.agent.runtime import ModeDecision, ReasoningMode, RuntimeAdapter, detect_capabilities
from app.agent.services import Services
from app.agent.store import AgentStore
from app.agent.summarize import EvidenceSummarizer
from app.agent.web import WebFetcher, build_search_provider
from app.db import registry
from app.inference.config import InferenceSettings
from app.retrieval.context import TokenCounter

logger = logging.getLogger("rag.agent")
UPLOAD_DIR = Path(registry.DB_PATH).parent / "uploads"


def generation_counter(s: InferenceSettings) -> TokenCounter:
    """Counts with the *generation* model's tokenizer (not BGE-M3's). Falls back to a labelled estimate."""
    return TokenCounter(s.model)


def ingest_pdf(user_id: str, filename: str, pdf: bytes) -> dict:
    """Feed PDF bytes into the existing pipeline exactly as POST /documents does: content-hash
    duplicate check, registry row, RQ job running ingestion.worker.process_document."""
    from app.core.queue import ingestion_queue
    from app.ingestion.worker import process_document
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    doc_id = str(uuid4())
    safe = Path(filename).name
    path = UPLOAD_DIR / f"{doc_id}_{safe}"
    path.write_bytes(pdf)
    existing = registry.register_upload(doc_id, safe, str(path.resolve()), hashlib.sha256(pdf).hexdigest(), user_id)
    if existing is not None:
        path.unlink(missing_ok=True)
        return {"document_id": existing["document_id"], "job_id": existing["job_id"], "status": existing["status"], "duplicate": True}
    try:
        registry.update_status(doc_id, "queued")
        job = ingestion_queue.enqueue(process_document, doc_id, str(path.resolve()), job_timeout=3600, result_ttl=86400)
        registry.set_job_id(doc_id, job.id)
    except Exception:
        path.unlink(missing_ok=True)
        registry.update_status(doc_id, registry.FAILED_STATUS, error="Could not queue the file")
        raise
    return {"document_id": doc_id, "job_id": job.id, "status": "queued", "duplicate": False}


async def chunk_exists(document_id: str, chunk_id: str) -> bool:
    from opensearchpy.exceptions import NotFoundError
    from app.db.opensearch import INDEX_NAME, get_opensearch_client

    def q():
        try:
            r = get_opensearch_client().get(index=INDEX_NAME, id=chunk_id, _source_includes=["document_id"])
            return r["_source"].get("document_id") == document_id
        except NotFoundError:
            return False
    return await asyncio.to_thread(q)


async def chunk_count(document_id: str) -> int | None:
    from app.db.opensearch import INDEX_NAME, get_opensearch_client

    def q():
        try:
            return get_opensearch_client().count(index=INDEX_NAME, body={"query": {"term": {"document_id": document_id}}})["count"]
        except Exception:
            return None  # metadata stays useful when the index is unreachable
    return await asyncio.to_thread(q)


async def chunk_neighbors(document_id: str, chunk_id: str, before: int, after: int):
    from opensearchpy.exceptions import NotFoundError
    from app.db.opensearch import INDEX_NAME, get_opensearch_client

    def q():
        c = get_opensearch_client()
        try:
            src = c.get(index=INDEX_NAME, id=chunk_id)["_source"]
        except NotFoundError:
            return None
        if src.get("document_id") != document_id:
            return None
        idx = src["chunk_index"]
        must = [{"term": {"document_id": document_id}}, {"range": {"chunk_index": {"gte": max(0, idx - before), "lte": idx + after}}}]
        if src.get("document_version"):
            must.append({"term": {"document_version": src["document_version"]}})
        hits = c.search(index=INDEX_NAME, body={"size": before + after + 1, "sort": [{"chunk_index": "asc"}],
                                                "query": {"bool": {"must": must}}})["hits"]["hits"]
        return [h["_source"] for h in hits]
    return await asyncio.to_thread(q)


def build_llm_call(adapter: RuntimeAdapter):
    """LlmCall for the summarizer: non-thinking, JSON mode, tiny temperature."""
    from langchain_core.messages import HumanMessage, SystemMessage

    async def call(system: str, user: str, max_tokens: int):
        from app.agent.runtime import ReasoningController
        d = ReasoningController(adapter.caps).request(ReasoningMode.non_thinking, "evidence summarization")
        llm = adapter.build_llm(d, max_tokens=max_tokens, json_mode=True)
        r = await llm.ainvoke([SystemMessage(system), HumanMessage(user)])
        u = dict(getattr(r, "usage_metadata", None) or {})
        u["finish_reason"] = (r.response_metadata or {}).get("finish_reason")
        return (r.content if isinstance(r.content, str) else ""), u
    return call


def build_services(inference: InferenceSettings, retrieval, settings: AgentSettings | None = None) -> tuple[Services, RuntimeAdapter]:
    s = settings or get_agent_settings()
    store = AgentStore(s.store_path)
    counter = generation_counter(inference)
    caps = detect_capabilities(inference.model, inference.max_model_len)
    adapter = RuntimeAdapter(inference, caps)
    svc = Services(
        settings=s, store=store, counter=counter, retrieval=retrieval,
        summarizer=EvidenceSummarizer(build_llm_call(adapter), counter, batch_tokens=s.summary_batch_tokens, max_depth=s.summary_max_depth),
        search_provider=build_search_provider(s), fetcher=WebFetcher(s), image_provider=build_image_provider(s),
        artifact_store=ArtifactStore(s.artifact_dir, store, max_bytes=s.artifact_max_bytes, ttl_days=s.artifact_ttl_days),
        document_lookup=registry.get_document_for_owner, pages_lookup=registry.get_pages_for_owner,
        chunk_exists=chunk_exists, chunk_neighbors=chunk_neighbors, chunk_count=chunk_count, ingest_pdf=ingest_pdf)
    svc.extra["upload_dir"] = str(UPLOAD_DIR)
    return svc, adapter
