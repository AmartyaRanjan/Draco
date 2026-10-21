"""Level-3 atomic services, injected into capability handlers. Anything unset means "not configured"
and the capability reports it; it never pretends to work."""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.agent.config import AgentSettings
from app.agent.store import AgentStore


@dataclass
class Services:
    settings: AgentSettings
    store: AgentStore
    counter: Any                                   # TokenCounter for the *generation* model
    retrieval: Any = None                          # RetrievalService
    summarizer: Any = None                         # EvidenceSummarizer
    search_provider: Any = None                    # SearchProvider
    fetcher: Any = None                            # WebFetcher
    image_provider: Any = None                     # ImageProvider
    artifact_store: Any = None                     # ArtifactStore
    document_lookup: Callable[[str, str], dict | None] | None = None  # registry.get_document_for_owner
    pages_lookup: Callable[[str, str], list[dict]] | None = None      # registry.get_pages_for_owner
    chunk_exists: Callable[[str, str], Awaitable[bool]] | None = None  # index existence check
    chunk_neighbors: Callable[..., Awaitable[list[dict]]] | None = None
    chunk_count: Callable[[str], Awaitable[int | None]] | None = None
    ingest_pdf: Callable[[str, str, bytes], dict] | None = None       # (user, filename, pdf bytes) -> {document_id, job_id, status, duplicate}
    entailment: Callable[[str, str], Awaitable[str]] | None = None    # optional claim-vs-passage judge
    extra: dict = field(default_factory=dict)

    def missing(self) -> dict[str, str]:
        """Human-readable list of unconfigured providers (surfaced by GET /agent/capabilities)."""
        m = {}
        if self.retrieval is None: m["retrieval"] = "RetrievalService not wired"
        if self.summarizer is None or getattr(self.summarizer, "llm", None) is None:
            m["summarizer_llm"] = "no LLM call configured; summarize_evidence uses the extractive fallback"
        if self.search_provider is None: m["web_search"] = "set WEB_SEARCH_PROVIDER (searxng|brave) and WEB_SEARCH_URL / WEB_SEARCH_API_KEY"
        if self.image_provider is None: m["image_generation"] = "set IMAGE_API_URL, IMAGE_API_KEY, IMAGE_MODEL"
        if self.ingest_pdf is None: m["ingestion"] = "ingestion enqueue function not wired"
        return m
