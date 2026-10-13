from typing import Literal

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=64)
    document_ids: list[str] | None = Field(default=None, max_length=50, description="restricts private retrieval / names uploaded files")
    allow_web: bool = Field(default=False, description="opt in to web search/fetch for this request")
    reasoning: Literal["auto", "thinking", "non_thinking"] = "auto"
    max_output_tokens: int | None = Field(default=None, ge=16, le=1000)


class CitationOut(BaseModel):
    ref: str
    kind: Literal["private", "web"]
    document_id: str | None = None
    chunk_id: str | None = None
    filename: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    url: str | None = None
    passage_id: str | None = None
    title: str | None = None


class AgentTraceEntry(BaseModel):
    step: int
    tool_name: str
    capability_group: str
    purpose: str
    status: str
    input: dict
    output_summary: str
    duration_ms: float
    execution_id: str
    output_chars: int
    error_code: str | None = None
    retry_count: int = 0
    extra: dict = {}


class AgentChatResponse(BaseModel):
    answer: str
    status: Literal["ok", "partial", "abstained", "awaiting_approval", "failed"]
    citations: list[CitationOut]
    tool_trace: list[AgentTraceEntry]
    artifacts: list[dict] = []
    manifest: dict | None = None
    pending_approvals: list[dict] = []
    validation: dict | None = None
    reasoning: list[dict] = []        # requested vs applied mode per decision (no model reasoning text)
    context: dict = {}                # budget snapshots + compaction log
    warnings: list[str] = []
    metadata: dict = {}
