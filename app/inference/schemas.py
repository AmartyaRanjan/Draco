from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=64)
    mode: Literal["rag", "agent"] = "rag"
    thinking: bool | None = None  # None: off for rag, on for agent
    document_ids: list[str] | None = Field(default=None, max_length=50)
    max_output_tokens: int | None = Field(default=None, ge=16, le=1000)


class Citation(BaseModel):
    """Always built from retrieved evidence, never from model-written IDs."""
    label: str
    document_id: str
    chunk_id: str
    filename: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None


class ToolTraceEntry(BaseModel):
    """Written by the orchestrator at execution time."""
    step: int
    tool_name: str
    purpose: str
    input: dict
    output_summary: str
    output_ref: str | None = None  # key of the full output in the run store
    status: Literal["ok", "error", "invalid_arguments", "timeout", "denied", "limit_exceeded"]
    duration_ms: float


class ResponseMetadata(BaseModel):
    request_id: str
    conversation_id: str
    model: str
    mode: str
    thinking: bool
    evidence_sufficient: bool
    abstained: bool
    degraded_retrievers: list[str] = []
    invalid_citation_markers: list[str] = []
    finish_reason: str | None = None
    llm_calls: int = 0
    usage: dict = {}
    timings_ms: dict = {}


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    tool_trace: list[ToolTraceEntry]
    metadata: ResponseMetadata


class InferenceError(Exception):
    """Controlled failure with a client-safe message."""

    def __init__(self, message: str, code: str, status_code: int = 502):
        super().__init__(message)
        self.code, self.status_code = code, status_code
