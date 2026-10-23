"""Agent-layer settings. Every value comes from the environment; secrets are never defaulted."""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_STORAGE = Path(__file__).resolve().parents[1] / "storage"


def _i(n, d): return int(os.getenv(n, d))
def _f(n, d): return float(os.getenv(n, d))
def _b(n, d): return os.getenv(n, d).lower() == "true"


@dataclass(frozen=True, slots=True)
class AgentSettings:
    # state
    store_path: str
    artifact_dir: str
    artifact_max_bytes: int
    artifact_ttl_days: int
    feedback_retention_days: int
    # graph limits
    max_tool_calls: int
    max_retrieval_attempts: int
    max_llm_steps: int
    max_mode_changes: int
    artifact_output_tokens: int  # output cap for artifact/visualization drafting calls (ordinary answers stay at INFERENCE_MAX_OUTPUT_TOKENS)
    capability_timeout_s: float
    # context budget
    evidence_compaction_ratio: float  # compact when evidence use > ratio * evidence budget
    safety_margin_tokens: int
    tool_reserve_tokens: int
    summary_target_tokens: int
    summary_batch_tokens: int
    summary_max_depth: int
    # web
    web_search_provider: str | None  # searxng | brave
    web_search_url: str | None
    web_search_api_key: str | None
    web_allowed_ports: tuple
    web_max_bytes: int
    web_timeout_s: float
    web_max_redirects: int
    web_allow_private_hosts: bool  # tests/dev only
    # ingestion approval
    approval_ttl_hours: int
    web_snapshot_max_chars: int
    # image generation (OpenAI-compatible /images/generations)
    image_api_url: str | None
    image_api_key: str | None
    image_model: str | None
    # file analysis
    analysis_max_bytes: int

    def validate(self, *, production: bool = False) -> "AgentSettings":
        if not 0.3 <= self.evidence_compaction_ratio <= 1.0:
            raise ValueError("AGENT_EVIDENCE_COMPACTION_RATIO must be in [0.3, 1.0].")
        if min(self.max_tool_calls, self.max_retrieval_attempts, self.max_llm_steps) < 1:
            raise ValueError("Agent loop limits must be >= 1.")
        if not 256 <= self.artifact_output_tokens <= 4096:
            raise ValueError("AGENT_ARTIFACT_MAX_OUTPUT_TOKENS must be in [256, 4096].")
        if self.max_mode_changes < 0:
            raise ValueError("AGENT_MAX_MODE_CHANGES must be >= 0.")
        if self.summary_target_tokens < 64 or self.summary_batch_tokens < 256:
            raise ValueError("Summary budgets too small.")
        if production and self.web_allow_private_hosts:
            raise ValueError("WEB_ALLOW_PRIVATE_HOSTS=true is not allowed in production.")
        return self


@lru_cache
def get_agent_settings() -> AgentSettings:
    ports = tuple(int(p) for p in os.getenv("WEB_ALLOWED_PORTS", "80,443").split(",") if p.strip())
    return AgentSettings(
        store_path=os.getenv("AGENT_STORE_PATH", str(_STORAGE / "agent.db")),
        artifact_dir=os.getenv("ARTIFACT_DIR", str(_STORAGE / "artifacts")),
        artifact_max_bytes=_i("ARTIFACT_MAX_BYTES", 25 * 1024 * 1024),
        artifact_ttl_days=_i("ARTIFACT_TTL_DAYS", 30),
        feedback_retention_days=_i("FEEDBACK_RETENTION_DAYS", 365),
        max_tool_calls=_i("AGENT_MAX_TOOL_CALLS_PER_TASK", 12),
        max_retrieval_attempts=_i("AGENT_MAX_RETRIEVAL_ATTEMPTS", 2),
        max_llm_steps=_i("AGENT_MAX_LLM_STEPS", 4),
        max_mode_changes=_i("AGENT_MAX_MODE_CHANGES", 2),
        artifact_output_tokens=_i("AGENT_ARTIFACT_MAX_OUTPUT_TOKENS", 2400),
        capability_timeout_s=_f("AGENT_CAPABILITY_TIMEOUT_S", 30),
        evidence_compaction_ratio=_f("AGENT_EVIDENCE_COMPACTION_RATIO", 0.85),
        safety_margin_tokens=_i("AGENT_SAFETY_MARGIN_TOKENS", 128),
        tool_reserve_tokens=_i("AGENT_TOOL_RESERVE_TOKENS", 400),
        summary_target_tokens=_i("AGENT_SUMMARY_TARGET_TOKENS", 700),
        summary_batch_tokens=_i("AGENT_SUMMARY_BATCH_TOKENS", 2500),
        summary_max_depth=_i("AGENT_SUMMARY_MAX_DEPTH", 3),
        web_search_provider=os.getenv("WEB_SEARCH_PROVIDER") or None,
        web_search_url=os.getenv("WEB_SEARCH_URL") or None,
        web_search_api_key=os.getenv("WEB_SEARCH_API_KEY") or None,
        web_allowed_ports=ports,
        web_max_bytes=_i("WEB_MAX_BYTES", 2 * 1024 * 1024),
        web_timeout_s=_f("WEB_TIMEOUT_S", 10),
        web_max_redirects=_i("WEB_MAX_REDIRECTS", 4),
        web_allow_private_hosts=_b("WEB_ALLOW_PRIVATE_HOSTS", "false"),
        approval_ttl_hours=_i("INGEST_APPROVAL_TTL_HOURS", 24),
        web_snapshot_max_chars=_i("WEB_SNAPSHOT_MAX_CHARS", 400_000),
        image_api_url=os.getenv("IMAGE_API_URL") or None,
        image_api_key=os.getenv("IMAGE_API_KEY") or None,
        image_model=os.getenv("IMAGE_MODEL") or None,
        analysis_max_bytes=_i("ANALYSIS_MAX_BYTES", 20 * 1024 * 1024),
    ).validate()
