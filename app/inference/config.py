import os
from dataclasses import dataclass
from functools import lru_cache


def _b(name, default):
    return os.getenv(name, default).lower() == "true"


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    base_url: str
    api_key: str
    model: str
    timeout_s: float
    max_retries: int
    max_model_len: int
    max_output_tokens: int  # per generation call
    # Sampling profiles. Qwen3 guidance: non-thinking T~0.7/top_p 0.8, thinking T~0.6/top_p 0.95, top_k 20.
    grounded_temperature: float
    grounded_top_p: float
    reasoning_temperature: float
    reasoning_top_p: float
    top_k: int
    # Agent limits
    agent_max_steps: int
    agent_max_tool_calls: int
    agent_workflow_token_budget: int  # total completion tokens across all calls in one request
    tool_timeout_s: float
    tool_output_tokens: int
    # Prompt budgets (tokens)
    history_tokens: int
    memory_tokens: int
    prompt_overhead_tokens: int
    # Auth / persistence
    api_keys: dict  # key -> user_id
    auth_disabled: bool
    store_path: str | None

    def evidence_budget(self) -> int:
        """Input room left for document evidence after fixed sections and the output reserve."""
        return (self.max_model_len - self.max_output_tokens - self.history_tokens
                - self.memory_tokens - self.prompt_overhead_tokens)

    def validate(self):
        if not self.base_url or not self.model:
            raise ValueError("INFERENCE_BASE_URL and INFERENCE_MODEL are required.")
        if not 1 <= self.max_output_tokens <= 4096:
            raise ValueError("INFERENCE_MAX_OUTPUT_TOKENS must be in [1, 4096].")
        if self.evidence_budget() < 512:
            raise ValueError("Prompt budgets leave < 512 tokens for evidence; raise INFERENCE_MAX_MODEL_LEN.")
        if self.agent_max_steps < 1 or self.agent_max_tool_calls < 0 or self.max_retries < 0:
            raise ValueError("Agent/retry limits must be non-negative (steps >= 1).")
        if not self.auth_disabled and not self.api_keys:
            raise ValueError("Set APP_API_KEYS ('key:user,...') or APP_AUTH_DISABLED=true (development only).")
        return self


def _keys(raw: str) -> dict:
    out = {}
    for part in filter(None, (p.strip() for p in raw.split(","))):
        key, _, user = part.partition(":")
        if key and user:
            out[key] = user
    return out


@lru_cache
def get_inference_settings() -> InferenceSettings:
    f, i = float, int
    return InferenceSettings(
        base_url=os.getenv("INFERENCE_BASE_URL", "http://127.0.0.1:18001/v1"),
        api_key=os.getenv("INFERENCE_API_KEY", ""),
        model=os.getenv("INFERENCE_MODEL", "Qwen/Qwen3-4B-FP8"),
        timeout_s=f(os.getenv("INFERENCE_TIMEOUT_S", "90")),
        max_retries=i(os.getenv("INFERENCE_MAX_RETRIES", "1")),
        max_model_len=i(os.getenv("INFERENCE_MAX_MODEL_LEN", "8192")),
        max_output_tokens=i(os.getenv("INFERENCE_MAX_OUTPUT_TOKENS", "1000")),
        grounded_temperature=f(os.getenv("INFERENCE_GROUNDED_TEMPERATURE", "0.3")),
        grounded_top_p=f(os.getenv("INFERENCE_GROUNDED_TOP_P", "0.8")),
        reasoning_temperature=f(os.getenv("INFERENCE_REASONING_TEMPERATURE", "0.6")),
        reasoning_top_p=f(os.getenv("INFERENCE_REASONING_TOP_P", "0.95")),
        top_k=i(os.getenv("INFERENCE_TOP_K", "20")),
        agent_max_steps=i(os.getenv("AGENT_MAX_STEPS", "4")),
        agent_max_tool_calls=i(os.getenv("AGENT_MAX_TOOL_CALLS", "4")),
        agent_workflow_token_budget=i(os.getenv("AGENT_WORKFLOW_TOKEN_BUDGET", "3000")),
        tool_timeout_s=f(os.getenv("AGENT_TOOL_TIMEOUT_S", "30")),
        tool_output_tokens=i(os.getenv("AGENT_TOOL_OUTPUT_TOKENS", "2500")),
        history_tokens=i(os.getenv("INFERENCE_HISTORY_TOKENS", "800")),
        memory_tokens=i(os.getenv("INFERENCE_MEMORY_TOKENS", "400")),
        prompt_overhead_tokens=i(os.getenv("INFERENCE_PROMPT_OVERHEAD_TOKENS", "600")),
        api_keys=_keys(os.getenv("APP_API_KEYS", "")),
        auth_disabled=_b("APP_AUTH_DISABLED", "false"),
        store_path=os.getenv("INFERENCE_STORE_PATH") or None,
    ).validate()
