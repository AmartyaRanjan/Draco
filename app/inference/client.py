"""LangChain ChatOpenAI client for the vLLM OpenAI-compatible server."""
import openai
from langchain_openai import ChatOpenAI

from app.inference.config import InferenceSettings
from app.inference.schemas import InferenceError


def build_llm(s: InferenceSettings, *, thinking: bool, max_tokens: int | None = None,
              streaming: bool = False) -> ChatOpenAI:
    """One instance per call profile. Thinking is a per-request chat-template flag
    (Qwen3 `enable_thinking`); vLLM must run with `--reasoning-parser qwen3` so thinking
    is returned in a separate field and never reaches `content`."""
    return ChatOpenAI(
        model=s.model,
        base_url=s.base_url,
        api_key=s.api_key or "EMPTY",
        timeout=s.timeout_s,
        max_retries=s.max_retries,
        temperature=s.reasoning_temperature if thinking else s.grounded_temperature,
        top_p=s.reasoning_top_p if thinking else s.grounded_top_p,
        max_tokens=min(max_tokens or s.max_output_tokens, s.max_output_tokens),
        streaming=streaming,
        stream_usage=True,
        extra_body={"top_k": s.top_k, "chat_template_kwargs": {"enable_thinking": thinking}},
    )


def map_error(exc: BaseException) -> InferenceError:
    """Translate SDK failures into client-safe errors (no internals leaked)."""
    if isinstance(exc, InferenceError):
        return exc
    if isinstance(exc, openai.APITimeoutError):
        return InferenceError("The model timed out", "llm_timeout", 504)
    if isinstance(exc, openai.AuthenticationError):
        return InferenceError("Inference service rejected credentials", "llm_auth", 502)
    if isinstance(exc, openai.BadRequestError):
        return InferenceError("The model rejected the request (prompt may exceed the context window)",
                              "llm_bad_request", 502)
    if isinstance(exc, (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError)):
        return InferenceError("Inference service unavailable", "llm_unavailable", 503)
    return InferenceError("Generation failed", "llm_error", 502)
