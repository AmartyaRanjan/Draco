import re
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


_OVERFLOW = re.compile(r"maximum context length is \d+ tokens.*?requested \d+ output tokens and your prompt contains", re.S)


def reduced_output_cap(exc: BaseException, current: int, floor: int = 256, factor: float = 0.7) -> int | None:
    """If the server rejected the request only because prompt + max_tokens overran the window, a smaller max_tokens to try
    (our token estimate cannot see chat-template and tool-schema overhead). The server's "at least N input tokens" is
    just window + 1 - max_tokens, a bound that says nothing about the real prompt size, so shrink geometrically instead
    of parsing it. None if it is some other error or the cap would fall below `floor`."""
    if not _OVERFLOW.search(str(exc)):
        return None
    smaller = int(current * factor)
    return smaller if smaller >= floor else None


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
