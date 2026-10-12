"""Grounded answer generation over an OpenAI-compatible chat endpoint."""
import json
import re
import time
from dataclasses import dataclass, field

import httpx

from app.retrieval.context import BuiltContext, pages
from app.retrieval.schema import RetrievalError

SYSTEM = (
    "You answer strictly from the numbered evidence blocks provided. "
    "Cite every claim with the block label in square brackets, e.g. [S1]. "
    "Only use labels that exist. If the evidence does not contain the answer, say "
    "that the provided documents are insufficient instead of guessing. "
    "Never invent sources, page numbers or facts. Answer in the user's language."
)
WEAK_NOTE = ("Note: no evidence block passed the relevance threshold; the blocks are "
             "low-confidence fallbacks. Say so if they do not answer the question.")


@dataclass
class GenerationResult:
    answer: str
    citations: list[dict]
    invalid_markers: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    model: str | None = None


def build_messages(query: str, ctx: BuiltContext, weak: bool) -> list[dict]:
    evidence = "\n".join(e.text_block for e in ctx.items)
    user = f"Evidence:\n{evidence}\n{WEAK_NOTE if weak else ''}\nQuestion: {query}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def extract_citations(answer: str, ctx: BuiltContext):
    by_label = {e.label: e for e in ctx.items}
    seen, cites, invalid = set(), [], []
    for label in re.findall(r"\[(S\d+)\]", answer):
        if label in seen:
            continue
        seen.add(label)
        e = by_label.get(label)
        if e is None:
            invalid.append(label)
            continue
        c = e.candidate
        cites.append({"label": label, "chunk_id": c.chunk_id, "document_id": c.document_id,
                      "filename": e.filename, "page_start": c.page_start,
                      "page_end": c.page_end, "pages": pages(c), "section_path": c.section_path})
    return cites, invalid


class OpenAICompatGenerator:
    def __init__(self, base_url, api_key, model, timeout_s, max_tokens, temperature):
        self.base_url, self.api_key, self.model = base_url, api_key, model
        self.timeout_s, self.max_tokens, self.temperature = timeout_s, max_tokens, temperature
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def aclose(self):
        if self._client:
            await self._client.aclose()

    async def generate(self, query, ctx: BuiltContext, weak: bool) -> GenerationResult:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"model": self.model, "messages": build_messages(query, ctx, weak),
                   "max_tokens": self.max_tokens, "temperature": self.temperature,
                   "stream": True, "stream_options": {"include_usage": True}}
        t0, first, parts, usage = time.perf_counter(), None, [], {}
        try:
            async with self._http().stream(
                "POST", f"{self.base_url.rstrip('/')}/chat/completions",
                json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    raise RetrievalError(f"Generator returned HTTP {resp.status_code}", "llm", 502)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    usage = chunk.get("usage") or usage
                    for ch in chunk.get("choices", []):
                        delta = (ch.get("delta") or {}).get("content")
                        if delta:
                            first = first or time.perf_counter()
                            parts.append(delta)
        except httpx.TimeoutException as exc:
            raise RetrievalError("Generator timed out", "llm", 504) from exc
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RetrievalError("Generator request failed or returned a malformed response",
                                 "llm", 502) from exc
        end = time.perf_counter()
        answer = "".join(parts).strip()
        if not answer:
            raise RetrievalError("Generator returned an empty response", "llm", 502)
        cites, invalid = extract_citations(answer, ctx)
        return GenerationResult(
            answer, cites, invalid,
            {"llm_total_ms": round((end - t0) * 1e3, 3),
             "llm_ttft_ms": round((first - t0) * 1e3, 3) if first else None,
             "llm_generation_ms": round((end - (first or t0)) * 1e3, 3)},
            {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")},
            self.model)
