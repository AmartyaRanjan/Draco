"""Agent tools. The orchestrator validates arguments, enforces limits and writes the trace."""
import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from pydantic import BaseModel, Field

from app.db import registry


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="Specific search query")
    document_ids: list[str] | None = Field(default=None, max_length=20,
                                           description="Optional document ids to restrict the search")


class ListArgs(BaseModel):
    pass


@dataclass
class ToolContext:
    user_id: str
    search: Callable[[str, list[str] | None], Awaitable[tuple[str, str]]]  # (evidence text, summary)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    purpose: str
    args: type[BaseModel]
    run: Callable[[BaseModel, ToolContext], Awaitable[tuple[str, str]]]  # (full output, summary)


async def _search(args: SearchArgs, ctx: ToolContext) -> tuple[str, str]:
    return await ctx.search(args.query, args.document_ids)


async def _list(args: ListArgs, ctx: ToolContext) -> tuple[str, str]:
    rows = await asyncio.to_thread(registry.list_documents_for_owner, ctx.user_id, 50)
    text = "\n".join(f"{r['document_id']} | {r['filename']} | {r['status']}" for r in rows) or "No documents."
    return text, f"Listed {len(rows)} documents"


TOOLS: dict[str, Tool] = {t.name: t for t in (
    Tool("search_documents", "Search the indexed documents and return labelled evidence blocks [S#].",
         "Retrieve evidence from indexed documents", SearchArgs, _search),
    Tool("list_documents", "List the indexed documents (id, filename, status).",
         "Discover which documents exist", ListArgs, _list),
)}


def tool_specs() -> list[dict]:
    """OpenAI-style function specs, generated from the validation models."""
    return [{"type": "function", "function": {
        "name": t.name, "description": t.description, "parameters": t.args.model_json_schema()}}
        for t in TOOLS.values()]
