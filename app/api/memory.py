"""Long-term memory management. Every route acts only on the authenticated user's own items.

Needs MEMORY_BACKEND=hybrid; with the default `null` backend the routes answer 501 so clients can tell
"disabled" from "empty"."""
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.api.chat import get_orchestrator, heavy_user, read_user
from app.inference.generation import ChatOrchestrator
from app.inference.memory import HybridMemory

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1000)
    kind: Literal["fact", "preference"] = "fact"


def _memory(orch: Annotated[ChatOrchestrator, Depends(get_orchestrator)]) -> HybridMemory:
    memory = orch.memory
    if not isinstance(memory, HybridMemory):
        raise HTTPException(501, "Long-term memory is disabled (set MEMORY_BACKEND=hybrid)")
    return memory


Memory = Annotated[HybridMemory, Depends(_memory)]
ReadUser = Annotated[str, Depends(read_user)]
HeavyUser = Annotated[str, Depends(heavy_user)]


def _item(m) -> dict:
    return {"memory_id": m.memory_id, "text": m.text, "kind": m.kind}


@router.get("")
async def list_memories(user: ReadUser, memory: Memory, limit: int = 100):
    items = await memory.list(user, max(1, min(limit, 500)))
    return {"items": [_item(m) for m in items]}


@router.post("", status_code=201)
async def add_memory(body: MemoryCreate, user: HeavyUser, memory: Memory):
    try:
        item = await memory.add(user, body.text, body.kind)
    except ValueError as exc:  # empty, too long, bad kind, per-user limit
        raise HTTPException(422, str(exc)) from exc
    return _item(item)


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(memory_id: str, user: ReadUser, memory: Memory):
    await memory.delete(user, memory_id)


@router.delete("", status_code=204)
async def erase_memories(user: ReadUser, memory: Memory):
    await memory.erase(user)
