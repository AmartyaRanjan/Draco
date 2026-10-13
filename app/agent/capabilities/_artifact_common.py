import asyncio

from pydantic import BaseModel, Field

from app.agent import artifacts as A
from app.agent.capabilities.common import require
from app.agent.contracts import CapabilityError, ErrorCode

Cell = str | int | float | bool | None


class TableSpec(BaseModel):
    columns: list[str] = Field(min_length=1, max_length=20)
    rows: list[list[Cell]] = Field(max_length=200)
    caption: str | None = Field(default=None, max_length=200)

    def check(self):
        for i, r in enumerate(self.rows):
            if len(r) != len(self.columns):
                raise CapabilityError(ErrorCode.invalid_input, f"Row {i} has {len(r)} cells; expected {len(self.columns)}.")
        return self


class ArtifactRef(BaseModel):
    artifact_id: str
    kind: str
    format: str
    filename: str
    size_bytes: int
    sha256: str
    status: str
    download_path: str
    expires_at: float


def resolve_refs(ctx, refs: list[str]):
    """Source refs in artifact specs must exist in the evidence store; the reference list is built
    from stored provenance, never from model-written text."""
    items, unknown = [], []
    for r in dict.fromkeys(refs):
        it = ctx.evidence.get(r)
        (items if it else unknown).append(it or r)
    if unknown:
        raise CapabilityError(ErrorCode.invalid_input, f"Unknown source refs: {unknown[:6]}")
    return items


async def build_and_save(ctx, *, kind, fmt, title, builder, meta=None) -> ArtifactRef:
    store = require(ctx.services.artifact_store, "Artifact storage")
    try:
        data = await asyncio.to_thread(builder)
    except CapabilityError:
        raise
    except Exception as exc:
        raise CapabilityError(ErrorCode.internal, f"{kind} generation failed ({type(exc).__name__}).") from exc
    try:
        rec = await asyncio.to_thread(store.save, user_id=ctx.user_id, kind=kind, fmt=fmt, title=title, data=data, meta=meta)
    except A.ArtifactError as exc:
        code = ErrorCode.limit_exceeded if exc.code == "limit_exceeded" else ErrorCode.internal
        raise CapabilityError(code, str(exc)) from exc
    return ArtifactRef(**A.ref_view(rec))
