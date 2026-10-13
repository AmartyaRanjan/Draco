"""Group E - visualization: charts/tables from validated data, image generation via a configured
provider, and a manifest composer that only references verified artifacts."""
import asyncio
import io
import math
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from app.agent import artifacts as A
from app.agent.capabilities._artifact_common import ArtifactRef, TableSpec, build_and_save, resolve_refs
from app.agent.capabilities.common import require
from app.agent.contracts import CapabilityError, ErrorCode, Outcome
from app.agent.registry import Capability


class Series(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    values: list[float] = Field(min_length=1, max_length=200)


class ChartIn(BaseModel):
    output: Literal["chart", "table"] = "chart"
    chart_type: Literal["bar", "hbar", "line", "pie", "scatter"] = "bar"
    title: str = Field(min_length=1, max_length=150)
    x_label: str = Field(default="", max_length=80)
    y_label: str = Field(default="", max_length=80)
    unit: str | None = Field(default=None, max_length=20)
    categories: list[str] = Field(min_length=1, max_length=100)
    series: list[Series] = Field(min_length=1, max_length=8)
    source_refs: list[str] = Field(default=[], max_length=20)


class ChartOut(BaseModel):
    artifact: ArtifactRef | None = None
    table: TableSpec
    title: str
    unit: str | None = None
    x_label: str = ""
    y_label: str = ""
    references: list[str] = []
    summary: str = ""


async def generate_chart_or_table(a: ChartIn, ctx):
    for s in a.series:
        if len(s.values) != len(a.categories):
            raise CapabilityError(ErrorCode.invalid_input, f"Series '{s.name}' has {len(s.values)} values for {len(a.categories)} categories.")
        if not all(math.isfinite(v) for v in s.values):
            raise CapabilityError(ErrorCode.invalid_input, "Series values must be finite numbers.")
    if a.chart_type == "pie" and (len(a.series) != 1 or any(v < 0 for v in a.series[0].values)):
        raise CapabilityError(ErrorCode.invalid_input, "Pie charts need exactly one series of non-negative values.")
    if a.chart_type == "scatter":
        try:
            [float(c) for c in a.categories]
        except ValueError:
            raise CapabilityError(ErrorCode.invalid_input, "Scatter charts need numeric categories (x values).")
    used = resolve_refs(ctx, a.source_refs)
    refs = A.reference_lines(used)
    hdr = [a.x_label or "category"] + [f"{s.name} ({a.unit})" if a.unit else s.name for s in a.series]
    table = TableSpec(columns=hdr, rows=[[c] + [s.values[i] for s in a.series] for i, c in enumerate(a.categories)],
                      caption=a.title)  # the underlying data always travels with the visual
    ref = None
    if a.output == "chart":
        series = [s.model_dump() for s in a.series]
        ref = await build_and_save(ctx, kind="chart", fmt="png", title=a.title,
                                   builder=lambda: A.render_chart(a.chart_type, a.title, a.x_label, a.y_label, a.unit, a.categories, series),
                                   meta={"chart_type": a.chart_type, "refs": [i.ref for i in used]})
    return Outcome(ChartOut(artifact=ref, table=table, title=a.title, unit=a.unit, x_label=a.x_label, y_label=a.y_label,
                            references=refs, summary=f"{a.output}: {len(a.categories)} categories x {len(a.series)} series"),
                   extra={"artifact_id": ref.artifact_id} if ref else None)


# ---- image generation ---------------------------------------------------------------------
class ImageProvider:
    name = "base"

    async def generate(self, prompt: str, size: str) -> bytes:
        raise NotImplementedError


class OpenAICompatImageProvider(ImageProvider):
    """POST {base}/images/generations (response_format=b64_json) - the OpenAI-compatible contract."""
    name = "openai-compatible"

    def __init__(self, base_url, api_key, model, timeout=90, transport=None):
        self.base, self.key, self.model, self.timeout, self.transport = base_url.rstrip("/"), api_key, model, timeout, transport

    async def generate(self, prompt, size):
        import base64
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as c:
            r = await c.post(f"{self.base}/images/generations", headers={"Authorization": f"Bearer {self.key}"},
                             json={"model": self.model, "prompt": prompt, "size": size, "n": 1, "response_format": "b64_json"})
            r.raise_for_status()
        return base64.b64decode(r.json()["data"][0]["b64_json"], validate=True)


def build_image_provider(s) -> ImageProvider | None:
    if s.image_api_url and s.image_api_key and s.image_model:
        return OpenAICompatImageProvider(s.image_api_url, s.image_api_key, s.image_model)
    return None


class ImageIn(BaseModel):
    prompt: str = Field(min_length=3, max_length=1000)
    size: Literal["256x256", "512x512", "1024x1024"] = "1024x1024"


class ImageOut(BaseModel):
    artifact: ArtifactRef
    width: int
    height: int
    summary: str = ""


async def generate_image(a: ImageIn, ctx):
    prov = require(ctx.services.image_provider, "Image generation", "Set IMAGE_API_URL, IMAGE_API_KEY and IMAGE_MODEL.")
    try:
        data = await prov.generate(a.prompt, a.size)
    except Exception as exc:
        raise CapabilityError(ErrorCode.upstream_error, f"Image provider failed ({type(exc).__name__}).", retryable=True) from exc
    try:  # verify the provider really returned an image before storing a reference
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        im.verify()
        w, h = Image.open(io.BytesIO(data)).size
        if im.format != "PNG":
            data = await asyncio.to_thread(lambda: _to_png(data))
    except Exception as exc:
        raise CapabilityError(ErrorCode.upstream_error, "Provider did not return a valid image.") from exc
    ref = await build_and_save(ctx, kind="image", fmt="png", title="generated_image", builder=lambda: data,
                               meta={"prompt_chars": len(a.prompt), "provider": prov.name})
    return Outcome(ImageOut(artifact=ref, width=w, height=h, summary=f"{w}x{h} png"), extra={"artifact_id": ref.artifact_id})


def _to_png(data: bytes) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGBA").save(buf, format="PNG")
    return buf.getvalue()


# ---- compose_multimodal_response ----------------------------------------------------------
class Block(BaseModel):
    type: Literal["text", "table", "chart", "image", "artifact"]
    text: str | None = Field(default=None, max_length=8000)
    table: TableSpec | None = None
    artifact_id: str | None = Field(default=None, max_length=64)
    caption: str | None = Field(default=None, max_length=300)
    source_refs: list[str] = Field(default=[], max_length=20)


class ComposeIn(BaseModel):
    blocks: list[Block] = Field(min_length=1, max_length=40)


class ComposedBlock(BaseModel):
    order: int
    type: str
    text: str | None = None
    table: TableSpec | None = None
    caption: str | None = None
    artifact: ArtifactRef | None = None
    source_refs: list[str] = []


class ComposeOut(BaseModel):
    blocks: list[ComposedBlock]
    artifacts: list[ArtifactRef]
    summary: str = ""


async def compose_multimodal_response(a: ComposeIn, ctx):
    store, out, arts = ctx.services.artifact_store, [], {}
    for i, b in enumerate(a.blocks):
        resolve_refs(ctx, b.source_refs)
        art = None
        if b.type == "text" and not b.text:
            raise CapabilityError(ErrorCode.invalid_input, f"Block {i}: text block needs text.")
        if b.type == "table":
            if not b.table:
                raise CapabilityError(ErrorCode.invalid_input, f"Block {i}: table block needs a table.")
            b.table.check()
        if b.type in ("chart", "image", "artifact"):
            if not b.artifact_id:
                raise CapabilityError(ErrorCode.invalid_input, f"Block {i}: {b.type} block needs artifact_id.")
            require(store, "Artifact storage")
            try:
                rec, _ = store.resolve(b.artifact_id, ctx.user_id)  # verifies owner, expiry and that the file exists
            except A.ArtifactError:
                raise CapabilityError(ErrorCode.not_found, f"Block {i}: artifact {b.artifact_id} not found or not ready.")
            if b.type in ("chart", "image") and rec["fmt"] != "png":
                raise CapabilityError(ErrorCode.invalid_input, f"Block {i}: {b.type} must reference a png artifact.")
            art = ArtifactRef(**A.ref_view(rec))
            arts[art.artifact_id] = art
        out.append(ComposedBlock(order=i, type=b.type, text=b.text, table=b.table, caption=b.caption, artifact=art,
                                 source_refs=b.source_refs))
    return ComposeOut(blocks=out, artifacts=list(arts.values()), summary=f"{len(out)} blocks, {len(arts)} artifacts")


CAPS = [
    Capability("generate_chart_or_table", "visualization", "Make a chart (PNG) and/or table from validated structured data; "
               "labels, units and the underlying data are always returned.", ChartIn, ChartOut, generate_chart_or_table,
               permissions=frozenset({"read", "generate"}), cost="medium", mutates_state=True, parallel_safe=False),
    Capability("generate_image", "visualization", "Generate an image with the configured provider.", ImageIn, ImageOut,
               generate_image, permissions=frozenset({"read", "generate"}), cost="slow", mutates_state=True,
               parallel_safe=False, timeout_s=120),
    Capability("compose_multimodal_response", "visualization", "Assemble ordered text/table/chart/image/artifact blocks into a "
               "response manifest; every artifact is verified to exist.", ComposeIn, ComposeOut, compose_multimodal_response),
]
