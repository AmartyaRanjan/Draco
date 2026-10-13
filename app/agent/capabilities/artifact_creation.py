"""Group D - artifact_creation. Content is supplied as a validated spec (the model drafts it from
evidence); rendering is deterministic. Reference lists come from the evidence store."""
from typing import Literal

from pydantic import BaseModel, Field

from app.agent import artifacts as A
from app.agent.capabilities._artifact_common import ArtifactRef, TableSpec, build_and_save, resolve_refs
from app.agent.contracts import CapabilityError, ErrorCode, Outcome
from app.agent.registry import Capability

Refs = Field(default=[], max_length=20, description="evidence refs (S#/W#) supporting this part")


class Section(BaseModel):
    heading: str = Field(min_length=1, max_length=200)
    level: int = Field(default=1, ge=1, le=3)
    paragraphs: list[str] = Field(default=[], max_length=20)
    bullets: list[str] = Field(default=[], max_length=30)
    table: TableSpec | None = None
    source_refs: list[str] = Refs


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    format: Literal["docx", "pdf"] = "docx"
    sections: list[Section] = Field(min_length=1, max_length=40)
    include_references: bool = True


class ArtifactOut(BaseModel):
    artifact: ArtifactRef
    references: list[str] = []
    summary: str = ""


def _clip(sections):
    for s in sections:
        if any(len(p) > 4000 for p in s.paragraphs) or any(len(b) > 500 for b in s.bullets):
            raise CapabilityError(ErrorCode.invalid_input, "A paragraph (max 4000 chars) or bullet (max 500) is too long.")
        if s.table:
            s.table.check()


async def generate_document(a: DocumentIn, ctx):
    _clip(a.sections)
    used = resolve_refs(ctx, [r for s in a.sections for r in s.source_refs])
    refs = A.reference_lines(used) if a.include_references else []
    secs = [s.model_dump() for s in a.sections]
    render = A.render_docx if a.format == "docx" else A.render_pdf
    ref = await build_and_save(ctx, kind="document", fmt=a.format, title=a.title, builder=lambda: render(a.title, secs, refs),
                               meta={"sections": len(secs), "refs": [i.ref for i in used]})
    return Outcome(ArtifactOut(artifact=ref, references=refs, summary=f"{a.format} {ref.size_bytes} bytes"),
                   extra={"artifact_id": ref.artifact_id})


class Slide(BaseModel):
    title: str = Field(min_length=1, max_length=150)
    bullets: list[str] = Field(default=[], max_length=8)
    notes: str | None = Field(default=None, max_length=2000)
    table: TableSpec | None = None
    chart_artifact_id: str | None = Field(default=None, max_length=64)
    source_refs: list[str] = Refs


class PresentationIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    slides: list[Slide] = Field(min_length=1, max_length=40)
    include_sources_slide: bool = True


async def generate_presentation(a: PresentationIn, ctx):
    images = {}
    for s in a.slides:
        if any(len(b) > 200 for b in s.bullets):
            raise CapabilityError(ErrorCode.invalid_input, "Slide bullets must be <= 200 chars.")
        if s.table:
            s.table.check()
            if len(s.table.rows) > 12:
                raise CapabilityError(ErrorCode.invalid_input, "Slide tables are limited to 12 rows.")
        if s.chart_artifact_id:
            store = ctx.services.artifact_store
            if store is None:
                raise CapabilityError(ErrorCode.not_configured, "Artifact storage is not configured.")
            try:
                rec, _ = store.resolve(s.chart_artifact_id, ctx.user_id)
                if rec["fmt"] != "png":
                    raise A.ArtifactError("not an image", "invalid")
                images[s.chart_artifact_id] = store.read_bytes(s.chart_artifact_id, ctx.user_id)
            except A.ArtifactError:
                raise CapabilityError(ErrorCode.not_found, f"Chart artifact {s.chart_artifact_id} not found.")
    used = resolve_refs(ctx, [r for s in a.slides for r in s.source_refs])
    refs = A.reference_lines(used) if a.include_sources_slide else []
    slides = [s.model_dump() for s in a.slides]
    ref = await build_and_save(ctx, kind="presentation", fmt="pptx", title=a.title,
                               builder=lambda: A.render_pptx(a.title, slides, refs, images),
                               meta={"slides": len(slides), "refs": [i.ref for i in used]})
    return Outcome(ArtifactOut(artifact=ref, references=refs, summary=f"pptx {len(slides) + 1} slides"),
                   extra={"artifact_id": ref.artifact_id})


class FormulaSpec(BaseModel):
    cell: str = Field(max_length=10)
    formula: str = Field(max_length=200)


class SheetSpec(BaseModel):
    name: str = Field(min_length=1, max_length=31, pattern=r"^[^\[\]:*?/\\]+$")
    columns: list[str] = Field(min_length=1, max_length=40)
    rows: list[list[str | int | float | bool | None]] = Field(max_length=5000)
    formulas: list[FormulaSpec] = Field(default=[], max_length=50)


class SpreadsheetIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    format: Literal["xlsx", "csv"] = "xlsx"
    sheets: list[SheetSpec] = Field(min_length=1, max_length=10)
    source_refs: list[str] = Refs


async def generate_spreadsheet(a: SpreadsheetIn, ctx):
    names = [s.name.lower() for s in a.sheets]
    if len(set(names)) != len(names):
        raise CapabilityError(ErrorCode.invalid_input, "Sheet names must be unique.")
    for s in a.sheets:
        for i, r in enumerate(s.rows):
            if len(r) != len(s.columns):
                raise CapabilityError(ErrorCode.invalid_input, f"Sheet '{s.name}' row {i} has {len(r)} cells; expected {len(s.columns)}.")
        for f in s.formulas:
            p = A.formula_problem(f.cell, f.formula)
            if p:
                raise CapabilityError(ErrorCode.invalid_input, f"Sheet '{s.name}' formula {f.cell}: {p}")
    if a.format == "csv":
        if len(a.sheets) != 1 or a.sheets[0].formulas:
            raise CapabilityError(ErrorCode.invalid_input, "CSV supports exactly one sheet and no formulas; use xlsx.")
    used = resolve_refs(ctx, a.source_refs)
    refs = A.reference_lines(used)
    sheets = [s.model_dump() for s in a.sheets]
    builder = (lambda: A.render_csv(sheets[0]["columns"], sheets[0]["rows"])) if a.format == "csv" \
        else (lambda: A.render_xlsx(sheets, refs))
    ref = await build_and_save(ctx, kind="spreadsheet", fmt=a.format, title=a.title, builder=builder,
                               meta={"sheets": len(sheets), "refs": [i.ref for i in used]})
    return Outcome(ArtifactOut(artifact=ref, references=refs, summary=f"{a.format} {ref.size_bytes} bytes"),
                   extra={"artifact_id": ref.artifact_id})


CAPS = [
    Capability("generate_document", "artifact_creation", "Create a DOCX or PDF document from structured sections, tables and source refs.",
               DocumentIn, ArtifactOut, generate_document, permissions=frozenset({"read", "generate"}), cost="slow",
               mutates_state=True, parallel_safe=False, timeout_s=60),
    Capability("generate_presentation", "artifact_creation", "Create a PPTX deck from slides (bullets, tables, chart images, notes).",
               PresentationIn, ArtifactOut, generate_presentation, permissions=frozenset({"read", "generate"}), cost="slow",
               mutates_state=True, parallel_safe=False, timeout_s=60),
    Capability("generate_spreadsheet", "artifact_creation", "Create an XLSX (with safe formulas) or CSV from validated tabular data.",
               SpreadsheetIn, ArtifactOut, generate_spreadsheet, permissions=frozenset({"read", "generate"}), cost="slow",
               mutates_state=True, parallel_safe=False, timeout_s=60),
]
