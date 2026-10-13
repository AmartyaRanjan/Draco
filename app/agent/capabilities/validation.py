"""Group H - validation. validate_citations checks that cited identifiers resolve to real evidence
(and, for private chunks, to the index when an existence check is wired) and measures how well the
cited passages lexically support each sentence. It does NOT establish factual correctness."""
import re

from pydantic import BaseModel, Field

from app.agent.contracts import CapabilityError, ErrorCode
from app.agent.registry import Capability
from app.agent.summarize import _NEG, best_sentence, numbers

MARK = re.compile(r"\[([SW]\d+)\]")
_W = re.compile(r"[\w']+", re.U)
_STOP = frozenset("the a an of to in on for and or is are was were be by with as at that this it from has have had not no "
                  "which who their its also than then there these those can may will would should could into over under".split())
_ABSTAIN = re.compile(r"(insufficient|could not find|cannot find|no (relevant )?evidence|not (contain|provide|mention)|unable to)", re.I)
SUPPORTED, WEAK = 0.6, 0.35


_LBL = r"[SW]\d+"
_RNG = rf"{_LBL}(?:\s*(?:-|–|—|to)\s*[SW]?\d+)?"
_GROUP = re.compile(rf"[\[(【]\s*\**\s*({_RNG}(?:\s*(?:,|;|&|and)\s*{_RNG})*)\s*\**\s*[\])】]")


def normalize_citations(text: str) -> str:
    """Rewrite common model citation styles ([**S1**], [S1, S2], [S1-S3], (S1)) to canonical [S1][S2][S3]
    so identifiers can be checked. Expanded ranges that name non-existent labels are then reported as invalid."""
    def expand(m):
        labels = []
        for part in re.split(r"\s*(?:,|;|&|\band\b)\s*", m.group(1)):
            r = re.fullmatch(rf"([SW])(\d+)\s*(?:-|–|—|to)\s*[SW]?(\d+)", part.strip())
            if r and int(r[2]) <= int(r[3]) <= int(r[2]) + 9:
                labels += [f"{r[1]}{i}" for i in range(int(r[2]), int(r[3]) + 1)]
            elif r:
                labels += [f"{r[1]}{r[2]}", f"{r[1]}{r[3]}"]
            else:
                labels.append(part.strip())
        return "".join(f"[{x}]" for x in labels)
    return _GROUP.sub(expand, text)


class ValidateIn(BaseModel):
    answer: str = Field(min_length=1, max_length=12000)
    user_context: str | None = Field(default=None, max_length=8000, description="user request / tool results whose numbers are legitimate (set by the orchestrator)")
    max_sentences: int = Field(default=60, ge=1, le=200)
    mode: str = Field(default="strict", pattern="^(strict|tool_task)$",
                      description="tool_task: replies that describe produced files mix source facts with operational text, so lexical-only gaps are 'weak' (identifiers, numbers and negations are still enforced)")


class ClaimCheck(BaseModel):
    sentence: str
    refs: list[str]
    verdict: str  # supported | weak | unsupported | invalid_reference | abstention
    support: float | None = None
    reason: str | None = None


class ValidateOut(BaseModel):
    status: str  # pass | warn | fail
    invalid_refs: list[str]
    claims: list[ClaimCheck]
    uncited_sentences: list[str]
    coverage: dict
    method: str = "identifier_resolution + lexical_overlap"
    note: str = "Checks that identifiers exist and cited text overlaps the claim; it does not prove the claim is true."
    summary: str = ""


def _content(text: str) -> set[str]:
    return {w.lower().rstrip("s") for w in _W.findall(MARK.sub("", text)) if len(w) > 2 and w.lower() not in _STOP}


def split_answer(answer: str) -> list[str]:
    out = []
    for line in re.split(r"\n+", answer):
        line = re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", line)
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[\"'])", line) if s.strip()]
    return out


async def validate_citations(a: ValidateIn, ctx):
    store, svc = ctx.evidence, ctx.services
    sents = split_answer(a.answer)[: a.max_sentences]
    invalid, checks, uncited = [], [], []
    existence_cache: dict[str, bool] = {}
    for s in sents:
        refs = list(dict.fromkeys(MARK.findall(s)))
        if not refs:
            if len(_W.findall(s)) >= 6 and not s.rstrip().endswith("?") and not _ABSTAIN.search(s):
                uncited.append(s[:200])
            continue
        bad = []
        texts = []
        for r in refs:
            it = store.get(r)
            ok = it is not None and bool(it.text)
            if ok and it.kind == "private" and svc is not None and svc.chunk_exists is not None:
                key = it.source_key()
                if key not in existence_cache:
                    try:
                        existence_cache[key] = bool(await svc.chunk_exists(it.document_id, it.chunk_id))
                    except Exception:
                        raise CapabilityError(ErrorCode.upstream_error, "Chunk existence check failed.", retryable=True)
                ok = existence_cache[key]
            if not ok:
                bad.append(r)
                if r not in invalid:
                    invalid.append(r)
            else:
                texts.append(it.text)
        if bad and not texts:
            checks.append(ClaimCheck(sentence=s[:300], refs=refs, verdict="invalid_reference", reason=f"unresolved: {bad}"))
            continue
        if _ABSTAIN.search(s) and not numbers(MARK.sub("", s)):   # "the sources do not say X": a decline, not a factual claim
            checks.append(ClaimCheck(sentence=s[:300], refs=refs, verdict="abstention", reason="decline/insufficiency statement"))
            continue
        cw, src = _content(s), "\n".join(texts)
        sw = _content(src)
        support = len(cw & sw) / len(cw) if cw else 1.0
        low = "weak" if a.mode == "tool_task" else "unsupported"
        verdict, reason = ("supported" if support >= SUPPORTED else "weak" if support >= WEAK else low), None
        miss = numbers(MARK.sub("", s)) - numbers(src) - (numbers(a.user_context) if a.user_context else set())
        if miss:
            verdict, reason = "unsupported", f"numbers not in cited text: {sorted(miss)[:3]}"
        elif verdict != "unsupported":
            best, overlap = best_sentence(MARK.sub("", s), texts)
            if overlap >= 0.5 and bool(_NEG.search(MARK.sub("", s))) != bool(_NEG.search(best)):
                verdict, reason = "weak", "negation differs from the best-matching source sentence"
        if svc is not None and svc.entailment is not None and verdict != "unsupported":
            try:
                lab = await svc.entailment(MARK.sub("", s), src)
                if lab == "contradicted":
                    verdict, reason = "unsupported", "validator model: contradicted"
            except Exception:
                pass
        checks.append(ClaimCheck(sentence=s[:300], refs=refs, verdict=verdict, support=round(support, 2), reason=reason))
    n_abst = sum(c.verdict == "abstention" for c in checks)
    n_cited = len(checks) - n_abst
    n_sup = sum(c.verdict == "supported" for c in checks)
    factual = n_cited + len(uncited)
    cov = {"sentences": len(sents), "cited_sentences": n_cited, "abstention_sentences": n_abst, "uncited_factual_sentences": len(uncited),
           "citation_coverage": round(n_cited / factual, 3) if factual else 1.0,
           "support_rate": round(n_sup / n_cited, 3) if n_cited else None,
           "supported": n_sup, "weak": sum(c.verdict == "weak" for c in checks),
           "unsupported": sum(c.verdict in ("unsupported", "invalid_reference") for c in checks)}
    status = "fail" if invalid or cov["unsupported"] else "warn" if cov["weak"] or uncited else "pass"
    return ValidateOut(status=status, invalid_refs=invalid, claims=checks, uncited_sentences=uncited, coverage=cov,
                       summary=f"{status}: {n_sup}/{n_cited} supported, {len(invalid)} invalid refs")


class FeedbackIn(BaseModel):
    target_type: str = Field(pattern="^(answer|source|artifact)$")
    target_id: str = Field(min_length=1, max_length=128)
    rating: int = Field(ge=-1, le=1, description="-1 negative, 0 neutral, 1 positive")
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackOut(BaseModel):
    feedback_id: str
    stored: bool = True
    authorizes_ingestion: bool = False  # feedback never authorizes writes to the knowledge base
    summary: str = ""


async def record_feedback(a: FeedbackIn, ctx):
    store = ctx.services.store
    if a.target_type == "artifact" and store.get_artifact(a.target_id, ctx.user_id) is None:
        raise CapabilityError(ErrorCode.not_found, "Artifact not found.")
    if a.target_type == "source" and store.get_source(a.target_id, ctx.user_id) is None:
        raise CapabilityError(ErrorCode.not_found, "Source not found.")
    fid = store.add_feedback(ctx.user_id, a.target_type, a.target_id, a.rating, (a.comment or "").strip() or None)
    return FeedbackOut(feedback_id=fid, summary=f"feedback recorded for {a.target_type}")


CAPS = [
    Capability("validate_citations", "validation", "Check an answer's [S#]/[W#] citations resolve to real evidence and are "
               "lexically supported.", ValidateIn, ValidateOut, validate_citations, cost="fast", llm_exposed=False),
    Capability("record_feedback", "validation", "Record user feedback on an answer, source or artifact. Never authorizes ingestion.",
               FeedbackIn, FeedbackOut, record_feedback, permissions=frozenset({"read", "write"}), mutates_state=True),
]
