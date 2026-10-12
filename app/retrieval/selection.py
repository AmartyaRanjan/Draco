"""Adaptive relevance selection: threshold + minimum-evidence fallback."""
from dataclasses import dataclass

from app.retrieval.schema import Candidate

PASSED = "passed_threshold"
FALLBACK = "fallback_below_threshold"


@dataclass(slots=True)
class SelectionResult:
    selected: list[Candidate]
    passed: int
    fallback: int
    evidence_sufficient: bool  # at least one candidate passed the threshold
    basis: str  # "calibrated_probability" | "raw_colbert_score"


class Calibrator:
    """Optional monotone piecewise-linear raw score -> probability map."""

    def __init__(self, points: list[tuple[float, float]]):
        self.points = points

    def __call__(self, s: float) -> float:
        pts = self.points
        if s <= pts[0][0]:
            return pts[0][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if s <= x1:
                return y0 if x1 == x0 else y0 + (y1 - y0) * (s - x0) / (x1 - x0)
        return pts[-1][1]


def select(ranked: list[Candidate], threshold: float, min_selected: int, max_selected: int,
           calibrator: Calibrator | None = None) -> SelectionResult:
    """`ranked` must be sorted by descending ColBERT score. Candidates without a
    score or text are invalid and never selected. Threshold applies to the
    calibrated probability when a calibrator exists, otherwise to the raw score."""
    valid, seen = [], set()
    for c in ranked:
        if c.colbert_score is None or not c.text or c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        if calibrator:
            c.calibrated_probability = calibrator(c.colbert_score)
        valid.append(c)
    key = (lambda c: c.calibrated_probability) if calibrator else (lambda c: c.colbert_score)
    chosen = []
    for c in valid:
        if key(c) >= threshold and len(chosen) < max_selected:
            c.selection = PASSED
            chosen.append(c)
    passed = len(chosen)
    for c in valid:
        if len(chosen) >= min(min_selected, max_selected):
            break
        if c.selection is None:
            c.selection = FALLBACK
            chosen.append(c)
    return SelectionResult(chosen, passed, len(chosen) - passed, passed > 0,
                           "calibrated_probability" if calibrator else "raw_colbert_score")
