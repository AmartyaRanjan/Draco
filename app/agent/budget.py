"""Deterministic context-budget accounting.

Distinguishes the model's context window from the application's evidence budget, keeps explicit
reserves (generation, tool results, safety margin) and says whether counts are exact or estimated.
Compaction is decided *before* the next operation, never in reaction to a context-window error.
"""
import json
from dataclasses import dataclass, field

MESSAGE_OVERHEAD = 6   # <|im_start|>role\n ... <|im_end|>\n  (Qwen ChatML)
PROMPT_OVERHEAD = 8    # assistant generation prompt (+ empty think block when thinking is off)

COMPONENTS = ("system", "tools", "history", "request", "evidence", "evidence_summary", "memory")


@dataclass
class BudgetConfig:
    model_max_len: int           # the model's context window as served
    output_reserve: int          # max generation tokens for the next call
    tool_reserve: int            # room kept for the next tool result
    safety_margin: int           # tokenizer/template slack
    compaction_ratio: float = 0.85
    evidence_budget: int | None = None  # app-chosen cap on evidence+summary tokens (<= available)

    def input_limit(self) -> int:
        return self.model_max_len - self.output_reserve - self.tool_reserve - self.safety_margin


@dataclass
class BudgetDecision:
    ok: bool
    compaction_needed: bool
    used: int
    limit: int
    available: int
    over_by: int


@dataclass
class ContextBudget:
    counter: object
    cfg: BudgetConfig
    used_by: dict = field(default_factory=lambda: {c: 0 for c in COMPONENTS})
    history_log: list = field(default_factory=list)

    # ---- counting -----------------------------------------------------------------------
    def tokens(self, text: str) -> int:
        return self.counter.count(text)

    def message_tokens(self, text: str) -> int:
        return self.tokens(text) + MESSAGE_OVERHEAD

    def tools_tokens(self, specs: list[dict]) -> int:
        return self.tokens(json.dumps(specs, separators=(",", ":"))) if specs else 0

    @property
    def exact(self) -> bool:
        return bool(getattr(self.counter, "exact", False))

    def set(self, component: str, tokens: int):
        if component not in self.used_by:
            raise KeyError(component)
        self.used_by[component] = max(0, tokens)

    def used(self) -> int:
        return sum(self.used_by.values()) + PROMPT_OVERHEAD

    # ---- decisions ----------------------------------------------------------------------
    def evidence_limit(self) -> int:
        """Tokens evidence (+ its summary) may occupy: the app budget, never more than what is
        physically left once every other component is accounted for."""
        other = self.used() - self.used_by["evidence"] - self.used_by["evidence_summary"]
        physical = self.cfg.input_limit() - other
        cap = self.cfg.evidence_budget if self.cfg.evidence_budget is not None else physical
        return max(0, min(cap, physical))

    def check(self, planned_tokens: int = 0) -> BudgetDecision:
        """Would adding `planned_tokens` (next tool output / evidence) fit? Compaction is requested
        when usage would pass `compaction_ratio` of the limit, i.e. before overflow."""
        limit = self.cfg.input_limit()
        used = self.used() + planned_tokens
        ev = self.used_by["evidence"] + self.used_by["evidence_summary"] + planned_tokens
        over_ev = ev > self.cfg.compaction_ratio * self.evidence_limit() if self.evidence_limit() else ev > 0
        needs = used > self.cfg.compaction_ratio * limit or over_ev
        return BudgetDecision(ok=used <= limit and ev <= self.evidence_limit(), compaction_needed=needs,
                              used=used, limit=limit, available=limit - used, over_by=max(0, used - limit))

    def snapshot(self, label: str = "") -> dict:
        d = self.check()
        snap = {"label": label, "components": dict(self.used_by), "used": d.used, "input_limit": d.limit,
                "available": d.available, "evidence_limit": self.evidence_limit(),
                "model_max_len": self.cfg.model_max_len, "output_reserve": self.cfg.output_reserve,
                "tool_reserve": self.cfg.tool_reserve, "safety_margin": self.cfg.safety_margin,
                "token_counts": "exact" if self.exact else "estimate"}
        self.history_log.append(snap)
        return snap
