"""Model-runtime adapter: logical reasoning modes -> checkpoint-specific request parameters.

`set_reasoning_mode` is a control-plane operation owned by the orchestrator, not an LLM tool.
Facts this relies on (verified from the checkpoint files, see tools.md):
  * Qwen3 hybrid checkpoints (Qwen3-4B/8B, -FP8/-AWQ) ship a chat template that reads the Jinja
    variable `enable_thinking`; vLLM forwards `chat_template_kwargs` per request, so no restart.
  * Qwen3-*-Instruct-2507 is non-thinking only; Qwen3-*-Thinking-2507 always thinks.
There is no generic `thinking=true` API argument.
"""
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger("rag.agent")


class ReasoningMode(str, Enum):
    thinking = "thinking"
    non_thinking = "non_thinking"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    pattern: str
    supports_toggle: bool
    fixed_mode: ReasoningMode | None = None  # set when the checkpoint cannot switch
    template_flag: str | None = "enable_thinking"

    def matches(self, model_id: str) -> bool:
        return re.search(self.pattern, model_id, re.I) is not None


# Order matters: the 2507 variants must match before the generic hybrid pattern.
PROFILES = (
    ModelProfile("qwen3-instruct-2507", r"Qwen3-.*Instruct-2507", False, ReasoningMode.non_thinking, None),
    ModelProfile("qwen3-thinking-2507", r"Qwen3-.*Thinking-2507", False, ReasoningMode.thinking, None),
    ModelProfile("qwen3-hybrid", r"Qwen3-(0\.6|1\.7|4|8|14|32)B(?!.*2507)", True),
)


def profile_for(model_id: str) -> ModelProfile | None:
    return next((p for p in PROFILES if p.matches(model_id)), None)


@dataclass
class RuntimeCapabilities:
    model: str
    profile: str | None
    supports_toggle: bool
    fixed_mode: str | None
    template_declares_flag: bool | None  # None: template not inspectable offline
    served_max_model_len: int | None = None
    probe_verified: bool | None = None  # None: live probe not run
    notes: list[str] = field(default_factory=list)

    def supports(self, mode: ReasoningMode) -> bool:
        if self.profile is None:
            return False
        if self.fixed_mode:
            return mode.value == self.fixed_mode
        return self.supports_toggle and self.template_declares_flag is not False


def _template_declares(model_id: str, flag: str | None) -> bool | None:
    """Read the checkpoint's chat template from the local HF cache (no network)."""
    if not flag:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(model_id, "tokenizer_config.json")
        if not isinstance(p, (str, Path)):
            return None
        import json
        tpl = json.loads(Path(p).read_text()).get("chat_template") or ""
        if isinstance(tpl, list):
            tpl = " ".join(t.get("template", "") for t in tpl)
        return flag in tpl
    except Exception:
        return None


def detect_capabilities(model_id: str, served_max_model_len: int | None = None) -> RuntimeCapabilities:
    prof = profile_for(model_id)
    if prof is None:
        return RuntimeCapabilities(model_id, None, False, None, None, served_max_model_len,
                                   notes=["no model profile: reasoning mode cannot be controlled"])
    declares = _template_declares(model_id, prof.template_flag)
    notes = []
    if prof.supports_toggle and declares is None:
        notes.append("chat template not inspectable offline; relying on the model profile")
    if prof.supports_toggle and declares is False:
        notes.append("chat template does not reference the thinking flag; toggle disabled")
    return RuntimeCapabilities(model_id, prof.name, prof.supports_toggle, prof.fixed_mode.value if prof.fixed_mode else None,
                               declares, served_max_model_len, notes=notes)


@dataclass
class ModeDecision:
    """requested = what the policy asked for; applied = what the adapter actually put on the
    request. `applied` stays None until `RuntimeAdapter.build_llm` has built a client from it."""
    requested: ReasoningMode
    reason: str
    planned: ReasoningMode | None
    status: str  # ok | fallback_fixed_mode | unsupported | denied_mode_change_limit
    config: dict = field(default_factory=dict)  # chat_template_kwargs that will be sent
    applied: ReasoningMode | None = None
    applied_config: dict | None = None
    change_index: int = 0

    def as_dict(self) -> dict:
        return {"requested": self.requested.value, "planned": self.planned.value if self.planned else None,
                "applied": self.applied.value if self.applied else None, "status": self.status,
                "reason": self.reason, "config": self.applied_config if self.applied else self.config,
                "change_index": self.change_index}


class ReasoningController:
    """Per-task mode state with a bounded number of changes."""

    def __init__(self, caps: RuntimeCapabilities, default: ReasoningMode = ReasoningMode.non_thinking,
                 max_changes: int = 2):
        self.caps, self.max_changes = caps, max_changes
        self.current: ReasoningMode | None = None
        self.changes = 0
        self.log: list[ModeDecision] = []
        self._default = default
        self._prof = profile_for(caps.model)

    def request(self, mode: ReasoningMode | str, reason: str) -> ModeDecision:
        mode = ReasoningMode(mode)  # validates the enum
        first = self.current is None
        if not first and mode != self.current and self.changes >= self.max_changes:
            d = ModeDecision(mode, reason, self.current, "denied_mode_change_limit",
                             self._kwargs(self.current), change_index=self.changes)
            self.log.append(d)
            return d
        if self._prof is None:
            d = ModeDecision(mode, reason, None, "unsupported")
        elif self.caps.fixed_mode:
            fixed = ReasoningMode(self.caps.fixed_mode)
            d = ModeDecision(mode, reason, fixed, "ok" if fixed == mode else "fallback_fixed_mode", {})
        elif not self.caps.supports(mode):
            d = ModeDecision(mode, reason, None, "unsupported")
        else:
            d = ModeDecision(mode, reason, mode, "ok", self._kwargs(mode))
        if d.planned is not None:
            if not first and d.planned != self.current:
                self.changes += 1
            self.current = d.planned
        d.change_index = self.changes
        self.log.append(d)
        return d

    def _kwargs(self, mode: ReasoningMode | None) -> dict:
        if mode is None or self._prof is None or not self._prof.template_flag or self._prof.fixed_mode:
            return {}
        return {self._prof.template_flag: mode == ReasoningMode.thinking}


class RuntimeAdapter:
    """Builds the LangChain client for a ModeDecision and records what was really applied."""

    def __init__(self, settings, capabilities: RuntimeCapabilities, client_cls=None):
        self.s, self.caps = settings, capabilities
        self._client_cls = client_cls

    def build_llm(self, decision: ModeDecision, *, max_tokens: int | None = None, streaming: bool = False,
                  json_mode: bool = False, cap: int | None = None):
        """`cap` raises the per-call output ceiling above INFERENCE_MAX_OUTPUT_TOKENS (the 1000-token limit for
        ordinary answers); only explicit task policies such as artifact drafting pass it."""
        s = self.s
        cls = self._client_cls
        if cls is None:
            from langchain_openai import ChatOpenAI as cls
        thinking = decision.planned == ReasoningMode.thinking
        extra = {"top_k": s.top_k}
        if decision.config:
            extra["chat_template_kwargs"] = dict(decision.config)
        kwargs = dict(model=s.model, base_url=s.base_url, api_key=s.api_key or "EMPTY", timeout=s.timeout_s,
                      max_retries=s.max_retries,
                      temperature=s.reasoning_temperature if thinking else s.grounded_temperature,
                      top_p=s.reasoning_top_p if thinking else s.grounded_top_p,
                      max_tokens=min(max_tokens or s.max_output_tokens, cap or s.max_output_tokens),
                      streaming=streaming, extra_body=extra)
        if json_mode:  # via extra_body: the SDK's parse() helper would raise on truncated output instead of returning it
            extra["response_format"] = {"type": "json_object"}
        llm = cls(**kwargs)
        # Only now is the configuration actually applied to a request-bound client.
        decision.applied = decision.planned
        decision.applied_config = dict(decision.config)
        logger.info("reasoning_mode_applied", extra={"requested": decision.requested.value,
                                                     "applied": decision.applied.value if decision.applied else None,
                                                     "status": decision.status, "config": decision.applied_config})
        return llm

    async def probe(self) -> dict:
        """Live check that the toggle really changes behaviour (costs two tiny requests)."""
        import openai
        c = openai.AsyncOpenAI(base_url=self.s.base_url, api_key=self.s.api_key or "EMPTY", timeout=60)
        out = {}
        for mode in (ReasoningMode.thinking, ReasoningMode.non_thinking):
            ctl = ReasoningController(self.caps)
            d = ctl.request(mode, "probe")
            r = await c.chat.completions.create(
                model=self.s.model, max_tokens=200, temperature=0,
                messages=[{"role": "user", "content": "What is 17*23? Answer with the number."}],
                extra_body={"chat_template_kwargs": d.config} if d.config else {})
            m = r.choices[0].message
            reasoning = getattr(m, "reasoning", None) or getattr(m, "reasoning_content", None)
            out[mode.value] = {"reasoning_returned": bool(reasoning), "completion_tokens": r.usage.completion_tokens}
        self.caps.probe_verified = (out["thinking"]["reasoning_returned"] and not out["non_thinking"]["reasoning_returned"])
        return out


# ---- mode-selection policy (deterministic, configurable) -----------------------------------
@dataclass(frozen=True)
class ModePolicy:
    """Non-thinking by default; thinking where multi-step synthesis pays for the extra tokens."""
    thinking_intents: frozenset = frozenset({"artifact", "research", "multi_stage"})
    contradiction_triggers_thinking: bool = True
    evidence_items_for_thinking: int = 8

    def initial(self, intent: str, n_evidence: int = 0) -> tuple[ReasoningMode, str]:
        if intent in self.thinking_intents:
            return ReasoningMode.thinking, f"intent '{intent}' needs multi-step synthesis"
        if n_evidence >= self.evidence_items_for_thinking:
            return ReasoningMode.thinking, f"{n_evidence} evidence items to reconcile"
        return ReasoningMode.non_thinking, "straightforward request"

    def escalate(self, *, contradictions: int) -> tuple[ReasoningMode, str] | None:
        if self.contradiction_triggers_thinking and contradictions > 0:
            return ReasoningMode.thinking, f"{contradictions} contradictory evidence groups"
        return None
