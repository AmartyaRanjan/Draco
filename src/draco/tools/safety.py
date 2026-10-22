"""Heuristic detection of instruction-like text inside untrusted tool output."""
import re

_INJECTION = re.compile(
    r"(ignore (all |any |the )?(previous|prior|above) (instructions|prompts)|disregard (the )?(system|previous)|"
    r"you are now|new instructions:|system prompt|<\|im_start\|>|</?(system|assistant)>|"
    r"(reveal|print|send|disclose) .{0,30}(api[ _-]?key|password|secret)|"
    r"disregard (what|the user|your|all|any)|(reply|respond|answer|output|write) (only |exactly )?with (exactly |only )?(?-i:[\"'`A-Z])|"
    r"do not (answer|follow|obey) (the )?(user|question|above)|call (the )?(tool|function|capability) )", re.I)


def looks_injected(text: str) -> bool:
    """Heuristic only; one layer among several (untrusted labelling, approval for side effects). Paraphrases get past it."""
    return bool(_INJECTION.search(text))
