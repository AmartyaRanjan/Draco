from __future__ import annotations

from draco.core.errors import DuplicateRegistration
from draco.tools.contracts import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise DuplicateRegistration(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for t in self._tools.values():
            out.setdefault(t.group, []).append(t.name)
        return out
