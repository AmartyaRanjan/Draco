"""The injection heuristic is deliberately duplicated (the app image ships without draco); this guards drift."""
import pytest

from draco.tools.safety import _INJECTION as DRACO

APP = pytest.importorskip("app.agent.web")._INJECTION  # skipped where the starter app is not installed


def test_app_and_draco_injection_patterns_are_identical():
    assert (APP.pattern, APP.flags) == (DRACO.pattern, DRACO.flags)
