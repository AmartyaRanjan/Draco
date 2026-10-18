"""Layering rule: only draco.adapters and draco.tools may import the existing `app` package, and
the framework layers must be importable without the ML stack or LangGraph."""
import ast
import pathlib
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "draco"
FRAMEWORK = ["core", "plugins", "policies", "harness", "observability", "state", "agents"]
# draco.tools wraps the light app.agent contract/registry modules only
TOOLS_ALLOWED = {"app.agent.contracts", "app.agent.registry", "app.agent.web"}
ORDER = {"core": 0, "plugins": 1, "policies": 1, "observability": 1, "state": 1, "harness": 2, "agents": 3,
         "tools": 3, "adapters": 4}


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_only_adapters_and_tools_import_app():
    for pkg in FRAMEWORK:
        for f in (SRC / pkg).rglob("*.py"):
            bad = [m for m in imports(f) if m == "app" or m.startswith("app.")]
            assert not bad, f"{f.relative_to(SRC)} imports {bad}"
    for f in (SRC / "tools").rglob("*.py"):
        bad = [m for m in imports(f) if (m == "app" or m.startswith("app.")) and m not in TOOLS_ALLOWED]
        assert not bad, f"{f.relative_to(SRC)} imports {bad}"


def test_no_upward_dependencies():
    for pkg, level in ORDER.items():
        for f in (SRC / pkg).rglob("*.py"):
            for m in imports(f):
                if m.startswith("draco."):
                    target = m.split(".")[1]
                    if target in ORDER and target != pkg:
                        assert ORDER[target] <= level, f"{f.relative_to(SRC)} (layer {pkg}) imports upward: {m}"


def test_framework_imports_without_app_or_heavy_deps():
    root = SRC.parents[1]
    paths = [str(SRC.parent)] + [p for p in sys.path if p and pathlib.Path(p).resolve() != root]  # repo root (with `app`) excluded
    code = (f"import sys; sys.path[:0] = {paths!r}; import draco.core, draco.plugins, draco.policies, draco.harness, "
            "draco.observability, draco.state, draco.agents\n"
            "bad = [m for m in ('app', 'torch', 'langgraph', 'sentence_transformers', 'qdrant_client') if m in sys.modules]\n"
            "assert not bad, bad")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
