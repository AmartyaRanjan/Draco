import sys
from pathlib import Path

try:  # CI installs the framework (`pip install -e .`); a bare checkout uses src/
    import draco  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
