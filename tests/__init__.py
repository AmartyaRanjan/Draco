import os
import sys
from pathlib import Path

os.environ.setdefault("RATE_LIMIT_ENABLED", "false")  # unit tests must not depend on Redis or share buckets

try:  # the app imports the framework; CI installs it (`pip install -e .`), a bare checkout uses src/
    import draco  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
