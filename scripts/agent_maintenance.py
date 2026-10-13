"""Periodic cleanup (run from cron / a scheduler): expired artifacts and feedback past retention.
  python scripts/agent_maintenance.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from app.agent.artifacts import ArtifactStore  # noqa: E402
from app.agent.config import get_agent_settings  # noqa: E402
from app.agent.store import AgentStore  # noqa: E402

s = get_agent_settings()
store = AgentStore(s.store_path)
arts = ArtifactStore(s.artifact_dir, store, max_bytes=s.artifact_max_bytes, ttl_days=s.artifact_ttl_days)
print({"artifacts_deleted": arts.cleanup_expired(), "feedback_purged": store.purge_feedback(s.feedback_retention_days)})
