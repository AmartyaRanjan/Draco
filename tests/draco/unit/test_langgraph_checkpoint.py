import subprocess
import sys

import pytest

from draco.adapters.langgraph_checkpoint import (
    CheckpointConfig,
    build_sqlite_checkpointer,
    checkpoint_config,
    extract_interrupts,
    is_interrupted,
    resume_command,
)


def test_checkpoint_configuration_preserves_execution_identity():
    config = checkpoint_config(
        "execution-1",
        tenant_id="tenant-1",
        user_id="user-1",
        session_id="session-1",
        correlation_id="correlation-1",
        recursion_limit=9,
    )
    assert config["configurable"] == {"thread_id": "execution-1", "checkpoint_ns": ""}
    assert config["metadata"]["draco_tenant_id"] == "tenant-1"
    assert config["metadata"]["draco_user_id"] == "user-1"
    assert config["recursion_limit"] == 9


def test_checkpoint_configuration_rejects_invalid_values():
    with pytest.raises(ValueError):
        CheckpointConfig("")
    with pytest.raises(ValueError):
        CheckpointConfig("execution-1", recursion_limit=0)


def test_resume_command_validates_before_langgraph_invocation():
    with pytest.raises(ValueError):
        resume_command("invalid")
    pytest.importorskip("langgraph.types")
    command = resume_command("approve")
    assert command.resume == {"decision": "approve"}


def test_interrupt_helpers_extract_values():
    class Interrupt:
        value = {"type": "approval_required"}

    result = {"__interrupt__": [Interrupt()]}
    assert is_interrupted(result)
    assert extract_interrupts(result) == [{"type": "approval_required"}]
    assert not is_interrupted({"__interrupt__": []})
    assert extract_interrupts({"answer": "done"}) == []


def test_sqlite_checkpointer_factory_and_import_are_optional(tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    saver = build_sqlite_checkpointer(tmp_path / "langgraph.db")
    try:
        assert saver.is_setup
    finally:
        saver.conn.close()

    code = (
        "import sys; import draco.adapters.langgraph_checkpoint; "
        "assert 'langgraph' not in sys.modules and 'app' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
