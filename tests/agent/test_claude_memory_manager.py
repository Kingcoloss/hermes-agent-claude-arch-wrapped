#!/usr/bin/env python3
"""Tests for agent/claude_memory_manager.py"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.claude_memory_manager import (
    ClaudeMemoryManager,
    _build_frontmatter,
    _build_memory_file,
    _parse_frontmatter,
    _resolve_claude_md_path,
    _resolve_memory_dir,
    _resolve_project_id,
)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

def test_resolve_project_id():
    assert _resolve_project_id("/home/user/project") == "home-user-project"


def test_resolve_project_id_default():
    with patch("agent.claude_memory_manager.os.getcwd", return_value="/workspace/app"):
        assert _resolve_project_id() == "workspace-app"


def test_resolve_memory_dir():
    with patch.dict(os.environ, {"HOME": "/home/test"}):
        d = _resolve_memory_dir("/home/user/project")
        assert str(d).endswith("home-user-project/memory")


def test_resolve_claude_md_path():
    p = _resolve_claude_md_path("/workspace/app")
    assert str(p) == "/workspace/app/.claude/CLAUDE.md"


def test_parse_frontmatter_simple():
    content = "---\nname: foo\ndescription: bar\ntype: project\n---\n\nBody here."
    fm, body = _parse_frontmatter(content)
    assert fm["name"] == "foo"
    assert fm["description"] == "bar"
    assert fm["type"] == "project"
    assert body == "Body here."


def test_parse_frontmatter_no_frontmatter():
    content = "Just body text."
    fm, body = _parse_frontmatter(content)
    assert fm == {}
    assert body == "Just body text."


def test_build_frontmatter():
    fm = {"name": "test", "description": "A test"}
    result = _build_frontmatter(fm)
    assert result == "---\nname: \"test\"\ndescription: \"A test\"\n---"


def test_build_memory_file():
    result = _build_memory_file("test", "A test", "project", "Body content.", "sess-123")
    assert "---" in result
    assert 'name: "test"' in result
    assert 'type: "project"' in result
    assert 'originSessionId: "sess-123"' in result
    assert "Body content." in result


# ---------------------------------------------------------------------------
# Manager tests (with temp dirs)
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_manager(tmp_path):
    project = tmp_path / "myproject"
    project.mkdir()
    return ClaudeMemoryManager(str(project))


def test_manager_ensure_dirs(temp_manager):
    assert temp_manager.memory_dir.exists()


def test_write_and_read_memory(temp_manager):
    result = temp_manager.write_memory(
        name="test-mem",
        body="This is a test.",
        mem_type="project",
        description="Test memory",
    )
    assert result["success"] is True
    assert (temp_manager.memory_dir / "test-mem.md").exists()

    mem = temp_manager.read_memory("test-mem")
    assert mem is not None
    assert mem["name"] == "test-mem"
    assert mem["body"] == "This is a test."
    assert mem["type"] == "project"


def test_write_memory_invalid_type(temp_manager):
    with pytest.raises(ValueError, match="Invalid memory type"):
        temp_manager.write_memory("bad", "body", mem_type="invalid")


def test_list_memories(temp_manager):
    temp_manager.write_memory("alpha", "Alpha body", "project", "First memory")
    temp_manager.write_memory("beta", "Beta body", "user", "Second memory")
    memories = temp_manager.list_memories()
    # MEMORY.md should be excluded from the list
    assert len(memories) == 2
    names = {m["name"] for m in memories}
    assert names == {"alpha", "beta"}


def test_delete_memory(temp_manager):
    temp_manager.write_memory("to-delete", "Body", "project")
    assert (temp_manager.memory_dir / "to-delete.md").exists()

    result = temp_manager.delete_memory("to-delete")
    assert result["success"] is True
    assert not (temp_manager.memory_dir / "to-delete.md").exists()


def test_delete_memory_not_found(temp_manager):
    result = temp_manager.delete_memory("missing")
    assert result["success"] is False


def test_memory_index_auto_updated(temp_manager):
    temp_manager.write_memory("indexed", "Body", "project", "An indexed memory")
    entries = temp_manager.read_memory_index()
    assert len(entries) == 1
    assert entries[0]["title"] == "indexed"
    assert "An indexed memory" in entries[0]["description"]


def test_memory_index_remove_on_delete(temp_manager):
    temp_manager.write_memory("to-remove", "Body", "project")
    temp_manager.delete_memory("to-remove")
    entries = temp_manager.read_memory_index()
    assert len(entries) == 0


def test_read_claude_md_missing(temp_manager):
    assert temp_manager.read_claude_md() is None


def test_write_and_read_claude_md(temp_manager):
    temp_manager.write_claude_md("# Project Instructions\n\nBe helpful.")
    content = temp_manager.read_claude_md()
    assert content == "# Project Instructions\n\nBe helpful."


def test_update_claude_md(temp_manager):
    temp_manager.write_claude_md("# Base")
    temp_manager.update_claude_md("## Section 2\nMore info.")
    content = temp_manager.read_claude_md()
    assert "# Base" in content
    assert "## Section 2" in content


def test_sync_to_hermes(temp_manager, tmp_path):
    temp_manager.write_memory("sync-test", "Sync me", "project")
    with patch("agent.claude_memory_manager.get_hermes_home", return_value=tmp_path):
        result = temp_manager.sync_to_hermes()
    assert result["success"] is True
    assert result["synced"] == 1
    assert result["skipped"] == 0


def test_sync_from_hermes(temp_manager, tmp_path):
    hermes_dir = tmp_path / "memories"
    hermes_dir.mkdir(parents=True)
    entry = {
        "session_id": "sess-abc",
        "timestamp": 1234567890,
        "fact": {"key": "value"},
        "fact_hash": "deadbeef1234",
    }
    (hermes_dir / "sess-abc.jsonl").write_text(json.dumps(entry) + "\n")
    with patch("agent.claude_memory_manager.get_hermes_home", return_value=tmp_path):
        result = temp_manager.sync_from_hermes()
    assert result["success"] is True
    assert result["synced"] == 1


def test_sync_from_hermes_no_dir(temp_manager, tmp_path):
    with patch("agent.claude_memory_manager.get_hermes_home", return_value=tmp_path):
        result = temp_manager.sync_from_hermes()
    assert result["success"] is False
    assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# Tool wrapper tests
# ---------------------------------------------------------------------------


def test_claude_memory_list_tool(temp_manager):
    temp_manager.write_memory("listable", "Body", "project")
    from agent.claude_memory_manager import claude_memory_list
    with patch("agent.claude_memory_manager.ClaudeMemoryManager") as MockMgr:
        MockMgr.return_value = temp_manager
        result = json.loads(claude_memory_list())
    assert result["success"] is True
    # MEMORY.md excluded, only listable.md counts
    assert result["count"] == 1


def test_claude_memory_read_tool(temp_manager):
    temp_manager.write_memory("readable", "Readable body", "project")
    from agent.claude_memory_manager import claude_memory_read
    with patch("agent.claude_memory_manager.ClaudeMemoryManager") as MockMgr:
        MockMgr.return_value = temp_manager
        result = json.loads(claude_memory_read("readable"))
    assert result["success"] is True
    assert result["memory"]["body"] == "Readable body"


def test_claude_memory_write_tool(temp_manager):
    from agent.claude_memory_manager import claude_memory_write
    with patch("agent.claude_memory_manager.ClaudeMemoryManager") as MockMgr:
        MockMgr.return_value = temp_manager
        result = json.loads(claude_memory_write("written", "Written body", mem_type="project"))
    assert result["success"] is True
    assert (temp_manager.memory_dir / "written.md").exists()


def test_claude_memory_delete_tool(temp_manager):
    temp_manager.write_memory("deletable", "Body", "project")
    from agent.claude_memory_manager import claude_memory_delete
    with patch("agent.claude_memory_manager.ClaudeMemoryManager") as MockMgr:
        MockMgr.return_value = temp_manager
        result = json.loads(claude_memory_delete("deletable"))
    assert result["success"] is True
    assert not (temp_manager.memory_dir / "deletable.md").exists()
