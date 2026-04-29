"""Integration tests for Claude Code sub-agent spawning and memory management.

These tests exercise the full tool handlers and memory manager with mocked
subprocess / filesystem boundaries so they run without the real ``claude`` CLI
or Anthropic API keys.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration

# Ensure imports trigger self-registration
import tools.claude_memory_tool  # noqa: F401
import tools.claude_subagent_tool  # noqa: F401
from agent.claude_memory_manager import (
    ClaudeMemoryManager,
    claude_memory_delete,
    claude_memory_list,
    claude_memory_read,
    claude_memory_read_claude_md,
    claude_memory_read_index,
    claude_memory_sync_from_hermes,
    claude_memory_sync_to_hermes,
    claude_memory_write,
    claude_memory_write_claude_md,
)
from agent.memory_extractor import MemoryExtractor
from tools.claude_subagent_tool import (
    claude_subagent,
    claude_subagent_batch,
)
from tools.registry import registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_manager(tmp_path):
    """Return a ClaudeMemoryManager rooted in a temp directory."""
    project = tmp_path / "claude_project"
    project.mkdir()
    return ClaudeMemoryManager(str(project))


class FakeParentAgent:
    def __init__(self):
        self.callbacks = []
        self.tool_progress_callback = self._callback

    def _callback(self, event, **kwargs):
        self.callbacks.append({"event": event, **kwargs})


# ---------------------------------------------------------------------------
# 1-5: claude_subagent() single task
# ---------------------------------------------------------------------------


@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
@patch("tools.claude_subagent_tool.subprocess.run")
def test_claude_subagent_single_task(mock_run, _mock_check):
    """1. claude_subagent() single task — mock subprocess to return JSON, verify args contain --print."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="Done! I created a file.",
        stderr="",
    )
    result = json.loads(claude_subagent(goal="Say hello"))
    assert result["success"] is True
    assert result["output"] == "Done! I created a file."

    # Verify subprocess was called with --print
    args = mock_run.call_args[0][0]
    assert "--print" in args


@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
@patch("tools.claude_subagent_tool.subprocess.run")
def test_claude_subagent_custom_agent(mock_run, _mock_check):
    """2. claude_subagent() with custom agent — verify --agent flag in args."""
    mock_run.return_value = MagicMock(returncode=0, stdout="reviewed", stderr="")
    result = json.loads(
        claude_subagent(
            goal="Review code",
            agent_name="reviewer",
        )
    )
    assert result["success"] is True
    args = mock_run.call_args[0][0]
    assert "--agent" in args
    assert "reviewer" in args


@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
@patch("tools.claude_subagent_tool.subprocess.run")
def test_claude_subagent_tool_restriction(mock_run, _mock_check):
    """3. claude_subagent() with tool restriction — verify --allowed-tools in args."""
    mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
    result = json.loads(
        claude_subagent(
            goal="Fix bug",
            allowed_tools=["Bash", "Edit"],
        )
    )
    assert result["success"] is True
    args = mock_run.call_args[0][0]
    assert "--allowed-tools" in args
    assert "Bash, Edit" in args


@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
@patch("tools.claude_subagent_tool.subprocess.run")
def test_claude_subagent_permission_mode(mock_run, _mock_check):
    """4. claude_subagent() with permission mode — verify --permission-mode in args."""
    mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
    result = json.loads(
        claude_subagent(
            goal="Deploy app",
            permission_mode="acceptEdits",
        )
    )
    assert result["success"] is True
    args = mock_run.call_args[0][0]
    assert "--permission-mode" in args
    assert "acceptEdits" in args


@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
@patch("tools.claude_subagent_tool.subprocess.run")
def test_claude_subagent_error_handling(mock_run, _mock_check):
    """5. claude_subagent() error handling — subprocess failure returns JSON error."""
    mock_run.return_value = MagicMock(
        returncode=1,
        stdout="",
        stderr="Something went wrong",
    )
    result = json.loads(claude_subagent(goal="Fail task"))
    assert result["success"] is False
    assert "Something went wrong" in result["error"]
    assert result.get("status") == "error" or "error" in result


# ---------------------------------------------------------------------------
# 6-8: claude_subagent_batch() parallel execution
# ---------------------------------------------------------------------------


@patch("tools.claude_subagent_tool._resolve_claude_command", return_value="claude")
@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
@patch("tools.claude_subagent_tool.subprocess.run")
def test_claude_subagent_batch_parallel_execution(mock_run, _mock_check, _mock_resolve):
    """6. claude_subagent_batch() parallel execution — verify ThreadPoolExecutor usage, max 5 tasks."""
    mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
    tasks = [{"goal": f"Task {i}"} for i in range(5)]
    result = json.loads(claude_subagent_batch(tasks=tasks, max_concurrent=3))
    assert result["success"] is True
    assert result["total_tasks"] == 5
    assert result["completed"] == 5
    assert result["failed"] == 0
    assert len(result["results"]) == 5
    # subprocess.run should have been called 5 times (one per task)
    assert mock_run.call_count == 5


@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
def test_claude_subagent_batch_limit_enforcement(_mock_check):
    """7. claude_subagent_batch() batch limit enforcement — more than 5 tasks raises error."""
    tasks = [{"goal": f"Task {i}"} for i in range(6)]
    result = json.loads(claude_subagent_batch(tasks=tasks))
    assert result["success"] is False
    assert "Too many tasks" in result["error"]


@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
@patch("tools.claude_subagent_tool.subprocess.run")
def test_claude_subagent_batch_progress_callback(mock_run, _mock_check):
    """8. claude_subagent_batch() progress callback — verify callback called for each task."""
    mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
    fake_parent = FakeParentAgent()
    tasks = [{"goal": "Task A"}, {"goal": "Task B"}]
    result = json.loads(
        claude_subagent_batch(
            tasks=tasks,
            max_concurrent=2,
            parent_agent=fake_parent,
        )
    )
    assert result["success"] is True
    # There should be start + complete callbacks for each task
    start_events = [c for c in fake_parent.callbacks if c["event"] == "subagent.start"]
    complete_events = [c for c in fake_parent.callbacks if c["event"] == "subagent.complete"]
    assert len(start_events) == 2
    assert len(complete_events) == 2
    for c in complete_events:
        assert c["status"] == "completed"


# ---------------------------------------------------------------------------
# 9-16: ClaudeMemoryManager
# ---------------------------------------------------------------------------


def test_memory_manager_write_memory_with_frontmatter(isolated_manager):
    """9. ClaudeMemoryManager.write_memory() — creates file with YAML frontmatter."""
    result = isolated_manager.write_memory(
        name="test-mem",
        body="This is a test memory.",
        mem_type="project",
        description="Test description",
        origin_session_id="sess-123",
    )
    assert result["success"] is True
    filepath = isolated_manager.memory_dir / "test-mem.md"
    assert filepath.exists()
    content = filepath.read_text(encoding="utf-8")
    assert content.startswith("---")
    assert 'name: "test-mem"' in content
    assert 'type: "project"' in content
    assert 'originSessionId: "sess-123"' in content
    assert "This is a test memory." in content


def test_memory_manager_read_memory_parses_frontmatter(isolated_manager):
    """10. ClaudeMemoryManager.read_memory() — reads file and parses frontmatter."""
    isolated_manager.write_memory(
        name="readable",
        body="Readable body content.",
        mem_type="user",
        description="A user memory",
    )
    mem = isolated_manager.read_memory("readable")
    assert mem is not None
    assert mem["name"] == "readable"
    assert mem["body"] == "Readable body content."
    assert mem["type"] == "user"
    assert mem["description"] == "A user memory"
    assert "frontmatter" in mem


def test_memory_manager_list_memories_excludes_index(isolated_manager):
    """11. ClaudeMemoryManager.list_memories() — lists all memory files (excluding MEMORY.md)."""
    isolated_manager.write_memory("alpha", "Alpha body", "project", "First")
    isolated_manager.write_memory("beta", "Beta body", "reference", "Second")
    # MEMORY.md is auto-created by write_memory; it must NOT appear in the list
    memories = isolated_manager.list_memories()
    filenames = {m["filename"] for m in memories}
    assert "alpha.md" in filenames
    assert "beta.md" in filenames
    assert "MEMORY.md" not in filenames
    assert len(memories) == 2


def test_memory_manager_delete_memory_updates_index(isolated_manager):
    """12. ClaudeMemoryManager.delete_memory() — removes file and updates index."""
    isolated_manager.write_memory("to-delete", "Body", "project", "To be deleted")
    assert (isolated_manager.memory_dir / "to-delete.md").exists()

    result = isolated_manager.delete_memory("to-delete")
    assert result["success"] is True
    assert not (isolated_manager.memory_dir / "to-delete.md").exists()

    # MEMORY.md index should no longer reference it
    entries = isolated_manager.read_memory_index()
    for entry in entries:
        assert entry["file"] != "to-delete.md"


def test_memory_manager_read_memory_index(isolated_manager):
    """13. ClaudeMemoryManager.read_memory_index() — parses MEMORY.md index."""
    isolated_manager.write_memory("indexed", "Body", "project", "An indexed memory")
    entries = isolated_manager.read_memory_index()
    assert len(entries) == 1
    assert entries[0]["title"] == "indexed"
    assert entries[0]["file"] == "indexed.md"
    assert "An indexed memory" in entries[0]["description"]


def test_memory_manager_claude_md_crud(isolated_manager):
    """14. ClaudeMemoryManager.write_claude_md() / read_claude_md() — CLAUDE.md CRUD."""
    assert isolated_manager.read_claude_md() is None

    isolated_manager.write_claude_md("# Project Instructions\n\nBe helpful.")
    content = isolated_manager.read_claude_md()
    assert content == "# Project Instructions\n\nBe helpful."

    isolated_manager.update_claude_md("## Section 2\nMore info.")
    updated = isolated_manager.read_claude_md()
    assert "# Project Instructions" in updated
    assert "## Section 2" in updated


@patch("agent.claude_memory_manager.get_hermes_home")
def test_memory_manager_sync_to_hermes(mock_get_home, isolated_manager, tmp_path):
    """15. ClaudeMemoryManager.sync_to_hermes() — syncs to JSONL in HERMES_HOME/memories/."""
    hermes_dir = tmp_path / "hermes_test"
    hermes_dir.mkdir(parents=True, exist_ok=True)
    (hermes_dir / "memories").mkdir(parents=True, exist_ok=True)
    mock_get_home.return_value = hermes_dir

    isolated_manager.write_memory("sync-test", "Sync me to Hermes", "project")

    # Mock _extract_facts to avoid LLM calls while keeping real JSONL I/O
    with patch.object(MemoryExtractor, "_extract_facts", return_value=["Sync me to Hermes"]):
        result = isolated_manager.sync_to_hermes()

    assert result["success"] is True
    assert result["synced"] == 1
    assert result["skipped"] == 0
    # Verify JSONL was written
    jsonl_files = list((hermes_dir / "memories").glob("*.jsonl"))
    assert len(jsonl_files) == 1


@patch("agent.claude_memory_manager.get_hermes_home")
def test_memory_manager_sync_from_hermes(mock_get_home, isolated_manager, tmp_path):
    """16. ClaudeMemoryManager.sync_from_hermes() — syncs from JSONL to markdown files."""
    hermes_dir = tmp_path / "hermes_test"
    hermes_dir.mkdir(parents=True, exist_ok=True)
    memories_dir = hermes_dir / "memories"
    memories_dir.mkdir(parents=True, exist_ok=True)
    mock_get_home.return_value = hermes_dir

    entry = {
        "session_id": "sess-abc",
        "timestamp": 1234567890,
        "fact": {"key": "value"},
        "fact_hash": "deadbeef1234",
    }
    (memories_dir / "sess-abc.jsonl").write_text(json.dumps(entry) + "\n")

    result = isolated_manager.sync_from_hermes()
    assert result["success"] is True
    assert result["synced"] == 1
    # Verify a markdown file was created from the JSONL entry
    mem_files = [f for f in isolated_manager.memory_dir.iterdir() if f.suffix == ".md" and f.name != "MEMORY.md"]
    assert len(mem_files) == 1


# ---------------------------------------------------------------------------
# 17: Tool wrappers return JSON strings
# ---------------------------------------------------------------------------


def test_tool_wrapper_claude_memory_list(isolated_manager):
    """17a. claude_memory_list returns JSON string."""
    isolated_manager.write_memory("listable", "Body", "project")
    with patch("agent.claude_memory_manager.ClaudeMemoryManager") as MockMgr:
        MockMgr.return_value = isolated_manager
        result = claude_memory_list()
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["count"] == 1


def test_tool_wrapper_claude_memory_read(isolated_manager):
    """17b. claude_memory_read returns JSON string."""
    isolated_manager.write_memory("readable", "Readable body", "project")
    with patch("agent.claude_memory_manager.ClaudeMemoryManager") as MockMgr:
        MockMgr.return_value = isolated_manager
        result = claude_memory_read("readable")
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["memory"]["body"] == "Readable body"


def test_tool_wrapper_claude_memory_write(isolated_manager):
    """17c. claude_memory_write returns JSON string."""
    with patch("agent.claude_memory_manager.ClaudeMemoryManager") as MockMgr:
        MockMgr.return_value = isolated_manager
        result = claude_memory_write("written", "Written body", mem_type="project")
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["success"] is True
    assert (isolated_manager.memory_dir / "written.md").exists()


def test_tool_wrapper_claude_memory_delete(isolated_manager):
    """17d. claude_memory_delete returns JSON string."""
    isolated_manager.write_memory("deletable", "Body", "project")
    with patch("agent.claude_memory_manager.ClaudeMemoryManager") as MockMgr:
        MockMgr.return_value = isolated_manager
        result = claude_memory_delete("deletable")
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["success"] is True
    assert not (isolated_manager.memory_dir / "deletable.md").exists()


# ---------------------------------------------------------------------------
# 18: Tool registry — all claude tools registered
# ---------------------------------------------------------------------------


def test_all_claude_tools_registered():
    """18. Tool registry: all 11 claude tools are registered in toolsets.py and the registry."""
    expected_tools = {
        "claude_subagent",
        "claude_subagent_batch",
        "claude_memory_list",
        "claude_memory_read",
        "claude_memory_write",
        "claude_memory_delete",
        "claude_memory_read_index",
        "claude_memory_read_claude_md",
        "claude_memory_write_claude_md",
        "claude_memory_sync_to_hermes",
        "claude_memory_sync_from_hermes",
    }

    # Registry checks
    registered = set(registry.get_tool_names_for_toolset("claude"))
    for tool in expected_tools:
        assert registry.get_entry(tool) is not None, f"{tool} not registered"
    assert registered == expected_tools, f"Registry mismatch: {registered ^ expected_tools}"

    # Toolsets.py checks
    from toolsets import TOOLSETS
    assert "claude" in TOOLSETS
    toolset_tools = set(TOOLSETS["claude"]["tools"])
    assert toolset_tools == expected_tools, f"Toolsets mismatch: {toolset_tools ^ expected_tools}"

    # Role toolsets checks (spot-check a few roles)
    for role in ("devops", "quant-trader", "fullstack-dev"):
        role_tools = set(TOOLSETS[role]["tools"])
        assert "claude_subagent" in role_tools, f"Missing claude_subagent in {role}"
        assert "claude_memory_list" in role_tools, f"Missing claude_memory_list in {role}"
