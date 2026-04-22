#!/usr/bin/env python3
"""Tests for tools/claude_subagent_tool.py"""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.claude_subagent_tool import (
    _build_claude_args,
    _check_claude_available,
    _resolve_claude_command,
    _run_claude_subprocess,
    claude_subagent,
    claude_subagent_batch,
)
from tools.registry import registry


# ---------------------------------------------------------------------------
# Helpers / mocks
# ---------------------------------------------------------------------------


class FakeParentAgent:
    def __init__(self):
        self.tool_progress_callback = None


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


def test_resolve_claude_command_env_override():
    with patch.dict(os.environ, {"CLAUDE_CLI_PATH": "/custom/claude"}):
        assert _resolve_claude_command() == "/custom/claude"


def test_resolve_claude_command_fallback():
    with patch.dict(os.environ, {}, clear=True):
        cmd = _resolve_claude_command()
        assert cmd in ("claude", "/usr/local/bin/claude", os.path.expanduser("~/.local/bin/claude"))


@patch("tools.claude_subagent_tool.subprocess.run")
def test_check_claude_available_true(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    assert _check_claude_available() is True


@patch("tools.claude_subagent_tool.subprocess.run")
def test_check_claude_available_false(mock_run):
    mock_run.side_effect = FileNotFoundError()
    assert _check_claude_available() is False


def test_build_claude_args_basic():
    cmd = _build_claude_args(goal="Say hello")
    assert cmd[0] == "claude"
    assert "--print" in cmd
    assert "--bare" in cmd
    assert cmd[-1] == "TASK:\nSay hello\n\nComplete this task thoroughly. Return a clear, concise summary of what you did, what you found, any files created or modified, and any issues encountered."


def test_build_claude_args_with_context():
    cmd = _build_claude_args(goal="Fix bug", context="File: main.py line 42")
    assert "Fix bug" in cmd[-1]
    assert "main.py line 42" in cmd[-1]


def test_build_claude_args_with_options():
    cmd = _build_claude_args(
        goal="Refactor",
        agent_name="reviewer",
        model="claude-sonnet-4-6",
        allowed_tools=["Bash", "Edit"],
        permission_mode="dontAsk",
        bare=False,
    )
    assert "--agent" in cmd
    assert "reviewer" in cmd
    assert "--model" in cmd
    assert "claude-sonnet-4-6" in cmd
    assert "--allowed-tools" in cmd
    assert "Bash, Edit" in cmd
    assert "--permission-mode" in cmd
    assert "dontAsk" in cmd
    assert "--bare" not in cmd


def test_build_claude_args_custom_agent():
    cmd = _build_claude_args(goal="Review code", custom_agent_prompt="You are a strict reviewer")
    assert "--agents" in cmd
    assert "custom" in cmd
    assert "--agent" in cmd
    idx = cmd.index("--agent")
    assert cmd[idx + 1] == "custom"


@patch("tools.claude_subagent_tool._resolve_claude_command", return_value="claude")
@patch("tools.claude_subagent_tool.subprocess.run")
def test_run_claude_subprocess_success(mock_run, _mock_cmd):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="Done! I created a file.",
        stderr="",
    )
    result = _run_claude_subprocess(["claude", "--print", "test"], timeout=10)
    assert result["status"] == "completed"
    assert result["output"] == "Done! I created a file."
    assert result["returncode"] == 0
    assert "duration_seconds" in result


@patch("tools.claude_subagent_tool._resolve_claude_command", return_value="claude")
@patch("tools.claude_subagent_tool.subprocess.run")
def test_run_claude_subprocess_error(mock_run, _mock_cmd):
    mock_run.return_value = MagicMock(
        returncode=1,
        stdout="",
        stderr="Something went wrong",
    )
    result = _run_claude_subprocess(["claude", "--print", "test"], timeout=10)
    assert result["status"] == "error"
    assert "Something went wrong" in result["error"]


@patch("tools.claude_subagent_tool._resolve_claude_command", return_value="claude")
@patch("tools.claude_subagent_tool.subprocess.run")
def test_run_claude_subprocess_timeout(mock_run, _mock_cmd):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=1)
    result = _run_claude_subprocess(["claude", "--print", "test"], timeout=1)
    assert result["status"] == "timeout"
    assert "timed out" in result["error"]


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------


@patch("tools.claude_subagent_tool._check_claude_available", return_value=False)
def test_claude_subagent_no_cli(mock_check):
    result = json.loads(claude_subagent(goal="test"))
    assert result["success"] is False
    assert "not found" in result["error"]


@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
@patch("tools.claude_subagent_tool._run_claude_subprocess")
def test_claude_subagent_success(mock_run, _mock_check):
    mock_run.return_value = {
        "status": "completed",
        "output": "All done",
        "duration_seconds": 1.5,
        "returncode": 0,
    }
    result = json.loads(claude_subagent(goal="Say hello"))
    assert result["success"] is True
    assert result["output"] == "All done"
    assert result["duration_seconds"] == 1.5


@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
@patch("tools.claude_subagent_tool._run_claude_subprocess")
def test_claude_subagent_failure(mock_run, _mock_check):
    mock_run.return_value = {
        "status": "error",
        "error": "CLI crashed",
        "duration_seconds": 0.5,
    }
    result = json.loads(claude_subagent(goal="test"))
    assert result["success"] is False
    assert "CLI crashed" in result["error"]


@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
@patch("tools.claude_subagent_tool._run_claude_subprocess")
def test_claude_subagent_batch(mock_run, _mock_check):
    mock_run.return_value = {
        "status": "completed",
        "output": "done",
        "duration_seconds": 1.0,
        "returncode": 0,
    }
    tasks = [
        {"goal": "Task 1"},
        {"goal": "Task 2"},
    ]
    result = json.loads(claude_subagent_batch(tasks=tasks, max_concurrent=2))
    assert result["success"] is True
    assert result["total_tasks"] == 2
    assert result["completed"] == 2
    assert result["failed"] == 0
    assert len(result["results"]) == 2


@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
def test_claude_subagent_batch_too_many_tasks(_mock_check):
    tasks = [{"goal": f"Task {i}"} for i in range(10)]
    result = json.loads(claude_subagent_batch(tasks=tasks))
    assert result["success"] is False
    assert "Too many tasks" in result["error"]


@patch("tools.claude_subagent_tool._check_claude_available", return_value=True)
def test_claude_subagent_batch_empty(_mock_check):
    result = json.loads(claude_subagent_batch(tasks=[]))
    assert result["success"] is False


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_claude_subagent_registered():
    entry = registry.get_entry("claude_subagent")
    assert entry is not None
    assert entry.toolset == "claude"
    assert entry.emoji == "🧠"


def test_claude_subagent_batch_registered():
    entry = registry.get_entry("claude_subagent_batch")
    assert entry is not None
    assert entry.toolset == "claude"


# ---------------------------------------------------------------------------
# Toolset membership tests
# ---------------------------------------------------------------------------


def test_claude_tools_in_toolsets():
    from toolsets import TOOLSETS
    assert "claude" in TOOLSETS
    tools = TOOLSETS["claude"]["tools"]
    assert "claude_subagent" in tools
    assert "claude_subagent_batch" in tools
    assert "claude_memory_list" in tools
    assert "claude_memory_sync_to_hermes" in tools


def test_claude_tools_in_role_toolsets():
    from toolsets import TOOLSETS
    for role in ("devops", "quant-trader", "propfirm-trader", "content-creator", "fullstack-dev", "system-engineer"):
        tools = TOOLSETS[role]["tools"]
        assert "claude_subagent" in tools, f"Missing claude_subagent in {role}"
        assert "claude_memory_list" in tools, f"Missing claude_memory_list in {role}"
