"""Integration tests for TUI Gateway RPC methods: role / KPI management.

These tests exercise the JSON-RPC handler layer in ``tui_gateway/server.py``
for role and gamification endpoints.  The underlying ``RoleManager`` and
``KPITracker`` are mocked so the tests isolate RPC envelope behaviour
(response shape, error codes, session side-effects) from business-logic
state.

Run with::

    python -m pytest tests/integration/test_tui_gateway_rpc.py -v
"""

import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable (mirrors conftest.py behaviour)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tui_gateway import server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_server_sessions():
    """Prevent cross-test pollution of the global session registry."""
    server._sessions.clear()
    yield
    server._sessions.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session(agent=None, **extra):
    """Build a minimal session dict matching the shape expected by server.py."""
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }


def _fake_role(name: str):
    """Return a lightweight fake RoleProfile-like object."""
    return types.SimpleNamespace(
        name=name,
        description=f"{name} description",
        toolsets=[name],
        default_model=None,
        skin=None,
        kpi_weights={},
        resolved_tools=["tool_a", "tool_b"],
    )


def _mock_role_manager(role_names=None):
    """Return a MagicMock configured like RoleManager."""
    role_names = role_names or ["devops", "quant-trader"]
    rm = MagicMock()
    rm.list_roles.return_value = role_names
    rm.get_role.side_effect = lambda n: _fake_role(n) if n in role_names else None
    return rm


def _mock_kpi_tracker():
    """Return a MagicMock configured like a KPITracker instance."""
    instance = MagicMock()
    instance.get_kpi_summary.return_value = {
        "record_count": 3,
        "task_success_rate": 0.95,
        "avg_tokens_per_task": 120.0,
        "tool_diversity_score": 0.8,
        "error_recovery_rate": 0.9,
        "role_proficiency_score": 0.85,
    }
    instance.get_level.return_value = {
        "skill_name": "devops",
        "level": 2,
        "xp": 150.0,
        "xp_to_next": 50.0,
    }
    instance.get_achievements.return_value = [
        {
            "id": 1,
            "achievement_id": "first_win",
            "name": "First Win",
            "description": "Win your first session",
            "role": "devops",
            "unlocked_at": 1_700_000_000.0,
        }
    ]
    instance.get_leaderboard.return_value = [
        {"rank": 1, "skill_name": "devops", "level": 5, "xp": 500.0},
        {"rank": 2, "skill_name": "quant-trader", "level": 3, "xp": 300.0},
    ]
    return instance


# ---------------------------------------------------------------------------
# role.list
# ---------------------------------------------------------------------------

def test_role_list_returns_jsonrpc_envelope_with_roles():
    """role.list must return a JSON-RPC result containing the mocked role list."""
    with patch("agent.role_manager.get_role_manager") as mock_get_rm:
        mock_get_rm.return_value = _mock_role_manager()
        resp = server.handle_request({"id": "r1", "method": "role.list", "params": {}})

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "r1"
    assert "result" in resp
    roles = resp["result"]["roles"]
    assert len(roles) == 2
    names = {r["name"] for r in roles}
    assert names == {"devops", "quant-trader"}
    for r in roles:
        assert "description" in r
        assert "toolsets" in r
        assert "resolved_tool_count" in r
        assert r["resolved_tool_count"] == 2


def test_role_list_returns_error_on_exception():
    """role.list must surface unexpected RoleManager errors as RPC error 5025."""
    with patch("agent.role_manager.get_role_manager", side_effect=RuntimeError("db down")):
        resp = server.handle_request({"id": "r2", "method": "role.list", "params": {}})

    assert "error" in resp
    assert resp["error"]["code"] == 5025
    assert "db down" in resp["error"]["message"]
    assert resp["id"] == "r2"


# ---------------------------------------------------------------------------
# role.get
# ---------------------------------------------------------------------------

def test_role_get_returns_single_role():
    """role.get must return the requested role profile."""
    with patch("agent.role_manager.get_role_manager") as mock_get_rm:
        mock_get_rm.return_value = _mock_role_manager()
        resp = server.handle_request(
            {"id": "r3", "method": "role.get", "params": {"name": "devops"}}
        )

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "r3"
    assert resp["result"]["name"] == "devops"
    assert resp["result"]["description"] == "devops description"
    assert resp["result"]["resolved_tool_count"] == 2


def test_role_get_returns_404_for_unknown_role():
    """role.get must return error code 4041 when the role does not exist."""
    with patch("agent.role_manager.get_role_manager") as mock_get_rm:
        mock_get_rm.return_value = _mock_role_manager()
        resp = server.handle_request(
            {"id": "r4", "method": "role.get", "params": {"name": "ghost"}}
        )

    assert "error" in resp
    assert resp["error"]["code"] == 4041
    assert "ghost" in resp["error"]["message"]


def test_role_get_returns_error_on_exception():
    """role.get must surface unexpected errors as RPC error 5026."""
    with patch("agent.role_manager.get_role_manager", side_effect=RuntimeError("boom")):
        resp = server.handle_request(
            {"id": "r5", "method": "role.get", "params": {"name": "devops"}}
        )

    assert resp["error"]["code"] == 5026


# ---------------------------------------------------------------------------
# role.switch
# ---------------------------------------------------------------------------

def test_role_switch_updates_session_and_agent():
    """role.switch must update both the session dict and the agent object."""
    agent = types.SimpleNamespace(role=None)
    server._sessions["sid"] = _session(agent=agent)

    with patch("agent.role_manager.get_role_manager") as mock_get_rm:
        mock_get_rm.return_value = _mock_role_manager()
        resp = server.handle_request(
            {
                "id": "r6",
                "method": "role.switch",
                "params": {"name": "devops", "session_id": "sid"},
            }
        )

    assert resp["jsonrpc"] == "2.0"
    assert resp["result"]["success"] is True
    assert resp["result"]["role"] == "devops"
    assert server._sessions["sid"]["role"] == "devops"
    assert agent.role == "devops"


def test_role_switch_succeeds_without_session():
    """role.switch must succeed even when no session is active."""
    with patch("agent.role_manager.get_role_manager") as mock_get_rm:
        mock_get_rm.return_value = _mock_role_manager()
        resp = server.handle_request(
            {"id": "r7", "method": "role.switch", "params": {"name": "quant-trader"}}
        )

    assert resp["result"]["success"] is True
    assert resp["result"]["role"] == "quant-trader"


def test_role_switch_rejects_unknown_role():
    """role.switch must return error 4041 for an unknown role."""
    with patch("agent.role_manager.get_role_manager") as mock_get_rm:
        mock_get_rm.return_value = _mock_role_manager()
        resp = server.handle_request(
            {"id": "r8", "method": "role.switch", "params": {"name": "ghost"}}
        )

    assert resp["error"]["code"] == 4041
    assert "ghost" in resp["error"]["message"]


def test_role_switch_returns_error_on_exception():
    """role.switch must surface unexpected errors as RPC error 5027."""
    with patch("agent.role_manager.get_role_manager", side_effect=RuntimeError("boom")):
        resp = server.handle_request(
            {"id": "r9", "method": "role.switch", "params": {"name": "devops"}}
        )

    assert resp["error"]["code"] == 5027


# ---------------------------------------------------------------------------
# kpi.summary
# ---------------------------------------------------------------------------

def test_kpi_summary_returns_summary():
    """kpi.summary must return the mocked KPI aggregate."""
    with patch("agent.gamification.KPITracker") as MockKPI:
        MockKPI.return_value = _mock_kpi_tracker()
        resp = server.handle_request(
            {"id": "r10", "method": "kpi.summary", "params": {"role": "devops", "days": 7}}
        )

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "r10"
    result = resp["result"]
    assert result["record_count"] == 3
    assert result["task_success_rate"] == 0.95


def test_kpi_summary_returns_error_on_exception():
    """kpi.summary must surface unexpected errors as RPC error 5028."""
    with patch("agent.gamification.KPITracker", side_effect=RuntimeError("boom")):
        resp = server.handle_request(
            {"id": "r11", "method": "kpi.summary", "params": {}}
        )

    assert resp["error"]["code"] == 5028
    assert "boom" in resp["error"]["message"]


# ---------------------------------------------------------------------------
# xp.status
# ---------------------------------------------------------------------------

def test_xp_status_with_skill_name():
    """xp.status must return level info when skill_name is provided."""
    with patch("agent.gamification.KPITracker") as MockKPI:
        MockKPI.return_value = _mock_kpi_tracker()
        resp = server.handle_request(
            {"id": "r12", "method": "xp.status", "params": {"skill_name": "devops"}}
        )

    assert resp["jsonrpc"] == "2.0"
    assert resp["result"]["skill_name"] == "devops"
    assert resp["result"]["level"] == 2
    assert resp["result"]["xp"] == 150.0
    assert resp["result"]["xp_to_next"] == 50.0


def test_xp_status_falls_back_to_session_role():
    """xp.status must fall back to the session's role when skill_name is omitted."""
    server._sessions["sid"] = _session()
    server._sessions["sid"]["role"] = "quant-trader"

    with patch("agent.gamification.KPITracker") as MockKPI:
        tracker = _mock_kpi_tracker()
        tracker.get_level.return_value = {
            "skill_name": "quant-trader",
            "level": 4,
            "xp": 350.0,
            "xp_to_next": 50.0,
        }
        MockKPI.return_value = tracker
        resp = server.handle_request(
            {"id": "r13", "method": "xp.status", "params": {"session_id": "sid"}}
        )

    assert resp["result"]["skill_name"] == "quant-trader"
    assert resp["result"]["level"] == 4


def test_xp_status_errors_without_skill_or_session():
    """xp.status must return error 4002 when neither skill_name nor a session role is available."""
    with patch("agent.gamification.KPITracker") as MockKPI:
        MockKPI.return_value = _mock_kpi_tracker()
        resp = server.handle_request(
            {"id": "r14", "method": "xp.status", "params": {}}
        )

    assert resp["error"]["code"] == 4002
    assert "skill_name or active session with role required" in resp["error"]["message"]


def test_xp_status_returns_error_on_exception():
    """xp.status must surface unexpected errors as RPC error 5029."""
    with patch("agent.gamification.KPITracker", side_effect=RuntimeError("boom")):
        resp = server.handle_request(
            {"id": "r15", "method": "xp.status", "params": {"skill_name": "devops"}}
        )

    assert resp["error"]["code"] == 5029


# ---------------------------------------------------------------------------
# achievements.list
# ---------------------------------------------------------------------------

def test_achievements_list_returns_achievements():
    """achievements.list must return the mocked achievement list."""
    with patch("agent.gamification.KPITracker") as MockKPI:
        MockKPI.return_value = _mock_kpi_tracker()
        resp = server.handle_request(
            {"id": "r16", "method": "achievements.list", "params": {"role": "devops"}}
        )

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "r16"
    achievements = resp["result"]["achievements"]
    assert len(achievements) == 1
    assert achievements[0]["achievement_id"] == "first_win"


def test_achievements_list_returns_error_on_exception():
    """achievements.list must surface unexpected errors as RPC error 5030."""
    with patch("agent.gamification.KPITracker", side_effect=RuntimeError("boom")):
        resp = server.handle_request(
            {"id": "r17", "method": "achievements.list", "params": {}}
        )

    assert resp["error"]["code"] == 5030


# ---------------------------------------------------------------------------
# leaderboard
# ---------------------------------------------------------------------------

def test_leaderboard_returns_ranked_list():
    """leaderboard must return the mocked leaderboard."""
    with patch("agent.gamification.KPITracker") as MockKPI:
        MockKPI.return_value = _mock_kpi_tracker()
        resp = server.handle_request(
            {"id": "r18", "method": "leaderboard", "params": {"limit": 5}}
        )

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "r18"
    board = resp["result"]["leaderboard"]
    assert len(board) == 2
    assert board[0]["rank"] == 1
    assert board[0]["skill_name"] == "devops"


def test_leaderboard_returns_error_on_exception():
    """leaderboard must surface unexpected errors as RPC error 5031."""
    with patch("agent.gamification.KPITracker", side_effect=RuntimeError("boom")):
        resp = server.handle_request(
            {"id": "r19", "method": "leaderboard", "params": {}}
        )

    assert resp["error"]["code"] == 5031
