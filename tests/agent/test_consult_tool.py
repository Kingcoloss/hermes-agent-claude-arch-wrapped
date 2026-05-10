"""Tests for tools/consult_tool.py."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from agent.consultation_manager import ConsultationManager
from agent.agent_manager import AgentManager
from agent.team_manager import TeamManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_consult_tool.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


@pytest.fixture()
def agent_mgr(db):
    return AgentManager(db=db)


@pytest.fixture()
def team_mgr(db):
    return TeamManager(db=db)


@pytest.fixture()
def consult_mgr(db):
    return ConsultationManager(db=db, max_depth=2, cost_cap=5)


def _make_agent(db: SessionDB, agent_id: str, status: str = "active") -> None:
    now = time.time()
    db._execute_write(
        lambda conn: conn.execute(
            "INSERT OR IGNORE INTO agents (id, role, status, created_at) VALUES (?, ?, ?, ?)",
            (agent_id, "fullstack-dev", status, now),
        )
    )


def _canned_subagent_response(output_text: str) -> str:
    """Return a JSON string matching claude_subagent's happy-path return value."""
    return json.dumps({"success": True, "output": output_text, "duration_seconds": 0.1})


# ---------------------------------------------------------------------------
# consult_agent
# ---------------------------------------------------------------------------


class TestConsultAgentTool:
    def test_happy_path(self, db, consult_mgr):
        """consult_agent returns consultation_id + response on success."""
        _make_agent(db, "alice")

        canned = _canned_subagent_response("Alice's perspective")

        with (
            patch("tools.consult_tool._get_manager", return_value=consult_mgr),
            patch("tools.consult_tool._get_agent_manager") as mock_am,
            patch("tools.consult_tool._invoke_agent", return_value="Alice's perspective"),
        ):
            mock_am.return_value.get_agent.return_value = {
                "id": "alice", "role": "fullstack-dev", "status": "active", "created_at": 0.0
            }

            from tools.consult_tool import consult_agent_handler
            result_str = consult_agent_handler(
                target_agent_id="alice",
                question="What is your estimate?",
                context_summary="We are at 60% done.",
                session_id="sess-test",
            )

        result = json.loads(result_str)
        assert result.get("consultation_id") is not None
        assert result["response"] == "Alice's perspective"
        assert result["status"] == "done"

    def test_target_not_found_returns_tool_error(self, db, consult_mgr):
        with (
            patch("tools.consult_tool._get_manager", return_value=consult_mgr),
            patch("tools.consult_tool._get_agent_manager") as mock_am,
        ):
            mock_am.return_value.get_agent.return_value = None  # not found

            from tools.consult_tool import consult_agent_handler
            result_str = consult_agent_handler(
                target_agent_id="ghost",
                question="Q",
                context_summary="ctx",
                session_id="sess-x",
            )

        result = json.loads(result_str)
        assert result.get("success") is False
        assert "ghost" in result.get("error", "").lower() or "not found" in result.get("error", "").lower()

    def test_cost_cap_propagates_as_failed(self, db, consult_mgr):
        """When the manager raises ValueError (cost cap), the tool returns a failed-status result."""
        _make_agent(db, "target-cap")

        # Fill up the cost cap
        for i in range(5):
            _make_agent(db, f"filler-{i}")
            c = consult_mgr.create_consultation(
                caller_session_id="sess-cap-tool",
                target_agent_id=f"filler-{i}",
                question="Q",
                context_summary="ctx",
            )
            consult_mgr.complete_consultation(c["id"], "resp")

        with (
            patch("tools.consult_tool._get_manager", return_value=consult_mgr),
            patch("tools.consult_tool._get_agent_manager") as mock_am,
        ):
            mock_am.return_value.get_agent.return_value = {
                "id": "target-cap", "role": "fullstack-dev", "status": "active", "created_at": 0.0
            }

            from tools.consult_tool import consult_agent_handler
            result_str = consult_agent_handler(
                target_agent_id="target-cap",
                question="Q",
                context_summary="ctx",
                session_id="sess-cap-tool",
            )

        result = json.loads(result_str)
        # The guard failure surfaces as a tool_error (success=False) or failed consultation
        assert result.get("success") is False or result.get("status") == "failed"


# ---------------------------------------------------------------------------
# consult_panel
# ---------------------------------------------------------------------------


class TestConsultPanelTool:
    def test_panel_three_targets_all_consulted(self, db, consult_mgr):
        for aid in ("p1", "p2", "p3"):
            _make_agent(db, aid)

        def _fake_invoke(agent_id, question, context_summary):
            return f"Response from {agent_id}"

        with (
            patch("tools.consult_tool._get_manager", return_value=consult_mgr),
            patch("tools.consult_tool._invoke_agent", side_effect=_fake_invoke),
        ):
            from tools.consult_tool import consult_panel_handler
            result_str = consult_panel_handler(
                target_agent_ids=["p1", "p2", "p3"],
                question="Panel question",
                context_summary="ctx",
                session_id="sess-panel",
            )

        result = json.loads(result_str)
        consultations = result["consultations"]
        assert len(consultations) == 3
        agent_ids = {c["agent_id"] for c in consultations}
        assert agent_ids == {"p1", "p2", "p3"}
        for c in consultations:
            assert c["status"] == "done"
            assert c["consultation_id"] is not None


# ---------------------------------------------------------------------------
# consult_team
# ---------------------------------------------------------------------------


class TestConsultTeamTool:
    def _setup_team(self, db, team_mgr, agent_mgr):
        """Create a team with lead 'lead-x' and non-lead member 'member-x'."""
        _make_agent(db, "lead-x")
        _make_agent(db, "member-x")

        db._execute_write(
            lambda conn: conn.execute(
                "INSERT INTO teams (id, name, lead_agent_id, created_at) VALUES (?, ?, ?, ?)",
                ("team-x", "Team X", "lead-x", time.time()),
            )
        )
        for aid in ("lead-x", "member-x"):
            db._execute_write(
                lambda conn, a=aid: conn.execute(
                    "INSERT INTO team_members (team_id, agent_id, joined_at, status) VALUES (?, ?, ?, 'active')",
                    ("team-x", a, time.time()),
                )
            )

    def test_include_lead_false_excludes_lead(self, db, consult_mgr, team_mgr, agent_mgr):
        self._setup_team(db, team_mgr, agent_mgr)

        def _fake_invoke(agent_id, question, context_summary):
            return f"Response from {agent_id}"

        with (
            patch("tools.consult_tool._get_manager", return_value=consult_mgr),
            patch("tools.consult_tool._get_team_manager", return_value=team_mgr),
            patch("tools.consult_tool._invoke_agent", side_effect=_fake_invoke),
        ):
            from tools.consult_tool import consult_team_handler
            result_str = consult_team_handler(
                team_id="team-x",
                question="Are you available for feature X?",
                context_summary="ctx",
                include_lead=False,
                session_id="sess-team",
            )

        result = json.loads(result_str)
        assert result["team_id"] == "team-x"
        consultations = result["consultations"]
        # Only member-x should be consulted (not lead-x)
        agent_ids = {c["agent_id"] for c in consultations}
        assert "lead-x" not in agent_ids
        assert "member-x" in agent_ids

    def test_team_not_found_returns_error(self, db, consult_mgr, team_mgr):
        with (
            patch("tools.consult_tool._get_manager", return_value=consult_mgr),
            patch("tools.consult_tool._get_team_manager", return_value=team_mgr),
        ):
            from tools.consult_tool import consult_team_handler
            result_str = consult_team_handler(
                team_id="no-such-team",
                question="Q",
                context_summary="ctx",
                session_id="sess-x",
            )

        result = json.loads(result_str)
        assert result.get("success") is False
