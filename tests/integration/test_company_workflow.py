"""Integration tests for the full CEO company-OS workflow.

Exercises the end-to-end pipeline:
  AgentManager -> TeamManager -> ProjectManager -> ReleaseManager

All tests use the _hermetic_environment autouse fixture (from conftest.py)
and create a fresh SessionDB per test to avoid cross-contamination.
Subprocess calls are mocked so no real git or test scripts are executed.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_manager import AgentManager
from agent.project_manager import ProjectManager
from agent.release_manager import ReleaseManager
from agent.team_manager import TeamManager
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Path) -> SessionDB:
    """Create a fresh SessionDB at an explicit path under tmp_path."""
    db_path = tmp_path / "test.db"
    return SessionDB(db_path=db_path)


# ---------------------------------------------------------------------------
# Mock subprocess results
# ---------------------------------------------------------------------------

mock_pass = MagicMock()
mock_pass.returncode = 0
mock_pass.stdout = "All tests passed"
mock_pass.stderr = ""

mock_fail = MagicMock()
mock_fail.returncode = 1
mock_fail.stdout = "2 tests failed"
mock_fail.stderr = "AssertionError in test_foo"

mock_tag = MagicMock()
mock_tag.returncode = 0
mock_tag.stdout = ""
mock_tag.stderr = ""


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestCompanyWorkflow:
    """Full CEO workflow: agents -> teams -> projects -> releases."""

    # ── Test 1: Full E2E CEO workflow ──────────────────────────────────────

    def test_full_ceo_workflow(self, tmp_path):
        db = _make_db(tmp_path)
        am = AgentManager(db=db)
        tm = TeamManager(db=db)
        pm = ProjectManager(db=db)
        rm = ReleaseManager(db=db)

        # --- Create agents ---
        alice = am.create_agent("alice", "fullstack-dev")
        assert alice["id"] == "alice"
        assert alice["role"] == "fullstack-dev"
        assert alice["status"] == "active"

        bob = am.create_agent("bob", "devops")
        assert bob["id"] == "bob"
        assert bob["role"] == "devops"
        assert bob["status"] == "active"

        # --- Create team ---
        team = tm.create_team("web-team", "Web Team")
        assert team["id"] == "web-team"
        assert team["name"] == "Web Team"

        # --- Add members ---
        membership = tm.add_member("web-team", "alice")
        assert membership["agent_id"] == "alice"
        assert membership["team_id"] == "web-team"

        tm.add_member("web-team", "bob")
        members = tm.get_members("web-team")
        assert len(members) == 2
        member_ids = {m["agent_id"] for m in members}
        assert member_ids == {"alice", "bob"}

        # --- Create project ---
        # Use a git-tag-safe project name (no spaces or special characters).
        from pathlib import Path as _Path
        _raw_repo = "/tmp/fake-repo"
        _expected_repo = str(_Path(_raw_repo).expanduser().resolve())
        project = pm.create_project(
            "hermes-v2", "hermes-v2-launch", repo_path=_raw_repo
        )
        assert project["id"] == "hermes-v2"
        assert project["name"] == "hermes-v2-launch"
        assert project["status"] == "proposed"
        assert project["repo_path"] == _expected_repo

        # --- Assign team to project ---
        result = pm.assign_team("hermes-v2", "web-team")
        assert result is True
        teams = pm.get_project_teams("hermes-v2")
        assert len(teams) == 1
        assert teams[0]["team_id"] == "web-team"
        assert teams[0]["name"] == "Web Team"

        # --- Create release ---
        release = rm.create_release("hermes-v2", "0.1.0")
        assert release["project_id"] == "hermes-v2"
        assert release["version"] == "0.1.0"
        assert release["status"] == "draft"
        assert release["git_tag"] is None
        release_id = release["id"]

        # --- Run check (mocked subprocess: tests pass) ---
        with patch("agent.release_manager.subprocess.run", return_value=mock_pass):
            check = rm.run_check(release_id, "tests")
        assert check["status"] == "passed"
        assert check["check_type"] == "tests"

        # Verify release transitioned to 'testing'
        release_after_check = rm.get_release(release_id)
        assert release_after_check["status"] == "testing"

        # --- Ship release (mocked subprocess: git tag succeeds) ---
        with patch("agent.release_manager.subprocess.run", return_value=mock_tag):
            shipped = rm.ship_release(release_id)
        assert shipped["status"] == "shipped"
        assert shipped["git_tag"] == "hermes-v2-launch/v0.1.0"
        assert shipped["shipped_at"] is not None

        # --- Verify project status changed to 'active' ---
        updated_project = pm.get_project("hermes-v2")
        assert updated_project["status"] == "active"

        # --- Full project info ---
        info = pm.get_project_info("hermes-v2")
        assert info is not None
        assert info["id"] == "hermes-v2"
        assert len(info["teams"]) == 1

        # --- Direct DB assertions ---
        with db._lock:
            count = db._conn.execute(
                "SELECT COUNT(*) AS cnt FROM agents WHERE id IN ('alice', 'bob')"
            ).fetchone()
        assert count["cnt"] == 2

        with db._lock:
            count = db._conn.execute(
                "SELECT COUNT(*) AS cnt FROM team_members WHERE team_id = 'web-team'"
            ).fetchone()
        assert count["cnt"] == 2

        with db._lock:
            count = db._conn.execute(
                "SELECT COUNT(*) AS cnt FROM project_teams WHERE project_id = 'hermes-v2'"
            ).fetchone()
        assert count["cnt"] == 1

        with db._lock:
            row = db._conn.execute(
                "SELECT status, git_tag FROM releases WHERE project_id = 'hermes-v2'"
            ).fetchone()
        assert row["status"] == "shipped"
        assert row["git_tag"] == "hermes-v2-launch/v0.1.0"

    # ── Test 2: Agent multi-team membership ─────────────────────────────────

    def test_agent_multi_team_membership(self, tmp_path):
        db = _make_db(tmp_path)
        am = AgentManager(db=db)
        tm = TeamManager(db=db)

        am.create_agent("flex", "fullstack-dev")
        tm.create_team("team-alpha", "Alpha Team")
        tm.create_team("team-beta", "Beta Team")

        tm.add_member("team-alpha", "flex")
        tm.add_member("team-beta", "flex")

        teams = am.get_agent_teams("flex")
        assert len(teams) == 2
        team_ids = {t["team_id"] for t in teams}
        assert team_ids == {"team-alpha", "team-beta"}

    # ── Test 3: Team lead workflow ──────────────────────────────────────────

    def test_team_lead_workflow(self, tmp_path):
        db = _make_db(tmp_path)
        am = AgentManager(db=db)
        tm = TeamManager(db=db)

        am.create_agent("alice", "fullstack-dev")
        tm.create_team("web-team", "Web Team")

        tm.add_member("web-team", "alice")

        # Set lead
        team = tm.set_lead("web-team", "alice")
        assert team["lead_agent_id"] == "alice"

        # Verify via get_team
        fetched = tm.get_team("web-team")
        assert fetched["lead_agent_id"] == "alice"

        # Remove lead member — lead should be cleared
        removed = tm.remove_member("web-team", "alice")
        assert removed is True

        fetched = tm.get_team("web-team")
        assert fetched["lead_agent_id"] is None

    # ── Test 4: Ship without checks raises ValueError ────────────────────────

    def test_release_ship_requires_checks(self, tmp_path):
        db = _make_db(tmp_path)
        pm = ProjectManager(db=db)
        rm = ReleaseManager(db=db)

        pm.create_project("proj-x", "Project X", repo_path="/tmp/fake-repo")
        release = rm.create_release("proj-x", "1.0.0")

        with pytest.raises(ValueError, match="run checks first"):
            rm.ship_release(release["id"])

    # ── Test 5: Failed check blocks ship ────────────────────────────────────

    def test_release_failed_check_blocks_ship(self, tmp_path):
        db = _make_db(tmp_path)
        pm = ProjectManager(db=db)
        rm = ReleaseManager(db=db)

        pm.create_project("proj-y", "Project Y", repo_path="/tmp/fake-repo")
        release = rm.create_release("proj-y", "2.0.0")
        release_id = release["id"]

        # Run check with mocked subprocess returning failure
        with patch("agent.release_manager.subprocess.run", return_value=mock_fail):
            check = rm.run_check(release_id, "tests")
        assert check["status"] == "failed"

        # Attempting to ship should raise ValueError
        with pytest.raises(ValueError, match="checks not passed"):
            rm.ship_release(release_id)

    # ── Test 6: Default agents seeded on fresh DB ───────────────────────────

    def test_default_agents_seeded(self, tmp_path):
        db = _make_db(tmp_path)

        expected_ids = {"dev-1", "devops-1", "quant-1", "prop-1", "content-1", "eng-1"}

        with db._lock:
            cursor = db._conn.execute("SELECT id FROM agents")
            rows = cursor.fetchall()
        actual_ids = {row["id"] for row in rows}

        assert expected_ids.issubset(actual_ids), (
            f"Missing default agents: {expected_ids - actual_ids}"
        )

        # Verify each default agent has status='active' and a valid role
        with db._lock:
            cursor = db._conn.execute(
                "SELECT id, role, status FROM agents WHERE id IN (?, ?, ?, ?, ?, ?)",
                tuple(expected_ids),
            )
            rows = cursor.fetchall()

        for row in rows:
            assert row["status"] == "active"
            assert row["role"] is not None