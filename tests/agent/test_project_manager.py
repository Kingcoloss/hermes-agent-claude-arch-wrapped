#!/usr/bin/env python3
"""Unit tests for agent.project_manager.ProjectManager."""

import time

import pytest

from agent.project_manager import ProjectManager, get_project_manager
from hermes_constants import get_hermes_home
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fresh_db():
    """Create a SessionDB with an explicit per-test path under HERMES_HOME."""
    return SessionDB(db_path=get_hermes_home() / "state.db")


def _insert_team(db: SessionDB, team_id: str, name: str):
    """Insert a team row directly so FK constraints on project_teams pass."""
    now = time.time()

    def _do(conn):
        conn.execute(
            "INSERT OR IGNORE INTO teams (id, name, created_at) VALUES (?, ?, ?)",
            (team_id, name, now),
        )

    db._execute_write(_do)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """Provide a fresh SessionDB for each test."""
    return _make_fresh_db()


@pytest.fixture
def pm(db):
    """Provide a ProjectManager backed by the fresh test DB."""
    return ProjectManager(db=db)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_project(pm):
    """Create project with default repo_path (cwd), then get returns it."""
    result = pm.create_project("proj-1", "My Project")
    assert result is not None
    assert result["id"] == "proj-1"
    assert result["name"] == "My Project"
    assert result["status"] == "proposed"
    assert result["repo_path"]  # defaults to cwd
    assert result["created_at"] is not None
    assert result["completed_at"] is None

    # get_project returns the same thing
    fetched = pm.get_project("proj-1")
    assert fetched["id"] == "proj-1"
    assert fetched["name"] == "My Project"


def test_create_project_custom_repo(pm, tmp_path):
    """Pass repo_path explicitly; stored path is the resolved absolute path."""
    from pathlib import Path
    raw = str(tmp_path / "my-repo")
    expected = str(Path(raw).expanduser().resolve())
    result = pm.create_project("proj-2", "Custom Repo", repo_path=raw)
    assert result["repo_path"] == expected

    fetched = pm.get_project("proj-2")
    assert fetched["repo_path"] == expected


def test_create_duplicate_raises(pm):
    """Creating a project with the same id twice raises ValueError."""
    pm.create_project("dup-1", "First")
    with pytest.raises(ValueError, match="already exists"):
        pm.create_project("dup-1", "Second")


def test_list_projects(pm):
    """Create 2 projects, list returns both."""
    pm.create_project("list-1", "Project A")
    pm.create_project("list-2", "Project B")
    projects = pm.list_projects()
    assert len(projects) == 2
    ids = {p["id"] for p in projects}
    assert ids == {"list-1", "list-2"}


def test_assign_team(pm, db):
    """Assign a team, then get_project_teams shows it."""
    _insert_team(db, "team-alpha", "Alpha Team")
    pm.create_project("proj-t1", "Team Project")

    result = pm.assign_team("proj-t1", "team-alpha")
    assert result is True

    teams = pm.get_project_teams("proj-t1")
    assert len(teams) == 1
    assert teams[0]["team_id"] == "team-alpha"
    assert teams[0]["name"] == "Alpha Team"


def test_assign_nonexistent_team_raises(pm):
    """Assigning a team not in the teams table raises ValueError."""
    pm.create_project("proj-t2", "Bad Team Project")
    with pytest.raises(ValueError, match="does not exist"):
        pm.assign_team("proj-t2", "ghost-team")


def test_unassign_team(pm, db):
    """Unassign a team, get_project_teams returns empty."""
    _insert_team(db, "team-bravo", "Bravo Team")
    pm.create_project("proj-t3", "Unassign Project")
    pm.assign_team("proj-t3", "team-bravo")

    removed = pm.unassign_team("proj-t3", "team-bravo")
    assert removed is True

    teams = pm.get_project_teams("proj-t3")
    assert len(teams) == 0

    # Unassigning again returns False (was not assigned)
    removed_again = pm.unassign_team("proj-t3", "team-bravo")
    assert removed_again is False


def test_complete_project(pm):
    """Completing a project sets status and completed_at."""
    pm.create_project("proj-done", "Done Project")
    # Transition to active first (not required by complete_project but realistic)
    pm.update_project("proj-done", status="active")

    result = pm.complete_project("proj-done")
    assert result["status"] == "completed"
    assert result["completed_at"] is not None


def test_complete_already_completed_raises(pm):
    """Completing an already-completed project raises ValueError."""
    pm.create_project("proj-twice", "Twice Project")
    pm.update_project("proj-twice", status="active")
    pm.complete_project("proj-twice")

    with pytest.raises(ValueError, match="already completed"):
        pm.complete_project("proj-twice")


def test_get_project_info(pm, db):
    """get_project_info returns full project dict with teams list."""
    _insert_team(db, "team-charlie", "Charlie Team")
    _insert_team(db, "team-delta", "Delta Team")
    pm.create_project("proj-info", "Info Project")
    pm.assign_team("proj-info", "team-charlie")
    pm.assign_team("proj-info", "team-delta")

    info = pm.get_project_info("proj-info")
    assert info is not None
    assert info["id"] == "proj-info"
    assert info["name"] == "Info Project"
    assert "teams" in info
    assert len(info["teams"]) == 2
    team_names = {t["name"] for t in info["teams"]}
    assert team_names == {"Charlie Team", "Delta Team"}


# ---------------------------------------------------------------------------
# Fix #5 — repo_path normalization
# ---------------------------------------------------------------------------

def test_create_project_normalizes_repo_path(pm, tmp_path):
    """Fix #5: repo_path with '~' or '..' is stored as resolved absolute path."""
    # Build a path with a dotdot component and resolve it ourselves.
    nested = tmp_path / "sub" / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    raw_path = str(nested / ".." / "sibling")
    expected = str((nested / ".." / "sibling").resolve())

    result = pm.create_project("norm-1", "Norm Project", repo_path=raw_path)
    assert result["repo_path"] == expected
    # Confirm it's an absolute path with no dotdot.
    assert ".." not in result["repo_path"]
    assert result["repo_path"].startswith("/")


def test_update_project_normalizes_repo_path(pm, tmp_path):
    """Fix #5: update_project with repo_path resolves it to absolute."""
    pm.create_project("norm-2", "Norm Project 2")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True, exist_ok=True)
    raw_path = str(nested / ".." / "c")
    expected = str((nested / ".." / "c").resolve())

    updated = pm.update_project("norm-2", repo_path=raw_path)
    assert updated["repo_path"] == expected
    assert ".." not in updated["repo_path"]