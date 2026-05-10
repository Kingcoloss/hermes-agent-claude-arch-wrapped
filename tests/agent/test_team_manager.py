"""Tests for agent/team_manager.py — TeamManager CRUD operations."""

import time

import pytest

from agent.team_manager import TeamManager, get_team_manager
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    """Fresh SessionDB per test."""
    from hermes_constants import get_hermes_home
    db_path = tmp_path / "test_state.db"
    database = SessionDB(db_path=db_path)
    yield database
    database.close()


@pytest.fixture
def tm(db):
    """TeamManager backed by a fresh SessionDB."""
    return TeamManager(db=db)


def _insert_agent(db, agent_id, role="dev", status="active"):
    """Helper: insert a row into the agents table."""
    db._execute_write(
        lambda c: c.execute(
            "INSERT INTO agents (id, role, status, created_at) VALUES (?, ?, ?, ?)",
            (agent_id, role, status, time.time()),
        )
    )


# ── test_create_team ──

def test_create_team(tm):
    tm.create_team("alpha", "Alpha Team")
    team = tm.get_team("alpha")
    assert team is not None
    assert team["id"] == "alpha"
    assert team["name"] == "Alpha Team"
    assert team["lead_agent_id"] is None


# ── test_create_duplicate_raises ──

def test_create_duplicate_raises(tm):
    tm.create_team("alpha", "Alpha Team")
    with pytest.raises(ValueError, match="already exists"):
        tm.create_team("alpha", "Another Alpha")


# ── test_add_member ──

def test_add_member(tm, db):
    _insert_agent(db, "alice")
    tm.create_team("alpha", "Alpha Team")
    member = tm.add_member("alpha", "alice")
    assert member["agent_id"] == "alice"
    assert member["status"] == "active"
    members = tm.get_members("alpha")
    assert len(members) == 1
    assert members[0]["agent_id"] == "alice"
    assert members[0]["role"] == "dev"


# ── test_add_member_idempotent ──

def test_add_member_idempotent(tm, db):
    _insert_agent(db, "alice")
    tm.create_team("alpha", "Alpha Team")
    tm.add_member("alpha", "alice")
    tm.add_member("alpha", "alice")  # should not raise
    members = tm.get_members("alpha")
    assert len(members) == 1


# ── test_add_nonexistent_agent_raises ──

def test_add_nonexistent_agent_raises(tm):
    tm.create_team("alpha", "Alpha Team")
    with pytest.raises(ValueError, match="does not exist"):
        tm.add_member("alpha", "ghost_agent")


# ── test_remove_member ──

def test_remove_member(tm, db):
    _insert_agent(db, "alice")
    tm.create_team("alpha", "Alpha Team")
    tm.add_member("alpha", "alice")
    removed = tm.remove_member("alpha", "alice")
    assert removed is True
    members = tm.get_members("alpha")
    assert len(members) == 0


# ── test_set_lead ──

def test_set_lead(tm, db):
    _insert_agent(db, "alice")
    tm.create_team("alpha", "Alpha Team")
    tm.add_member("alpha", "alice")
    team = tm.set_lead("alpha", "alice")
    assert team["lead_agent_id"] == "alice"


# ── test_set_lead_non_member_raises ──

def test_set_lead_non_member_raises(tm, db):
    _insert_agent(db, "bob")
    tm.create_team("alpha", "Alpha Team")
    with pytest.raises(ValueError, match="not a member"):
        tm.set_lead("alpha", "bob")


# ── test_remove_lead_clears ──

def test_remove_lead_clears(tm, db):
    _insert_agent(db, "alice")
    tm.create_team("alpha", "Alpha Team")
    tm.add_member("alpha", "alice")
    tm.set_lead("alpha", "alice")

    # Remove the lead — lead_agent_id should become None
    tm.remove_member("alpha", "alice")
    team = tm.get_team("alpha")
    assert team["lead_agent_id"] is None


# ── test_get_team_info ──

def test_get_team_info(tm, db):
    _insert_agent(db, "alice", role="lead")
    _insert_agent(db, "bob", role="dev")
    tm.create_team("alpha", "Alpha Team")
    tm.add_member("alpha", "alice")
    tm.add_member("alpha", "bob")
    tm.set_lead("alpha", "alice")

    info = tm.get_team_info("alpha")
    assert info is not None
    assert info["team"]["id"] == "alpha"
    assert len(info["members"]) == 2
    assert info["lead"] is not None
    assert info["lead"]["id"] == "alice"
    assert info["lead"]["role"] == "lead"


# ── test_readd_inactive_member ──

def test_readd_inactive_member_becomes_active(tm, db):
    """Fix #4: re-adding an inactive member must set status back to 'active'."""
    _insert_agent(db, "charlie")
    tm.create_team("gamma", "Gamma Team")
    tm.add_member("gamma", "charlie")

    # Manually mark the member as inactive (simulating a remove/soft-delete).
    db._execute_write(
        lambda c: c.execute(
            "UPDATE team_members SET status = 'inactive' "
            "WHERE team_id = 'gamma' AND agent_id = 'charlie'"
        )
    )

    # Verify inactive state.
    with db._lock:
        row = db._conn.execute(
            "SELECT status FROM team_members WHERE team_id = 'gamma' AND agent_id = 'charlie'"
        ).fetchone()
    assert row["status"] == "inactive"

    # Re-add — should become active again.
    member = tm.add_member("gamma", "charlie")
    assert member["status"] == "active"

    # Confirm there is still only one row (no duplicate).
    with db._lock:
        count = db._conn.execute(
            "SELECT COUNT(*) FROM team_members WHERE team_id = 'gamma' AND agent_id = 'charlie'"
        ).fetchone()[0]
    assert count == 1


# ── test_singleton ──

def test_get_team_manager_returns_singleton():
    """get_team_manager returns the same instance on repeated calls."""
    import agent.team_manager as mod
    old = mod._team_manager
    mod._team_manager = None
    try:
        tm1 = get_team_manager()
        tm2 = get_team_manager()
        assert tm1 is tm2
    finally:
        mod._team_manager = old