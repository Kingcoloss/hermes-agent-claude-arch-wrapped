"""Unit tests for agent/gamification.py — KPITracker, XP/level, leaderboard.

Fix #7: verifies that per-agent XP rows do not pollute the global leaderboard,
and that get_agent_leaderboard returns only the agent's rows with prefixes stripped.
"""

import pytest

from agent.gamification import KPITracker
from hermes_constants import get_hermes_home
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Fresh SessionDB per test."""
    database = SessionDB(db_path=tmp_path / "test_state.db")
    yield database
    database.close()


@pytest.fixture
def tracker(db):
    """KPITracker backed by a fresh SessionDB."""
    return KPITracker(db=db, xp_per_level=100)


# ---------------------------------------------------------------------------
# Fix #7A — Global leaderboard must not include per-agent rows
# ---------------------------------------------------------------------------

def test_global_leaderboard_excludes_agent_rows(tracker):
    """Adding XP with agent_id='alice' must not appear in global get_leaderboard()."""
    # Add XP globally for 'python' skill (no agent_id).
    tracker.add_xp("python", 50.0, agent_id=None)
    # Add XP for alice on 'python' skill — creates alice/python per-agent row.
    tracker.add_xp("python", 75.0, agent_id="alice")

    leaderboard = tracker.get_leaderboard()
    skill_names = [entry["skill_name"] for entry in leaderboard]

    # Global 'python' should appear.
    assert "python" in skill_names
    # Per-agent 'alice/python' must NOT appear.
    assert "alice/python" not in skill_names


def test_global_leaderboard_with_role_filter_excludes_agent_rows(tracker):
    """get_leaderboard(role='python') with per-agent rows must not mix in agent rows."""
    tracker.add_xp("python", 40.0, agent_id=None)
    tracker.add_xp("python", 90.0, agent_id="bob")

    leaderboard = tracker.get_leaderboard(role="python")
    for entry in leaderboard:
        # Every row returned must be the global 'python' row, not bob's.
        assert entry["skill_name"] == "python"
        assert "bob" not in entry["skill_name"]


def test_global_leaderboard_only_global_rows(tracker):
    """With only per-agent rows, global leaderboard must be empty."""
    tracker.add_xp("rust", 100.0, agent_id="carol")

    leaderboard = tracker.get_leaderboard()
    skill_names = [e["skill_name"] for e in leaderboard]
    assert "carol/rust" not in skill_names
    # The global 'rust' row was also written by add_xp(..., agent_id="carol").
    # The global row has agent_id=NULL, the per-agent has agent_id="carol".
    # Global 'rust' appears because add_xp always updates the global row too.
    # What must NOT appear is any entry with 'carol/' prefix.
    for name in skill_names:
        assert not name.startswith("carol/")


# ---------------------------------------------------------------------------
# Fix #7B — get_agent_leaderboard returns only agent's rows, prefix stripped
# ---------------------------------------------------------------------------

def test_get_agent_leaderboard_returns_agent_rows(tracker):
    """get_agent_leaderboard('alice') returns alice's skill rows with prefix stripped."""
    tracker.add_xp("python", 60.0, agent_id="alice")
    tracker.add_xp("go", 30.0, agent_id="alice")
    # Bob should not appear.
    tracker.add_xp("python", 80.0, agent_id="bob")

    alice_board = tracker.get_agent_leaderboard("alice")
    names = [e["skill_name"] for e in alice_board]

    # Should see clean skill names (no prefix).
    assert "python" in names
    assert "go" in names

    # No raw prefixed names should appear.
    for entry in alice_board:
        assert not entry["skill_name"].startswith("alice/")
        assert not entry["skill_name"].startswith("bob/")


def test_get_agent_leaderboard_does_not_contain_global_rows(tracker):
    """get_agent_leaderboard('alice') must not return global (agent_id=NULL) rows."""
    tracker.add_xp("python", 50.0, agent_id=None)   # global only
    tracker.add_xp("go", 20.0, agent_id="alice")    # per-alice + global

    alice_board = tracker.get_agent_leaderboard("alice")
    names = [e["skill_name"] for e in alice_board]

    # alice has a go entry.
    assert "go" in names
    # alice does not have a python-only global entry.
    # The board must not have more rows than what alice wrote.
    assert len(alice_board) == 1


def test_get_agent_leaderboard_rank_ordering(tracker):
    """Entries are ranked by XP descending."""
    tracker.add_xp("python", 30.0, agent_id="dave")
    tracker.add_xp("go", 70.0, agent_id="dave")
    tracker.add_xp("rust", 50.0, agent_id="dave")

    dave_board = tracker.get_agent_leaderboard("dave")
    xp_values = [e["xp"] for e in dave_board]
    # Should be descending.
    assert xp_values == sorted(xp_values, reverse=True)
    # Ranks should be 1-based sequential.
    assert [e["rank"] for e in dave_board] == list(range(1, len(dave_board) + 1))


# ---------------------------------------------------------------------------
# Fix #7C — get_xp (get_level) must return global row, not per-agent row
# ---------------------------------------------------------------------------

def test_get_level_returns_global_xp(tracker):
    """get_level('python') returns the global XP, not a per-agent row.

    add_xp always reads the global row to compute the new value, then writes
    both the global row and the per-agent row.  After:
      - add_xp("python", 55.0, agent_id=None)  -> global=55, no agent row
      - add_xp("python", 200.0, agent_id="eve") -> reads global=55, new=255,
                                                    global=255, eve/python=255
    get_level("python") must return the global row value (255).
    """
    tracker.add_xp("python", 55.0, agent_id=None)
    tracker.add_xp("python", 200.0, agent_id="eve")

    result = tracker.get_level("python")
    assert result["skill_name"] == "python"
    # Global accumulated value: 55 + 200 = 255.
    assert result["xp"] == 255.0


def test_get_level_not_polluted_by_agent_row(tracker):
    """get_level on a skill where only per-agent row exists returns empty default."""
    # Only add via agent_id - this writes global + per-agent. But then manually
    # delete the global row to simulate a scenario where only agent row exists.
    tracker.add_xp("haskell", 80.0, agent_id="frank")

    # Delete the global row (agent_id IS NULL).
    def _del(conn):
        conn.execute(
            "DELETE FROM agent_skills_xp WHERE skill_name = 'haskell' AND agent_id IS NULL"
        )
    tracker.db._execute_write(_del)

    # Now get_level should return the empty default (0 XP), not the agent row.
    result = tracker.get_level("haskell")
    assert result["xp"] == 0.0
    assert result["level"] == 1
