"""Tests for agent/agent_manager.py — CRUD for persistent agent identities."""

from pathlib import Path

from hermes_state import SessionDB


# _hermetic_environment (autouse) redirects HERMES_HOME to a temp dir,
# so SessionDB() creates its database there instead of at ~/.hermes.


def _fresh_db(tmp_path: Path) -> SessionDB:
    """Create a SessionDB at a temporary path (full schema included)."""
    db_path = tmp_path / "test_state.db"
    return SessionDB(db_path)


# ---------------------------------------------------------------------------
# test_create_agent
# ---------------------------------------------------------------------------


def test_create_agent(tmp_path):
    """Create an agent, then get_agent returns it with all expected keys."""
    db = _fresh_db(tmp_path)
    mgr = __import__("agent.agent_manager", fromlist=["AgentManager"]).AgentManager(db=db)

    agent = mgr.create_agent("alpha-01", "fullstack-dev")

    assert agent is not None
    assert agent["id"] == "alpha-01"
    assert agent["role"] == "fullstack-dev"
    assert agent["status"] == "active"
    assert "created_at" in agent

    # Confirm via get_agent
    fetched = mgr.get_agent("alpha-01")
    assert fetched is not None
    assert fetched["id"] == "alpha-01"
    assert fetched["role"] == "fullstack-dev"


# ---------------------------------------------------------------------------
# test_create_duplicate_raises
# ---------------------------------------------------------------------------


def test_create_duplicate_raises(tmp_path):
    """Creating an agent with an existing id raises ValueError."""
    db = _fresh_db(tmp_path)
    mgr = __import__("agent.agent_manager", fromlist=["AgentManager"]).AgentManager(db=db)

    mgr.create_agent("dup-agent", "devops")

    try:
        mgr.create_agent("dup-agent", "quant-trader")
        assert False, "Expected ValueError for duplicate agent_id"
    except ValueError as exc:
        assert "already exists" in str(exc).lower()


# ---------------------------------------------------------------------------
# test_invalid_agent_id
# ---------------------------------------------------------------------------


def test_invalid_agent_id(tmp_path):
    """Agent IDs with spaces or uppercase are rejected."""
    db = _fresh_db(tmp_path)
    mgr = __import__("agent.agent_manager", fromlist=["AgentManager"]).AgentManager(db=db)

    for bad_id in ("has space", "UpperCase", "no!special", ""):
        try:
            mgr.create_agent(bad_id, "devops")
            assert False, f"Expected ValueError for agent_id='{bad_id}'"
        except ValueError:
            pass  # expected


# ---------------------------------------------------------------------------
# test_list_agents
# ---------------------------------------------------------------------------


def test_list_agents(tmp_path):
    """Create two agents; list_agents includes both (schema may seed defaults)."""
    db = _fresh_db(tmp_path)
    mgr = __import__("agent.agent_manager", fromlist=["AgentManager"]).AgentManager(db=db)

    mgr.create_agent("agent-a", "devops")
    mgr.create_agent("agent-b", "fullstack-dev")

    agents = mgr.list_agents()
    ids = {a["id"] for a in agents}
    # The schema seeds default agents, so we check for a superset.
    assert {"agent-a", "agent-b"}.issubset(ids)


# ---------------------------------------------------------------------------
# test_list_agents_by_status
# ---------------------------------------------------------------------------


def test_list_agents_by_status(tmp_path):
    """Deactivate one agent; list_agents(status='active') includes only the other."""
    db = _fresh_db(tmp_path)
    mgr = __import__("agent.agent_manager", fromlist=["AgentManager"]).AgentManager(db=db)

    mgr.create_agent("active-one", "devops")
    mgr.create_agent("active-two", "fullstack-dev")
    mgr.deactivate_agent("active-two")

    active_ids = {a["id"] for a in mgr.list_agents(status="active")}
    assert "active-one" in active_ids
    assert "active-two" not in active_ids

    inactive_ids = {a["id"] for a in mgr.list_agents(status="inactive")}
    assert "active-two" in inactive_ids
    assert "active-one" not in inactive_ids


# ---------------------------------------------------------------------------
# test_update_agent_role
# ---------------------------------------------------------------------------


def test_update_agent_role(tmp_path):
    """Update an agent's role; the change is persisted."""
    db = _fresh_db(tmp_path)
    mgr = __import__("agent.agent_manager", fromlist=["AgentManager"]).AgentManager(db=db)

    mgr.create_agent("up-agent", "devops")
    updated = mgr.update_agent("up-agent", role="fullstack-dev")

    assert updated["role"] == "fullstack-dev"

    # Verify via fresh get
    assert mgr.get_agent("up-agent")["role"] == "fullstack-dev"


# ---------------------------------------------------------------------------
# test_deactivate_agent
# ---------------------------------------------------------------------------


def test_deactivate_agent(tmp_path):
    """Deactivating an agent sets status to 'inactive'."""
    db = _fresh_db(tmp_path)
    mgr = __import__("agent.agent_manager", fromlist=["AgentManager"]).AgentManager(db=db)

    mgr.create_agent("deac-agent", "devops")
    result = mgr.deactivate_agent("deac-agent")

    assert result["status"] == "inactive"

    # Also confirm via get_agent
    assert mgr.get_agent("deac-agent")["status"] == "inactive"


# ---------------------------------------------------------------------------
# test_get_agent_teams
# ---------------------------------------------------------------------------


def test_get_agent_teams(tmp_path):
    """get_agent_teams returns active memberships after manual team setup."""
    db = _fresh_db(tmp_path)
    mgr = __import__("agent.agent_manager", fromlist=["AgentManager"]).AgentManager(db=db)

    # Create agent
    mgr.create_agent("team-agent", "devops")

    # Manually insert team + membership rows (team_manager is a separate module)
    now = 1700000000.0

    def _seed(conn):
        conn.execute(
            "INSERT INTO teams (id, name, created_at) VALUES (?, ?, ?)",
            ("team-alpha", "Alpha Squad", now),
        )
        conn.execute(
            "INSERT INTO teams (id, name, created_at) VALUES (?, ?, ?)",
            ("team-beta", "Beta Crew", now),
        )
        conn.execute(
            "INSERT INTO team_members (team_id, agent_id, joined_at, status) "
            "VALUES (?, ?, ?, ?)",
            ("team-alpha", "team-agent", now, "active"),
        )
        conn.execute(
            "INSERT INTO team_members (team_id, agent_id, joined_at, status) "
            "VALUES (?, ?, ?, ?)",
            ("team-beta", "team-agent", now, "inactive"),
        )

    db._execute_write(_seed)

    teams = mgr.get_agent_teams("team-agent")
    # Only the active membership should be returned
    assert len(teams) == 1
    assert teams[0]["team_id"] == "team-alpha"
    assert teams[0]["team_name"] == "Alpha Squad"
    assert teams[0]["status"] == "active"