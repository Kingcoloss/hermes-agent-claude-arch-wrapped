#!/usr/bin/env python3
"""Integration tests for role-based management and gamification system.

These tests verify the end-to-end integration of RoleManager and KPITracker
with the actual database and filesystem, using the _isolate_hermes_home fixture
to ensure hermetic test execution.
"""

import math
import time
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from agent.gamification import KPITracker, DEFAULT_XP_PER_LEVEL
from agent.role_manager import (
    RoleManager,
    RoleProfile,
    DEFAULT_ROLES,
    get_role_manager,
)
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from model_tools import get_tool_definitions
from toolsets import resolve_multiple_toolsets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fresh_db():
    """Create a SessionDB with an explicit per-test path.

    DEFAULT_DB_PATH is evaluated at module-import time, so relying on it
    would cause tests to share a single database.  We always pass an
    explicit db_path derived from the current HERMES_HOME.
    """
    return SessionDB(db_path=get_hermes_home() / "state.db")


def _create_session(db: SessionDB, session_id: str, role: str = None):
    """Insert a minimal session row so FK constraints on agent_kpi are satisfied."""
    now = time.time()

    def _do(conn):
        conn.execute(
            """
            INSERT INTO sessions (id, title, started_at, source, role)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, f"Test {session_id}", now, "integration_test", role),
        )

    db._execute_write(_do)


def _write_role_yaml(name: str, data: dict) -> Path:
    """Write a role YAML file into the isolated HERMES_HOME/roles directory."""
    roles_dir = get_hermes_home() / "roles"
    roles_dir.mkdir(parents=True, exist_ok=True)
    path = roles_dir / f"{name}.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_role_manager():
    """Return a fresh RoleManager instance with no cached state."""
    mgr = RoleManager()
    mgr.reload()
    return mgr


@pytest.fixture
def temp_db():
    """Provide a fresh SessionDB instance pointing at the isolated HERMES_HOME."""
    return _make_fresh_db()


@pytest.fixture
def kpi_tracker(temp_db):
    """Provide a KPITracker backed by the isolated temp_db."""
    return KPITracker(db=temp_db)


# ---------------------------------------------------------------------------
# 1. RoleManager lists all 6 default roles correctly
# ---------------------------------------------------------------------------

def test_role_manager_lists_six_default_roles(fresh_role_manager):
    roles = fresh_role_manager.list_roles()
    expected = sorted(DEFAULT_ROLES.keys())
    assert len(roles) == 6
    assert roles == expected
    for name in expected:
        assert name in roles


# ---------------------------------------------------------------------------
# 2. RoleManager loads custom roles from ~/.hermes/roles/*.yaml
# ---------------------------------------------------------------------------

def test_role_manager_loads_custom_roles():
    # Write the custom role file BEFORE instantiating RoleManager
    _write_role_yaml("test-role", {
        "name": "test-role",
        "description": "A temporary test role for integration testing.",
        "toolsets": ["web", "terminal"],
        "default_model": "test-model",
        "skin": "test-skin",
        "kpi_weights": {
            "task_success_rate": 2.0,
            "avg_tokens_per_task": 0.5,
        },
        "system_prompt_extra": "You are a test role. Be very testy.",
    })

    mgr = RoleManager()
    mgr.reload()
    roles = mgr.list_roles()
    assert "test-role" in roles
    role = mgr.get_role("test-role")
    assert role is not None
    assert role.name == "test-role"
    assert role.description == "A temporary test role for integration testing."
    assert role.toolsets == ["web", "terminal"]
    assert role.default_model == "test-model"
    assert role.skin == "test-skin"


def test_role_manager_user_roles_override_defaults():
    """User-defined roles with the same name as defaults should override them."""
    _write_role_yaml("devops", {
        "name": "devops",
        "description": "Overridden devops role.",
        "toolsets": ["web"],
        "kpi_weights": {"task_success_rate": 99.0},
        "system_prompt_extra": "Overridden.",
    })

    mgr = RoleManager()
    mgr.reload()
    role = mgr.get_role("devops")
    assert role.description == "Overridden devops role."
    assert role.toolsets == ["web"]
    assert role.kpi_weights["task_success_rate"] == 99.0


# ---------------------------------------------------------------------------
# 3. RoleManager.get_role() returns correct profile with toolsets and KPI weights
# ---------------------------------------------------------------------------

def test_get_role_returns_correct_profile(fresh_role_manager):
    role = fresh_role_manager.get_role("fullstack-dev")
    assert role is not None
    assert role.name == "fullstack-dev"
    assert "frontend, backend, database, api, testing, and deployment." in role.description.lower()
    assert role.toolsets == ["fullstack-dev"]
    assert "task_success_rate" in role.kpi_weights
    assert role.kpi_weights["task_success_rate"] == 1.2
    assert role.kpi_weights["role_proficiency_score"] == 1.2


def test_get_role_returns_none_for_unknown_role(fresh_role_manager):
    assert fresh_role_manager.get_role("nonexistent-role-12345") is None


def test_get_role_case_sensitive(fresh_role_manager):
    assert fresh_role_manager.get_role("FullStack-Dev") is None
    assert fresh_role_manager.get_role("fullstack-dev") is not None


# ---------------------------------------------------------------------------
# 4. RoleManager.build_role_system_prompt() injects role context correctly
# ---------------------------------------------------------------------------

def test_build_role_system_prompt_includes_role_context(fresh_role_manager):
    prompt = fresh_role_manager.build_role_system_prompt("quant-trader")
    assert "# Role: quant-trader" in prompt
    assert "quantitative trader" in prompt.lower()
    assert "statistical rigor" in prompt
    assert "Available tools" in prompt


def test_build_role_system_prompt_empty_for_missing_role(fresh_role_manager):
    prompt = fresh_role_manager.build_role_system_prompt("missing-role")
    assert prompt == ""


def test_build_role_system_prompt_empty_for_role_without_extra(fresh_role_manager):
    """A role with no system_prompt_extra should still produce a basic prompt."""
    _write_role_yaml("minimal-role", {
        "name": "minimal-role",
        "description": "Minimal role.",
        "toolsets": ["web"],
    })
    fresh_role_manager.reload()

    prompt = fresh_role_manager.build_role_system_prompt("minimal-role")
    assert "# Role: minimal-role" in prompt
    assert "Minimal role." in prompt


# ---------------------------------------------------------------------------
# 5. KPITracker records KPI metrics per session
# ---------------------------------------------------------------------------

def test_kpi_tracker_records_session_metrics(kpi_tracker, temp_db):
    session_id = str(uuid4())
    _create_session(temp_db, session_id, role="devops")

    metrics = {
        "task_success_rate": 0.95,
        "avg_tokens_per_task": 150.0,
        "tool_diversity_score": 3.5,
    }
    kpi_tracker.record_session_metrics(session_id, "devops", metrics)

    with temp_db._lock:
        cursor = temp_db._conn.execute(
            "SELECT COUNT(*) as cnt FROM agent_kpi WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
    assert row["cnt"] == 3


def test_kpi_tracker_records_multiple_sessions(kpi_tracker, temp_db):
    sid_a = str(uuid4())
    sid_b = str(uuid4())
    _create_session(temp_db, sid_a, role="devops")
    _create_session(temp_db, sid_b, role="quant-trader")

    kpi_tracker.record_session_metrics(sid_a, "devops", {"task_success_rate": 0.8})
    kpi_tracker.record_session_metrics(sid_b, "quant-trader", {"task_success_rate": 0.9})

    summary_all = kpi_tracker.get_kpi_summary()
    assert summary_all["record_count"] == 2

    summary_devops = kpi_tracker.get_kpi_summary(role="devops")
    assert summary_devops["record_count"] == 1
    assert summary_devops["task_success_rate"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 6. KPITracker.get_kpi_summary() returns correct aggregated metrics
# ---------------------------------------------------------------------------

def test_kpi_summary_aggregation(kpi_tracker, temp_db):
    sid1 = str(uuid4())
    sid2 = str(uuid4())
    _create_session(temp_db, sid1, role="content-creator")
    _create_session(temp_db, sid2, role="content-creator")

    kpi_tracker.record_session_metrics(
        sid1, "content-creator", {"task_success_rate": 0.8, "tool_diversity_score": 2.0}
    )
    kpi_tracker.record_session_metrics(
        sid2, "content-creator", {"task_success_rate": 1.0, "tool_diversity_score": 4.0}
    )

    summary = kpi_tracker.get_kpi_summary(role="content-creator")
    assert summary["record_count"] == 4  # 2 sessions * 2 metrics each
    assert summary["task_success_rate"] == pytest.approx(0.9)
    assert summary["tool_diversity_score"] == pytest.approx(3.0)


def test_kpi_summary_days_filter(kpi_tracker, temp_db):
    sid_old = str(uuid4())
    sid_new = str(uuid4())
    _create_session(temp_db, sid_old, role="devops")
    _create_session(temp_db, sid_new, role="devops")

    kpi_tracker.record_session_metrics(sid_old, "devops", {"task_success_rate": 0.5})
    # Update its timestamp to be 10 days old
    old_time = time.time() - (10 * 86400)
    with temp_db._lock:
        temp_db._conn.execute(
            "UPDATE agent_kpi SET timestamp = ? WHERE session_id = ?",
            (old_time, sid_old),
        )
        temp_db._conn.commit()

    kpi_tracker.record_session_metrics(sid_new, "devops", {"task_success_rate": 1.0})

    # Filter last 5 days — only the new session should count
    summary = kpi_tracker.get_kpi_summary(role="devops", days=5)
    assert summary["record_count"] == 1
    assert summary["task_success_rate"] == pytest.approx(1.0)


def test_kpi_summary_returns_none_for_missing_metrics(kpi_tracker):
    summary = kpi_tracker.get_kpi_summary(role="propfirm-trader")
    assert summary["record_count"] == 0
    assert summary["task_success_rate"] is None
    assert summary["avg_tokens_per_task"] is None


# ---------------------------------------------------------------------------
# 7. KPITracker XP/level calculation (level = floor(xp / 100) + 1)
# ---------------------------------------------------------------------------

def test_xp_level_calculation_formula():
    """Verify the level formula directly: level = floor(xp / 100) + 1."""
    assert math.floor(0 / 100) + 1 == 1
    assert math.floor(50 / 100) + 1 == 1
    assert math.floor(99 / 100) + 1 == 1
    assert math.floor(100 / 100) + 1 == 2
    assert math.floor(150 / 100) + 1 == 2
    assert math.floor(199 / 100) + 1 == 2
    assert math.floor(200 / 100) + 1 == 3
    assert math.floor(999 / 100) + 1 == 10


def test_kpi_tracker_uses_default_xp_per_level(kpi_tracker):
    assert kpi_tracker.xp_per_level == DEFAULT_XP_PER_LEVEL
    assert DEFAULT_XP_PER_LEVEL == 100


# ---------------------------------------------------------------------------
# 8. KPITracker.add_xp() and get_level() work correctly
# ---------------------------------------------------------------------------

def test_add_xp_increases_level(kpi_tracker):
    result = kpi_tracker.add_xp("fullstack-dev", 250)
    assert result["skill_name"] == "fullstack-dev"
    assert result["old_xp"] == 0.0
    assert result["new_xp"] == 250.0
    assert result["old_level"] == 1
    assert result["new_level"] == 3  # floor(250 / 100) + 1
    assert result["leveled_up"] is True
    assert result["xp_to_next"] == pytest.approx(50.0)  # 300 - 250


def test_add_xp_no_level_up(kpi_tracker):
    result = kpi_tracker.add_xp("devops", 50)
    assert result["new_level"] == 1
    assert result["leveled_up"] is False


def test_add_xp_negative_penalty(kpi_tracker):
    kpi_tracker.add_xp("quant-trader", 150)
    result = kpi_tracker.add_xp("quant-trader", -60)
    assert result["new_xp"] == 90.0
    assert result["new_level"] == 1  # floor(90 / 100) + 1
    assert result["leveled_up"] is False


def test_add_xp_never_below_zero(kpi_tracker):
    result = kpi_tracker.add_xp("content-creator", -50)
    assert result["new_xp"] == 0.0
    assert result["new_level"] == 1


def test_get_level_for_new_skill(kpi_tracker):
    level_info = kpi_tracker.get_level("system-engineer")
    assert level_info["skill_name"] == "system-engineer"
    assert level_info["level"] == 1
    assert level_info["xp"] == 0.0
    assert level_info["xp_to_next"] == 100.0


def test_get_level_after_adding_xp(kpi_tracker):
    kpi_tracker.add_xp("propfirm-trader", 350)
    level_info = kpi_tracker.get_level("propfirm-trader")
    assert level_info["level"] == 4  # floor(350 / 100) + 1
    assert level_info["xp"] == 350.0
    assert level_info["xp_to_next"] == 50.0  # 400 - 350


def test_add_xp_persists_to_db(kpi_tracker, temp_db):
    kpi_tracker.add_xp("devops", 275)
    with temp_db._lock:
        cursor = temp_db._conn.execute(
            "SELECT xp, level FROM agent_skills_xp WHERE skill_name = ?",
            ("devops",),
        )
        row = cursor.fetchone()
    assert row["xp"] == 275.0
    assert row["level"] == 3


# ---------------------------------------------------------------------------
# 9. KPITracker achievements unlock correctly
# ---------------------------------------------------------------------------

def test_unlock_achievement_new(kpi_tracker):
    result = kpi_tracker.unlock_achievement(
        achievement_id="first-task",
        name="First Task",
        description="Completed your first task.",
        role="devops",
    )
    assert result is True


def test_unlock_achievement_duplicate_returns_false(kpi_tracker):
    kpi_tracker.unlock_achievement("unique-id", "Unique Achievement")
    result = kpi_tracker.unlock_achievement("unique-id", "Unique Achievement")
    assert result is False


def test_get_achievements_all(kpi_tracker):
    kpi_tracker.unlock_achievement("ach-1", "Achievement One", role="devops")
    kpi_tracker.unlock_achievement("ach-2", "Achievement Two", role="quant-trader")
    kpi_tracker.unlock_achievement("ach-3", "Achievement Three", role="devops")

    all_achievements = kpi_tracker.get_achievements()
    assert len(all_achievements) == 3
    ids = {a["achievement_id"] for a in all_achievements}
    assert ids == {"ach-1", "ach-2", "ach-3"}


def test_get_achievements_filtered_by_role(kpi_tracker):
    kpi_tracker.unlock_achievement("ach-1", "One", role="devops")
    kpi_tracker.unlock_achievement("ach-2", "Two", role="quant-trader")
    kpi_tracker.unlock_achievement("ach-3", "Three", role="devops")

    devops_achievements = kpi_tracker.get_achievements(role="devops")
    assert len(devops_achievements) == 2
    assert {a["achievement_id"] for a in devops_achievements} == {"ach-1", "ach-3"}


def test_get_achievements_ordered_by_time(kpi_tracker):
    kpi_tracker.unlock_achievement("first", "First")
    time.sleep(0.01)
    kpi_tracker.unlock_achievement("second", "Second")

    achievements = kpi_tracker.get_achievements()
    assert achievements[0]["achievement_id"] == "second"
    assert achievements[1]["achievement_id"] == "first"


def test_achievement_persists_to_db(kpi_tracker, temp_db):
    kpi_tracker.unlock_achievement("db-test", "DB Test", description="Persisted.")
    with temp_db._lock:
        cursor = temp_db._conn.execute(
            "SELECT name, description FROM agent_achievements WHERE achievement_id = ?",
            ("db-test",),
        )
        row = cursor.fetchone()
    assert row["name"] == "DB Test"
    assert row["description"] == "Persisted."


# ---------------------------------------------------------------------------
# 10. Role switch updates active role and filters toolsets
# ---------------------------------------------------------------------------

def test_role_switch_changes_toolsets(fresh_role_manager):
    """Switching from one role to another yields different resolved toolsets."""
    devops = fresh_role_manager.get_role("devops")
    quant = fresh_role_manager.get_role("quant-trader")

    assert devops is not None
    assert quant is not None

    devops_tools = devops.resolved_tools
    quant_tools = quant.resolved_tools

    # They should be different sets of tools
    assert set(devops_tools) != set(quant_tools)

    # Quant-trader has quant-specific tools that devops lacks
    assert "quant_black_scholes" in quant_tools
    assert "quant_black_scholes" not in devops_tools

    # Devops has cronjob and home-assistant tools
    assert "cronjob" in devops_tools


def test_role_switch_simulates_agent_toolset_filtering(fresh_role_manager):
    """Simulate the AIAgent pattern: role -> toolsets -> get_tool_definitions."""
    devops = fresh_role_manager.get_role("devops")
    creator = fresh_role_manager.get_role("content-creator")

    # Verify that the raw toolsets are different
    assert devops.toolsets != creator.toolsets

    # Verify resolved tool lists differ
    devops_resolved = set(resolve_multiple_toolsets(devops.toolsets))
    creator_resolved = set(resolve_multiple_toolsets(creator.toolsets))
    assert devops_resolved != creator_resolved

    # Verify that get_tool_definitions also returns different sets
    devops_defs = get_tool_definitions(
        enabled_toolsets=devops.toolsets, quiet_mode=True
    )
    creator_defs = get_tool_definitions(
        enabled_toolsets=creator.toolsets, quiet_mode=True
    )
    devops_tool_names = {d["function"]["name"] for d in devops_defs}
    creator_tool_names = {d["function"]["name"] for d in creator_defs}

    # Devops has terminal and process tools; content creator does not
    assert "terminal" in devops_tool_names
    assert "process" in devops_tool_names
    assert "terminal" not in creator_tool_names
    assert "process" not in creator_tool_names

    # Content creator has text_to_speech; devops does not
    assert "text_to_speech" in creator_tool_names
    assert "text_to_speech" not in devops_tool_names

    # Overall sizes should differ
    assert len(devops_tool_names) != len(creator_tool_names)


def test_role_profile_resolved_tools_match_toolset_resolution(fresh_role_manager):
    """Verify that RoleProfile.resolved_tools matches the underlying toolsets module."""
    for role_name in fresh_role_manager.list_roles():
        role = fresh_role_manager.get_role(role_name)
        expected = resolve_multiple_toolsets(role.toolsets)
        assert role.resolved_tools == expected


def test_role_switch_updates_kpi_weights(fresh_role_manager):
    """Different roles carry different KPI weightings."""
    devops = fresh_role_manager.get_role("devops")
    quant = fresh_role_manager.get_role("quant-trader")

    assert devops.kpi_weights["error_recovery_rate"] == 1.5
    assert quant.kpi_weights["error_recovery_rate"] == 1.0

    assert quant.kpi_weights["role_proficiency_score"] == 1.5
    assert devops.kpi_weights["role_proficiency_score"] == 1.0


# ---------------------------------------------------------------------------
# Leaderboard tests (bonus coverage for integration completeness)
# ---------------------------------------------------------------------------

def test_leaderboard_ranking(kpi_tracker):
    kpi_tracker.add_xp("alpha", 500)
    kpi_tracker.add_xp("beta", 300)
    kpi_tracker.add_xp("gamma", 700)

    board = kpi_tracker.get_leaderboard()
    assert len(board) == 3
    assert board[0]["skill_name"] == "gamma"
    assert board[0]["rank"] == 1
    assert board[1]["skill_name"] == "alpha"
    assert board[2]["skill_name"] == "beta"


def test_leaderboard_limit(kpi_tracker):
    for i in range(15):
        kpi_tracker.add_xp(f"skill-{i}", i * 10)

    board = kpi_tracker.get_leaderboard(limit=5)
    assert len(board) == 5


def test_leaderboard_filtered_by_role(kpi_tracker):
    kpi_tracker.add_xp("devops", 200)
    kpi_tracker.add_xp("quant-trader", 400)

    devops_board = kpi_tracker.get_leaderboard(role="devops")
    assert len(devops_board) == 1
    assert devops_board[0]["skill_name"] == "devops"


# ---------------------------------------------------------------------------
# End-to-end: Role + Gamification integration
# ---------------------------------------------------------------------------

def test_end_to_end_role_kpi_xp_achievements(fresh_role_manager, kpi_tracker, temp_db):
    """Full workflow: use a role, record KPIs, gain XP, unlock achievement."""
    role_name = "fullstack-dev"
    session_id = str(uuid4())

    # 1. Verify role exists and has expected configuration
    role = fresh_role_manager.get_role(role_name)
    assert role is not None
    assert "frontend, backend, database, api, testing, and deployment." in role.description.lower()
    assert role.kpi_weights["task_success_rate"] == 1.2

    # 2. Record session KPIs for the role
    _create_session(temp_db, session_id, role=role_name)
    kpi_tracker.record_session_metrics(
        session_id,
        role_name,
        {
            "task_success_rate": 0.92,
            "avg_tokens_per_task": 200.0,
            "tool_diversity_score": 4.5,
            "error_recovery_rate": 0.88,
            "role_proficiency_score": 0.95,
        },
    )

    # 3. Verify KPI summary reflects the data
    summary = kpi_tracker.get_kpi_summary(role=role_name)
    assert summary["record_count"] == 5
    assert summary["task_success_rate"] == pytest.approx(0.92)

    # 4. Add XP for the role
    xp_result = kpi_tracker.add_xp(role_name, 220)
    assert xp_result["new_level"] == 3
    assert xp_result["leveled_up"] is True

    # 5. Unlock an achievement
    unlocked = kpi_tracker.unlock_achievement(
        achievement_id="e2e-first-deploy",
        name="First Deploy",
        description="Deployed your first application.",
        role=role_name,
    )
    assert unlocked is True

    # 6. Verify achievement appears in listings
    achievements = kpi_tracker.get_achievements(role=role_name)
    assert len(achievements) == 1
    assert achievements[0]["achievement_id"] == "e2e-first-deploy"

    # 7. Verify leaderboard shows the role
    board = kpi_tracker.get_leaderboard(role=role_name)
    assert len(board) == 1
    assert board[0]["skill_name"] == role_name
    assert board[0]["level"] == 3
    assert board[0]["xp"] == 220.0
