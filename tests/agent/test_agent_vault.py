"""Tests for agent/agent_vault.py — per-agent journal vault on disk."""

import datetime
from pathlib import Path

import pytest

from agent.agent_vault import (
    append_journal,
    get_vault_dir,
    list_agents_with_vaults,
    read_journal,
    rebuild_index,
)
from hermes_constants import get_hermes_home

# _hermetic_environment (autouse) redirects HERMES_HOME to a temp dir.


# ---------------------------------------------------------------------------
# test_append_journal_creates_files
# ---------------------------------------------------------------------------

def test_append_journal_creates_files():
    """Append once — both journal.md and INDEX.md exist with correct content."""
    ts = datetime.datetime(2026, 4, 30, 14, 32, tzinfo=datetime.timezone.utc).timestamp()
    journal_path = append_journal(
        agent_id="alice",
        session_id="abc123",
        role="fullstack-dev",
        project_name="hermes-v2",
        kpi_summary={"tasks": 3, "ok": 3},
        xp_delta=15,
        level_before=2,
        level_after=2,
        actions=["created agents/teams", "shipped 0.1.0"],
        timestamp=ts,
    )

    assert journal_path.exists()
    index_path = journal_path.parent / "INDEX.md"
    assert index_path.exists()

    journal_text = read_journal("alice")
    # Verify entry header
    assert "## 2026-04-30 14:32 [hermes-v2] fullstack-dev" in journal_text
    # Verify session line
    assert "session: abc123 | KPI: tasks=3 ok=3 | XP: +15 (level 2)" in journal_text
    # Verify actions line
    assert "actions: created agents/teams, shipped 0.1.0" in journal_text

    # Verify INDEX
    index_text = index_path.read_text(encoding="utf-8")
    assert "# Journal Index — alice" in index_text
    assert "| 2026-04-30 14:32 | hermes-v2 | fullstack-dev | tasks=3 ok=3 | +15 (level 2) |" in index_text
    assert "Total entries: 1" in index_text


# ---------------------------------------------------------------------------
# test_append_journal_appends
# ---------------------------------------------------------------------------

def test_append_journal_appends():
    """Append twice — journal has both entries, INDEX has 2 rows."""
    ts1 = datetime.datetime(2026, 4, 30, 14, 32, tzinfo=datetime.timezone.utc).timestamp()
    append_journal(
        agent_id="bob",
        session_id="s1",
        role="quant-trader",
        project_name="alpha",
        kpi_summary={"trades": 5},
        xp_delta=20,
        level_before=1,
        level_after=2,
        actions=["ran backtest"],
        timestamp=ts1,
    )

    ts2 = datetime.datetime(2026, 4, 30, 16, 0, tzinfo=datetime.timezone.utc).timestamp()
    append_journal(
        agent_id="bob",
        session_id="s2",
        role="quant-trader",
        project_name="beta",
        kpi_summary={"trades": 8},
        xp_delta=25,
        level_before=2,
        level_after=3,
        actions=["deployed model"],
        timestamp=ts2,
    )

    journal_text = read_journal("bob")
    assert "## 2026-04-30 14:32 [alpha] quant-trader" in journal_text
    assert "## 2026-04-30 16:00 [beta] quant-trader" in journal_text

    index_path = get_vault_dir("bob") / "INDEX.md"
    index_text = index_path.read_text(encoding="utf-8")
    # Two data rows (plus header row)
    assert "| 2026-04-30 14:32 | alpha | quant-trader | trades=5 | +20 (level 1→2) |" in index_text
    assert "| 2026-04-30 16:00 | beta | quant-trader | trades=8 | +25 (level 2→3) |" in index_text
    assert "Total entries: 2" in index_text


# ---------------------------------------------------------------------------
# test_append_handles_none_fields
# ---------------------------------------------------------------------------

def test_append_handles_none_fields():
    """None for project_name, kpi_summary, xp_delta -> '—' placeholders, no crash."""
    ts = datetime.datetime(2026, 4, 30, 10, 0, tzinfo=datetime.timezone.utc).timestamp()
    journal_path = append_journal(
        agent_id="carol",
        session_id="s3",
        role="researcher",
        project_name=None,
        kpi_summary=None,
        xp_delta=None,
        level_before=None,
        level_after=None,
        actions=None,
        timestamp=ts,
    )

    journal_text = read_journal("carol")
    assert "## 2026-04-30 10:00 [—] researcher" in journal_text
    assert "KPI: —" in journal_text
    assert "XP: —" in journal_text
    assert "actions: —" in journal_text

    index_path = journal_path.parent / "INDEX.md"
    index_text = index_path.read_text(encoding="utf-8")
    assert "| — | researcher | — | — |" in index_text


# ---------------------------------------------------------------------------
# test_rebuild_index_idempotent
# ---------------------------------------------------------------------------

def test_rebuild_index_idempotent():
    """Rebuild twice — content identical."""
    ts = datetime.datetime(2026, 4, 30, 12, 0, tzinfo=datetime.timezone.utc).timestamp()
    append_journal(
        agent_id="dave",
        session_id="s4",
        role="devops",
        project_name="infra",
        kpi_summary={"deploys": 2},
        xp_delta=10,
        level_before=1,
        level_after=1,
        actions=["rolled out canary"],
        timestamp=ts,
    )

    index_path = get_vault_dir("dave") / "INDEX.md"
    first = index_path.read_text(encoding="utf-8")

    rebuild_index("dave")
    second = index_path.read_text(encoding="utf-8")

    assert first == second


# ---------------------------------------------------------------------------
# test_list_agents_with_vaults
# ---------------------------------------------------------------------------

def test_list_agents_with_vaults():
    """Create vaults for 3 agents — returns sorted list of all 3."""
    for agent_id in ("zeta", "alpha", "mid"):
        get_vault_dir(agent_id)

    result = list_agents_with_vaults()
    assert result == ["alpha", "mid", "zeta"]


# ---------------------------------------------------------------------------
# test_get_vault_dir_creates_parent
# ---------------------------------------------------------------------------

def test_get_vault_dir_creates_parent():
    """Call on new agent_id — path exists on disk."""
    vault_dir = get_vault_dir("newbie")
    assert vault_dir.exists()
    assert vault_dir.is_dir()
    # Verify it is under vaults/agents/
    assert vault_dir.parent.name == "agents"
    assert vault_dir.parent.parent.name == "vaults"


# ---------------------------------------------------------------------------
# test_path_traversal_rejected (Fix #1)
# ---------------------------------------------------------------------------

def test_get_vault_dir_rejects_path_traversal():
    """'../etc' in agent_id must raise ValueError — no directory escaping."""
    with pytest.raises(ValueError, match="Invalid agent_id"):
        get_vault_dir("../etc")


def test_get_vault_dir_rejects_absolute_path():
    """Absolute path in agent_id must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid agent_id"):
        get_vault_dir("/etc/passwd")


def test_get_vault_dir_rejects_uppercase():
    """Uppercase letters are not allowed in agent_id."""
    with pytest.raises(ValueError, match="Invalid agent_id"):
        get_vault_dir("ALICE")


def test_list_agents_with_vaults_skips_invalid_dirs():
    """Directories with names that fail _AGENT_ID_RE are silently skipped."""
    vaults_root = get_hermes_home() / "vaults" / "agents"
    vaults_root.mkdir(parents=True, exist_ok=True)

    # Create a valid dir and an invalid dir that looks like path traversal.
    (vaults_root / "valid-agent").mkdir(exist_ok=True)
    (vaults_root / "INVALID").mkdir(exist_ok=True)
    (vaults_root / "../sneaky").mkdir(exist_ok=True)

    result = list_agents_with_vaults()
    assert "valid-agent" in result
    assert "INVALID" not in result
    # The "../sneaky" dir is created as a sibling, not inside agents/,
    # so it won't even appear in the listing.  Either way, it must not appear.
    assert "../sneaky" not in result