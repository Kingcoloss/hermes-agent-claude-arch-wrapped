"""Integration tests for M1.5 Peer Consultation & Hierarchy.

Exercises ConsultationManager + consult_tool + CLI handlers end-to-end.
External LLM calls (claude_subagent) are mocked.
"""
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

import pytest

from agent.agent_manager import AgentManager
from agent.team_manager import TeamManager
from agent.consultation_manager import ConsultationManager
from hermes_state import SessionDB


def _make_db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Test 1 — CEO consults alice who consults the rest of the team
# ---------------------------------------------------------------------------


class TestCeoConsultsAliceWhoConsultsTeam:
    def test_ceo_consults_alice_who_consults_team(self, tmp_path):
        db = _make_db(tmp_path)
        am = AgentManager(db=db)
        tm = TeamManager(db=db)
        mgr = ConsultationManager(db=db, max_depth=2, cost_cap=10)

        # Create agents
        am.create_agent("alice", "fullstack-dev")
        am.create_agent("bob", "devops")
        am.create_agent("charlie", "quant-trader")

        # Create team and set lead
        tm.create_team("web-team", "Web Team")
        tm.add_member("web-team", "alice")
        tm.add_member("web-team", "bob")
        tm.add_member("web-team", "charlie")
        tm.set_lead("web-team", "alice")

        # CEO → alice (depth 0)
        c1 = mgr.create_consultation(
            caller_session_id="s1",
            target_agent_id="alice",
            question="estimate hermes-v2",
            context_summary="ctx",
            caller_agent_id=None,
        )
        assert c1["depth"] == 0
        mgr.complete_consultation(c1["id"], "needs 2 weeks")

        # alice → bob (depth 1), child of c1
        c2 = mgr.create_consultation(
            caller_session_id="s1",
            target_agent_id="bob",
            question="your view?",
            context_summary="ctx",
            caller_agent_id="alice",
            parent_consultation_id=c1["id"],
        )
        assert c2["depth"] == 1
        mgr.complete_consultation(c2["id"], "agree with alice")

        # alice → charlie (depth 1), sibling of c2
        c3 = mgr.create_consultation(
            caller_session_id="s1",
            target_agent_id="charlie",
            question="your view?",
            context_summary="ctx",
            caller_agent_id="alice",
            parent_consultation_id=c1["id"],
        )
        assert c3["depth"] == 1
        mgr.complete_consultation(c3["id"], "also agree")

        # All 3 exist
        assert mgr.get_consultation(c1["id"]) is not None
        assert mgr.get_consultation(c2["id"]) is not None
        assert mgr.get_consultation(c3["id"]) is not None

        # Chain tree structure
        tree = mgr.get_chain_tree(c1["id"])
        assert tree["consultation"]["id"] == c1["id"]
        children = tree["children"]
        assert len(children) == 2

        child_ids = {n["consultation"]["id"] for n in children}
        assert child_ids == {c2["id"], c3["id"]}

        # Depths
        for child_node in children:
            assert child_node["consultation"]["depth"] == 1
            assert child_node["children"] == []


# ---------------------------------------------------------------------------
# Test 2 — Depth-3 is blocked
# ---------------------------------------------------------------------------


class TestDepth3Blocked:
    def test_depth_3_blocked(self, tmp_path):
        db = _make_db(tmp_path)
        am = AgentManager(db=db)
        mgr = ConsultationManager(db=db, max_depth=2, cost_cap=10)

        for aid in ("a1", "a2", "a3", "a4"):
            am.create_agent(aid, "fullstack-dev")

        # Root → a1 (depth 0)
        c0 = mgr.create_consultation(
            caller_session_id="s-depth",
            target_agent_id="a1",
            question="Q0",
            context_summary="ctx",
        )
        assert c0["depth"] == 0
        mgr.complete_consultation(c0["id"], "resp-a1")

        # a1 → a2 (depth 1)
        c1 = mgr.create_consultation(
            caller_session_id="s-depth",
            target_agent_id="a2",
            question="Q1",
            context_summary="ctx",
            caller_agent_id="a1",
            parent_consultation_id=c0["id"],
        )
        assert c1["depth"] == 1
        mgr.complete_consultation(c1["id"], "resp-a2")

        # a2 → a3 (depth 2 — still allowed)
        c2 = mgr.create_consultation(
            caller_session_id="s-depth",
            target_agent_id="a3",
            question="Q2",
            context_summary="ctx",
            caller_agent_id="a2",
            parent_consultation_id=c1["id"],
        )
        assert c2["depth"] == 2
        mgr.complete_consultation(c2["id"], "resp-a3")

        # a3 → a4 (depth 3 — should be blocked)
        with pytest.raises(ValueError, match="depth"):
            mgr.create_consultation(
                caller_session_id="s-depth",
                target_agent_id="a4",
                question="Q3",
                context_summary="ctx",
                caller_agent_id="a3",
                parent_consultation_id=c2["id"],
            )


# ---------------------------------------------------------------------------
# Test 3 — Cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_cycle_detection(self, tmp_path):
        db = _make_db(tmp_path)
        am = AgentManager(db=db)
        mgr = ConsultationManager(db=db, max_depth=2, cost_cap=10)

        am.create_agent("alice", "fullstack-dev")
        am.create_agent("bob", "devops")

        # alice → bob (depth 0, caller=alice)
        c_ab = mgr.create_consultation(
            caller_session_id="s-cycle",
            target_agent_id="bob",
            question="Q?",
            context_summary="ctx",
            caller_agent_id="alice",
        )
        mgr.complete_consultation(c_ab["id"], "resp-bob")

        # bob → alice (cycle: alice is in the chain as caller_agent_id)
        with pytest.raises(ValueError, match="[Cc]ycle"):
            mgr.create_consultation(
                caller_session_id="s-cycle",
                target_agent_id="alice",
                question="Q?",
                context_summary="ctx",
                caller_agent_id="bob",
                parent_consultation_id=c_ab["id"],
            )


# ---------------------------------------------------------------------------
# Test 4 — Session cost cap
# ---------------------------------------------------------------------------


class TestSessionCostCap:
    def test_session_cost_cap(self, tmp_path):
        db = _make_db(tmp_path)
        am = AgentManager(db=db)
        mgr = ConsultationManager(db=db, max_depth=2, cost_cap=5)

        # Create 6 distinct target agents
        agents = ["alice", "bob", "charlie", "dave", "eve", "frank"]
        for aid in agents:
            am.create_agent(aid, "fullstack-dev")

        # Create 5 consultations in session-X (should all succeed)
        for aid in agents[:5]:
            c = mgr.create_consultation(
                caller_session_id="session-X",
                target_agent_id=aid,
                question="Q",
                context_summary="ctx",
            )
            mgr.complete_consultation(c["id"], "resp")

        # 6th consultation should raise with "cap" in the message
        with pytest.raises(ValueError, match="[Cc]ap"):
            mgr.create_consultation(
                caller_session_id="session-X",
                target_agent_id="frank",
                question="Q6",
                context_summary="ctx",
            )


# ---------------------------------------------------------------------------
# Test 5 — Journal entries for consult via CLI handler
# ---------------------------------------------------------------------------


class TestJournalEntriesForConsultViaCLIHandler:
    def test_journal_entries_for_consult_via_cli_handler(self, tmp_path):
        import agent.consultation_manager as cm_mod
        import agent.agent_manager as am_mod

        db = _make_db(tmp_path)
        am = AgentManager(db=db)
        am.create_agent("alice", "fullstack-dev")

        # Inject singletons so cli handler and consult_tool use the fixture DB
        cm_mod._consultation_manager = ConsultationManager(db=db, max_depth=2, cost_cap=5)
        am_mod._agent_manager = am

        try:
            from cli import HermesCLI

            cli = HermesCLI.__new__(HermesCLI)
            cli.session_id = "test-session"
            cli._app = None

            # Mock _invoke_agent to avoid real subprocess calls
            with patch("tools.consult_tool._invoke_agent", return_value="stub response"):
                with patch("cli._cprint"):
                    cli._handle_consult_command("/consult alice 'how are you'")

            # Verify journal was written for alice
            from agent.agent_vault import read_journal
            journal_content = read_journal("alice")
            assert "consulted by CEO" in journal_content

        finally:
            # Cleanup singletons
            cm_mod._consultation_manager = None
            am_mod._agent_manager = None


# ---------------------------------------------------------------------------
# Test 6 — Consult-log tree renders correctly
# ---------------------------------------------------------------------------


class TestConsultLogTreeRenders:
    def test_consult_log_tree_renders(self, tmp_path):
        db = _make_db(tmp_path)
        am = AgentManager(db=db)
        mgr = ConsultationManager(db=db, max_depth=2, cost_cap=10)

        for aid in ("root-agent", "mid-agent", "leaf-agent"):
            am.create_agent(aid, "fullstack-dev")

        # root → mid-agent (depth 0)
        root = mgr.create_consultation(
            caller_session_id="s-tree",
            target_agent_id="root-agent",
            question="Root Q",
            context_summary="ctx",
        )
        mgr.complete_consultation(root["id"], "root resp")

        # mid-agent at depth 1
        child = mgr.create_consultation(
            caller_session_id="s-tree",
            target_agent_id="mid-agent",
            question="Child Q",
            context_summary="ctx",
            caller_agent_id="root-agent",
            parent_consultation_id=root["id"],
        )
        mgr.complete_consultation(child["id"], "child resp")

        # leaf-agent at depth 2
        grandchild = mgr.create_consultation(
            caller_session_id="s-tree",
            target_agent_id="leaf-agent",
            question="Grandchild Q",
            context_summary="ctx",
            caller_agent_id="mid-agent",
            parent_consultation_id=child["id"],
        )
        mgr.complete_consultation(grandchild["id"], "grandchild resp")

        tree = mgr.get_chain_tree(root["id"])

        # Root structure
        assert "consultation" in tree
        assert "children" in tree
        assert tree["consultation"]["id"] == root["id"]
        assert len(tree["children"]) == 1

        # Depth-1 child
        child_node = tree["children"][0]
        assert "consultation" in child_node
        assert "children" in child_node
        assert child_node["consultation"]["target_agent_id"] == "mid-agent"
        assert len(child_node["children"]) == 1

        # Depth-2 grandchild
        grandchild_node = child_node["children"][0]
        assert "consultation" in grandchild_node
        assert "children" in grandchild_node
        assert grandchild_node["consultation"]["target_agent_id"] == "leaf-agent"
        assert grandchild_node["children"] == []


# ---------------------------------------------------------------------------
# Test 7 — Cross-team consultation
# ---------------------------------------------------------------------------


class TestCrossTeamConsult:
    def test_cross_team_consult(self, tmp_path):
        db = _make_db(tmp_path)
        am = AgentManager(db=db)
        tm = TeamManager(db=db)
        mgr = ConsultationManager(db=db, max_depth=2, cost_cap=10)

        am.create_agent("alice", "fullstack-dev")
        am.create_agent("bob", "devops")

        tm.create_team("web-team", "Web Team")
        tm.add_member("web-team", "alice")
        tm.set_lead("web-team", "alice")

        tm.create_team("mobile-team", "Mobile Team")
        tm.add_member("mobile-team", "bob")
        tm.set_lead("mobile-team", "bob")

        # alice consults bob (root-level, cross-team)
        c = mgr.create_consultation(
            caller_session_id="s-cross",
            caller_agent_id="alice",
            target_agent_id="bob",
            question="Can we share the auth service?",
            context_summary="ctx",
        )
        assert c["status"] == "pending"

        # Complete it
        updated = mgr.complete_consultation(c["id"], "Yes, happy to share")
        assert updated["status"] == "done"

        # Appears in alice's caller list
        alice_callers = mgr.list_consultations_for_agent("alice", role="caller")
        assert any(r["id"] == c["id"] for r in alice_callers)

        # Appears in bob's target list
        bob_targets = mgr.list_consultations_for_agent("bob", role="target")
        assert any(r["id"] == c["id"] for r in bob_targets)
