"""Tests for agent/consultation_manager.py."""

import time
import pytest

from hermes_state import SessionDB
from agent.consultation_manager import ConsultationManager


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_consult.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


@pytest.fixture()
def manager(db):
    return ConsultationManager(db=db, max_depth=2, cost_cap=5)


def _create_agent(db: SessionDB, agent_id: str, status: str = "active") -> None:
    """Insert an agent row directly for testing."""
    now = time.time()
    db._execute_write(
        lambda conn: conn.execute(
            "INSERT OR IGNORE INTO agents (id, role, status, created_at) VALUES (?, ?, ?, ?)",
            (agent_id, "fullstack-dev", status, now),
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateConsultation:
    def test_happy_path_ceo_to_alice(self, manager, db):
        _create_agent(db, "alice")
        c = manager.create_consultation(
            caller_session_id="sess-1",
            target_agent_id="alice",
            question="What is the ETA for v2?",
            context_summary="We are at 60% completion.",
        )
        assert c["id"] is not None
        assert c["target_agent_id"] == "alice"
        assert c["caller_agent_id"] is None  # CEO call
        assert c["depth"] == 0
        assert c["status"] == "pending"
        assert c["parent_consultation_id"] is None

    def test_depth_limit_blocks_depth_3(self, manager, db):
        """Chain alice → bob → charlie → would-be = depth 3, should fail."""
        for aid in ("alice", "bob", "charlie", "dave"):
            _create_agent(db, aid)

        c0 = manager.create_consultation(
            caller_session_id="sess-2",
            target_agent_id="alice",
            question="Q0",
            context_summary="ctx",
        )  # depth 0
        manager.complete_consultation(c0["id"], "resp-alice")

        c1 = manager.create_consultation(
            caller_session_id="sess-2",
            target_agent_id="bob",
            question="Q1",
            context_summary="ctx",
            caller_agent_id="alice",
            parent_consultation_id=c0["id"],
        )  # depth 1
        manager.complete_consultation(c1["id"], "resp-bob")

        c2 = manager.create_consultation(
            caller_session_id="sess-2",
            target_agent_id="charlie",
            question="Q2",
            context_summary="ctx",
            caller_agent_id="bob",
            parent_consultation_id=c1["id"],
        )  # depth 2 — allowed
        manager.complete_consultation(c2["id"], "resp-charlie")

        # Depth 3 should be blocked
        with pytest.raises(ValueError, match="depth"):
            manager.create_consultation(
                caller_session_id="sess-2",
                target_agent_id="dave",
                question="Q3",
                context_summary="ctx",
                caller_agent_id="charlie",
                parent_consultation_id=c2["id"],
            )

    def test_cycle_detection_alice_bob_alice(self, manager, db):
        """alice → bob → alice cycle should be blocked."""
        for aid in ("alice", "bob"):
            _create_agent(db, aid)

        c0 = manager.create_consultation(
            caller_session_id="sess-3",
            target_agent_id="alice",
            question="Q?",
            context_summary="ctx",
        )
        manager.complete_consultation(c0["id"], "resp-alice")

        c1 = manager.create_consultation(
            caller_session_id="sess-3",
            target_agent_id="bob",
            question="Q?",
            context_summary="ctx",
            caller_agent_id="alice",
            parent_consultation_id=c0["id"],
        )
        manager.complete_consultation(c1["id"], "resp-bob")

        # alice is already in the chain — bob cannot consult alice again
        with pytest.raises(ValueError, match="[Cc]ycle"):
            manager.create_consultation(
                caller_session_id="sess-3",
                target_agent_id="alice",
                question="Q?",
                context_summary="ctx",
                caller_agent_id="bob",
                parent_consultation_id=c1["id"],
            )

    def test_cost_cap_sixth_consultation_fails(self, manager, db):
        """6th consultation in the same session should raise (cap=5)."""
        for i in range(6):
            _create_agent(db, f"agent-cap-{i}")

        for i in range(5):
            c = manager.create_consultation(
                caller_session_id="sess-cap",
                target_agent_id=f"agent-cap-{i}",
                question="Q",
                context_summary="ctx",
            )
            manager.complete_consultation(c["id"], "resp")

        with pytest.raises(ValueError, match="[Cc]ap"):
            manager.create_consultation(
                caller_session_id="sess-cap",
                target_agent_id="agent-cap-5",
                question="Q6",
                context_summary="ctx",
            )

    def test_target_agent_not_found_raises(self, manager, db):
        with pytest.raises(ValueError, match="not found"):
            manager.create_consultation(
                caller_session_id="sess-x",
                target_agent_id="nonexistent-agent",
                question="Q",
                context_summary="ctx",
            )

    def test_inactive_target_agent_raises(self, manager, db):
        _create_agent(db, "inactive-agent", status="inactive")
        with pytest.raises(ValueError, match="not active"):
            manager.create_consultation(
                caller_session_id="sess-x",
                target_agent_id="inactive-agent",
                question="Q",
                context_summary="ctx",
            )


class TestCompleteConsultation:
    def test_complete_updates_row_correctly(self, manager, db):
        _create_agent(db, "bob")
        c = manager.create_consultation(
            caller_session_id="sess-done",
            target_agent_id="bob",
            question="Q",
            context_summary="ctx",
        )
        assert c["status"] == "pending"

        updated = manager.complete_consultation(
            c["id"], response="Looks good!", cost_tokens=42
        )
        assert updated["status"] == "done"
        assert updated["response"] == "Looks good!"
        assert updated["cost_tokens"] == 42
        assert updated["completed_at"] is not None

    def test_fail_consultation_sets_failed(self, manager, db):
        _create_agent(db, "charlie")
        c = manager.create_consultation(
            caller_session_id="sess-fail",
            target_agent_id="charlie",
            question="Q",
            context_summary="ctx",
        )
        updated = manager.fail_consultation(c["id"], error="Sub-agent timed out")
        assert updated["status"] == "failed"
        assert "timed out" in updated["response"]
        assert updated["completed_at"] is not None


class TestGetChainTree:
    def test_chain_tree_three_levels(self, manager, db):
        """get_chain_tree should return correct nested structure for a 3-level chain."""
        for aid in ("lead", "mid", "leaf"):
            _create_agent(db, aid)

        root = manager.create_consultation(
            caller_session_id="sess-tree",
            target_agent_id="lead",
            question="Root Q",
            context_summary="ctx",
        )
        manager.complete_consultation(root["id"], "root resp")

        child = manager.create_consultation(
            caller_session_id="sess-tree",
            target_agent_id="mid",
            question="Child Q",
            context_summary="ctx",
            caller_agent_id="lead",
            parent_consultation_id=root["id"],
        )
        manager.complete_consultation(child["id"], "child resp")

        grandchild = manager.create_consultation(
            caller_session_id="sess-tree",
            target_agent_id="leaf",
            question="Grandchild Q",
            context_summary="ctx",
            caller_agent_id="mid",
            parent_consultation_id=child["id"],
        )
        manager.complete_consultation(grandchild["id"], "grandchild resp")

        tree = manager.get_chain_tree(root["id"])
        assert tree["consultation"]["id"] == root["id"]
        assert len(tree["children"]) == 1

        child_node = tree["children"][0]
        assert child_node["consultation"]["id"] == child["id"]
        assert len(child_node["children"]) == 1

        grandchild_node = child_node["children"][0]
        assert grandchild_node["consultation"]["id"] == grandchild["id"]
        assert grandchild_node["children"] == []

    def test_get_chain_tree_not_found_returns_empty(self, manager, db):
        result = manager.get_chain_tree("nonexistent-id")
        assert result == {}
