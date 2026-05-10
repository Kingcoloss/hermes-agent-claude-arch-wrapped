"""CRUD and guard logic for agent-to-agent consultations.

Consultations capture one-shot peer questions between agents. Each
consultation row records the question, context_summary, response, status
(pending | done | failed), and an optional parent that forms a chain
tree.

Hierarchy guards enforced on ``create_consultation``:
- **Depth limit** — depth may not exceed ``max_depth`` (default 2).
- **Cycle detection** — the same ``agent_id`` may not appear twice in a
  parent chain.
- **Cost cap** — at most ``cost_cap`` consultations per session (default 5).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from hermes_state import SessionDB

logger = logging.getLogger(__name__)


class ConsultationManager:
    """CRUD, guard, and chain-tree logic for consultations."""

    DEFAULT_MAX_DEPTH = 2
    DEFAULT_COST_CAP = 5  # max consults per session

    def __init__(
        self,
        db: Optional[SessionDB] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        cost_cap: int = DEFAULT_COST_CAP,
    ):
        self.db = db or SessionDB()
        self.max_depth = max_depth
        self.cost_cap = cost_cap

    # ── Create ──

    def create_consultation(
        self,
        caller_session_id: str,
        target_agent_id: str,
        question: str,
        context_summary: str,
        caller_agent_id: Optional[str] = None,
        parent_consultation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new consultation row with all hierarchy guards.

        Args:
            caller_session_id: Session that initiated the consultation.
            target_agent_id: Agent being consulted.
            question: The question text.
            context_summary: LLM-summarised context; never raw history.
            caller_agent_id: Calling agent id, or None when the CEO calls.
            parent_consultation_id: Parent consultation id for chained calls.

        Returns:
            Dict with the new consultation row.

        Raises:
            ValueError: On target-not-found, inactive target, depth exceeded,
                        cycle detected, or cost cap exceeded.
        """
        # 1 — Validate target agent exists and is active
        with self.db._lock:
            row = self.db._conn.execute(
                "SELECT id, status FROM agents WHERE id = ?",
                (target_agent_id,),
            ).fetchone()

        if row is None:
            raise ValueError(f"Target agent '{target_agent_id}' not found")

        target_status = row["status"] if hasattr(row, "keys") else row[1]
        if target_status != "active":
            raise ValueError(
                f"Target agent '{target_agent_id}' is not active (status='{target_status}')"
            )

        # 2 — Compute depth and validate depth limit
        depth = self._compute_depth(parent_consultation_id)
        if depth > self.max_depth:
            raise ValueError(
                f"Consultation depth {depth} exceeds max_depth={self.max_depth}"
            )

        # 3 — Cycle detection: target_agent_id must not appear in the parent chain
        self._check_cycle(parent_consultation_id, target_agent_id)

        # 4 — Per-session cost cap
        session_count = self._count_session_consultations(caller_session_id)
        if session_count >= self.cost_cap:
            raise ValueError(
                f"Session cost cap reached: {session_count} consultations "
                f"(cap={self.cost_cap})"
            )

        # 5 — Insert row
        consultation_id = str(uuid.uuid4())
        now = time.time()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO consultations (
                    id, caller_session_id, caller_agent_id, target_agent_id,
                    question, context_summary, status,
                    parent_consultation_id, depth, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    consultation_id,
                    caller_session_id,
                    caller_agent_id,
                    target_agent_id,
                    question,
                    context_summary,
                    parent_consultation_id,
                    depth,
                    now,
                ),
            )

        self.db._execute_write(_do)
        return self.get_consultation(consultation_id)  # type: ignore[return-value]

    # ── Complete / fail ──

    def complete_consultation(
        self,
        consultation_id: str,
        response: str,
        cost_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mark a consultation done, record response and optional token cost.

        Returns:
            Updated consultation dict.

        Raises:
            ValueError: If consultation_id not found.
        """
        if not self.get_consultation(consultation_id):
            raise ValueError(f"Consultation '{consultation_id}' not found")

        now = time.time()

        def _do(conn):
            conn.execute(
                """
                UPDATE consultations
                SET status = 'done', response = ?, completed_at = ?,
                    cost_tokens = COALESCE(?, cost_tokens)
                WHERE id = ?
                """,
                (response, now, cost_tokens, consultation_id),
            )

        self.db._execute_write(_do)
        return self.get_consultation(consultation_id)  # type: ignore[return-value]

    def fail_consultation(
        self, consultation_id: str, error: str
    ) -> Dict[str, Any]:
        """Mark a consultation failed, storing the error as the response.

        Returns:
            Updated consultation dict.
        """
        now = time.time()

        def _do(conn):
            conn.execute(
                """
                UPDATE consultations
                SET status = 'failed', response = ?, completed_at = ?
                WHERE id = ?
                """,
                (error, now, consultation_id),
            )

        self.db._execute_write(_do)
        result = self.get_consultation(consultation_id)
        if result is None:
            raise ValueError(f"Consultation '{consultation_id}' not found")
        return result

    # ── Read ──

    def get_consultation(self, consultation_id: str) -> Optional[Dict[str, Any]]:
        """Return consultation dict or None."""
        with self.db._lock:
            row = self.db._conn.execute(
                """
                SELECT id, caller_session_id, caller_agent_id, target_agent_id,
                       question, context_summary, response, status,
                       parent_consultation_id, depth, cost_tokens,
                       created_at, completed_at
                FROM consultations WHERE id = ?
                """,
                (consultation_id,),
            ).fetchone()

        if row is None:
            return None
        return dict(row)

    def list_consultations_for_session(
        self, caller_session_id: str
    ) -> List[Dict[str, Any]]:
        """Return all consultations for a session, ordered by creation time."""
        with self.db._lock:
            rows = self.db._conn.execute(
                """
                SELECT id, caller_session_id, caller_agent_id, target_agent_id,
                       question, context_summary, response, status,
                       parent_consultation_id, depth, cost_tokens,
                       created_at, completed_at
                FROM consultations
                WHERE caller_session_id = ?
                ORDER BY created_at
                """,
                (caller_session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_consultations_for_agent(
        self,
        agent_id: str,
        *,
        role: str = "any",
    ) -> List[Dict[str, Any]]:
        """Return consultations involving an agent.

        Args:
            agent_id: Agent identifier.
            role: ``"caller"`` — consultations where agent is the caller;
                  ``"target"`` — consultations where agent is the target;
                  ``"any"`` (default) — either side.

        Returns:
            List of consultation dicts ordered by creation time.
        """
        if role == "caller":
            sql = (
                "SELECT * FROM consultations WHERE caller_agent_id = ? ORDER BY created_at"
            )
            params: tuple = (agent_id,)
        elif role == "target":
            sql = (
                "SELECT * FROM consultations WHERE target_agent_id = ? ORDER BY created_at"
            )
            params = (agent_id,)
        else:
            sql = (
                "SELECT * FROM consultations "
                "WHERE caller_agent_id = ? OR target_agent_id = ? "
                "ORDER BY created_at"
            )
            params = (agent_id, agent_id)

        with self.db._lock:
            rows = self.db._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_chain_tree(self, root_consultation_id: str) -> Dict[str, Any]:
        """Return the chain rooted at *root_consultation_id* as a nested dict.

        Format::

            {
                "consultation": {...},
                "children": [
                    {"consultation": {...}, "children": [...]},
                    ...
                ]
            }

        Returns an empty dict if *root_consultation_id* is not found.
        """
        root = self.get_consultation(root_consultation_id)
        if root is None:
            return {}
        return self._build_tree(root)

    def _build_tree(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively build the chain tree from *node* downward."""
        with self.db._lock:
            rows = self.db._conn.execute(
                "SELECT * FROM consultations WHERE parent_consultation_id = ? ORDER BY created_at",
                (node["id"],),
            ).fetchall()
        children = [self._build_tree(dict(r)) for r in rows]
        return {"consultation": node, "children": children}

    # ── Private helpers ──

    def _check_cycle(
        self, parent_id: Optional[str], target_agent_id: str
    ) -> None:
        """Walk the parent chain and raise ValueError if target_agent_id appears.

        Args:
            parent_id: The parent consultation id to start walking from.
            target_agent_id: The agent we are about to add to the chain.

        Raises:
            ValueError: If *target_agent_id* already appears somewhere in the
                parent chain.
        """
        current_id = parent_id
        seen_agents: List[str] = []

        while current_id is not None:
            with self.db._lock:
                row = self.db._conn.execute(
                    "SELECT id, caller_agent_id, target_agent_id, parent_consultation_id "
                    "FROM consultations WHERE id = ?",
                    (current_id,),
                ).fetchone()
            if row is None:
                break
            row_dict = dict(row)
            for agent_field in ("caller_agent_id", "target_agent_id"):
                a = row_dict.get(agent_field)
                if a:
                    seen_agents.append(a)
            current_id = row_dict.get("parent_consultation_id")

        if target_agent_id in seen_agents:
            raise ValueError(
                f"Cycle detected: agent '{target_agent_id}' already appears in the "
                "consultation chain"
            )

    def _compute_depth(self, parent_id: Optional[str]) -> int:
        """Return parent.depth + 1, or 0 if parent_id is None."""
        if parent_id is None:
            return 0
        with self.db._lock:
            row = self.db._conn.execute(
                "SELECT depth FROM consultations WHERE id = ?",
                (parent_id,),
            ).fetchone()
        if row is None:
            return 0
        parent_depth = row["depth"] if hasattr(row, "keys") else row[0]
        return parent_depth + 1

    def _count_session_consultations(self, caller_session_id: str) -> int:
        """Return the number of consultations already created in *caller_session_id*."""
        with self.db._lock:
            row = self.db._conn.execute(
                "SELECT COUNT(*) FROM consultations WHERE caller_session_id = ?",
                (caller_session_id,),
            ).fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_consultation_manager: Optional[ConsultationManager] = None


def get_consultation_manager(
    db: Optional[SessionDB] = None,
) -> ConsultationManager:
    """Return the process-global ConsultationManager singleton."""
    global _consultation_manager
    if _consultation_manager is None:
        _consultation_manager = ConsultationManager(db)
    return _consultation_manager
