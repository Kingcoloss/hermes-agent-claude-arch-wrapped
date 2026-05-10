"""CRUD for persistent agent identities.

Uses the existing SessionDB class from hermes_state.py for persistence.
The ``agents`` table is defined in hermes_state.py schema version 12.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from hermes_state import SessionDB

logger = logging.getLogger(__name__)

#: Valid agent IDs: lowercase alphanumeric, hyphens, underscores.
_AGENT_ID_RE = re.compile(r"^[a-z0-9_-]+$")

#: Valid agent statuses.
_VALID_STATUSES = ("active", "inactive")


class AgentManager:
    """CRUD for persistent agent identities.

    Each agent has an ``id``, ``role``, ``status`` (active/inactive),
    and ``created_at`` timestamp.  Team membership is read from the
    ``team_members`` / ``teams`` tables.
    """

    def __init__(self, db: Optional[SessionDB] = None):
        """Initialize with a SessionDB instance.

        Args:
            db: SessionDB instance. If None, a new one is created at the
                default path.
        """
        self.db = db or SessionDB()

    @staticmethod
    def _validate_agent_id(agent_id: str) -> None:
        """Raise ValueError if *agent_id* is not a valid identifier."""
        if not _AGENT_ID_RE.match(agent_id):
            raise ValueError(
                f"Invalid agent_id '{agent_id}': must match [a-z0-9_-]+"
            )

    # ── Create ──

    def create_agent(self, agent_id: str, role: str) -> Dict[str, Any]:
        """Create a new agent identity.

        Args:
            agent_id: Unique identifier (must match ``^[a-z0-9_-]+$``).
            role: The role to assign to this agent.

        Returns:
            Dict with keys: id, role, status, created_at.

        Raises:
            ValueError: If *agent_id* is invalid or already exists.
        """
        self._validate_agent_id(agent_id)
        now = time.time()

        def _do(conn):
            try:
                conn.execute(
                    """
                    INSERT INTO agents (id, role, status, created_at)
                    VALUES (?, ?, 'active', ?)
                    """,
                    (agent_id, role, now),
                )
            except Exception as exc:
                err_msg = str(exc).lower()
                if "unique" in err_msg or "constraint" in err_msg:
                    raise ValueError(
                        f"Agent '{agent_id}' already exists"
                    ) from exc
                raise

        self.db._execute_write(_do)

        return self.get_agent(agent_id)  # type: ignore[return-value]

    # ── Read ──

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Return agent dict or None if not found.

        Args:
            agent_id: The agent identifier.

        Returns:
            Dict with keys: id, role, status, created_at, or None.
        """
        with self.db._lock:
            cursor = self.db._conn.execute(
                "SELECT id, role, status, created_at FROM agents WHERE id = ?",
                (agent_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "role": row["role"],
            "status": row["status"],
            "created_at": row["created_at"],
        }

    def list_agents(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all agents, optionally filtered by status.

        Args:
            status: If provided, only return agents with this status.

        Returns:
            List of dicts with keys: id, role, status, created_at.
        """
        if status is not None:
            with self.db._lock:
                cursor = self.db._conn.execute(
                    "SELECT id, role, status, created_at FROM agents WHERE status = ?",
                    (status,),
                )
                rows = cursor.fetchall()
        else:
            with self.db._lock:
                cursor = self.db._conn.execute(
                    "SELECT id, role, status, created_at FROM agents ORDER BY created_at",
                    (),
                )
                rows = cursor.fetchall()

        return [
            {
                "id": row["id"],
                "role": row["role"],
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # ── Update ──

    def update_agent(self, agent_id: str, **fields) -> Dict[str, Any]:
        """Update agent fields.

        Args:
            agent_id: The agent identifier.
            **fields: Keyword arguments. Only ``role`` and ``status`` are
                supported. ``status`` must be one of ('active', 'inactive').

        Returns:
            Updated agent dict.

        Raises:
            ValueError: If *agent_id* not found, or *status* is invalid,
                or unsupported fields are provided.
        """
        allowed = {"role", "status"}
        unsupported = set(fields.keys()) - allowed
        if unsupported:
            raise ValueError(f"Unsupported fields: {unsupported}")

        if "status" in fields and fields["status"] not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{fields['status']}': "
                f"must be one of {_VALID_STATUSES}"
            )

        if not self.get_agent(agent_id):
            raise ValueError(f"Agent '{agent_id}' not found")

        set_clauses = []
        params: list = []
        for key in ("role", "status"):
            if key in fields:
                set_clauses.append(f"{key} = ?")
                params.append(fields[key])

        if not set_clauses:
            # Nothing to update — return current state.
            return self.get_agent(agent_id)  # type: ignore[return-value]

        params.append(agent_id)

        def _do(conn):
            conn.execute(
                f"UPDATE agents SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )

        self.db._execute_write(_do)

        return self.get_agent(agent_id)  # type: ignore[return-value]

    def deactivate_agent(self, agent_id: str) -> Dict[str, Any]:
        """Set an agent's status to 'inactive'.

        Args:
            agent_id: The agent identifier.

        Returns:
            Updated agent dict.

        Raises:
            ValueError: If *agent_id* not found.
        """
        return self.update_agent(agent_id, status="inactive")

    # ── Team membership ──

    def get_agent_teams(self, agent_id: str) -> List[Dict[str, Any]]:
        """Return all active team memberships for this agent.

        Joins ``team_members`` with ``teams`` to include team names.

        Args:
            agent_id: The agent identifier.

        Returns:
            List of dicts with keys: team_id, team_name, joined_at, status.
        """
        with self.db._lock:
            cursor = self.db._conn.execute(
                """
                SELECT tm.team_id, t.name AS team_name,
                       tm.joined_at, tm.status
                FROM team_members tm
                JOIN teams t ON tm.team_id = t.id
                WHERE tm.agent_id = ? AND tm.status = 'active'
                ORDER BY tm.joined_at
                """,
                (agent_id,),
            )
            rows = cursor.fetchall()

        return [
            {
                "team_id": row["team_id"],
                "team_name": row["team_name"],
                "joined_at": row["joined_at"],
                "status": row["status"],
            }
            for row in rows
        ]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_agent_manager: Optional[AgentManager] = None


def get_agent_manager(db: Optional[SessionDB] = None) -> AgentManager:
    """Return the process-global AgentManager singleton."""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager(db)
    return _agent_manager