"""CRUD for teams and team membership in the multimodal agent platform.

Teams group agents together with an optional lead. All persistence uses the
existing SessionDB class (schema v12 tables: teams, team_members, agents).
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from hermes_state import SessionDB

logger = logging.getLogger(__name__)

_TEAM_ID_RE = re.compile(r"^[a-z0-9_-]+$")


class TeamManager:
    """CRUD for teams and team membership."""

    def __init__(self, db: Optional[SessionDB] = None):
        self.db = db or SessionDB()

    # ── Team CRUD ──

    def create_team(self, team_id: str, name: str) -> Dict[str, Any]:
        """Create a new team.

        Args:
            team_id: Unique identifier (must match ``^[a-z0-9_-]+$``).
            name: Human-readable team name.

        Returns:
            Dict with the created team row.

        Raises:
            ValueError: If team_id is invalid or already exists.
        """
        if not _TEAM_ID_RE.match(team_id):
            raise ValueError(
                f"Invalid team_id '{team_id}': must match ^[a-z0-9_-]+$"
            )

        now = time.time()

        def _do(conn):
            try:
                conn.execute(
                    "INSERT INTO teams (id, name, created_at) VALUES (?, ?, ?)",
                    (team_id, name, now),
                )
            except Exception as exc:
                err_msg = str(exc).lower()
                if "unique" in err_msg or "constraint" in err_msg:
                    raise ValueError(f"Team '{team_id}' already exists") from exc
                raise

        self.db._execute_write(_do)
        return self.get_team(team_id)  # type: ignore[return-value]

    def get_team(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Return team dict or None."""
        with self.db._lock:
            cursor = self.db._conn.execute(
                "SELECT id, name, lead_agent_id, created_at FROM teams WHERE id = ?",
                (team_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def list_teams(self) -> List[Dict[str, Any]]:
        """List all teams."""
        with self.db._lock:
            cursor = self.db._conn.execute(
                "SELECT id, name, lead_agent_id, created_at FROM teams ORDER BY created_at"
            )
            rows = cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Lead management ──

    def set_lead(self, team_id: str, agent_id: str) -> Dict[str, Any]:
        """Set team lead. Returns updated team dict.

        Raises:
            ValueError: If team_id doesn't exist or agent_id is not a member.
        """
        team = self.get_team(team_id)
        if team is None:
            raise ValueError(f"Team '{team_id}' does not exist")

        # Verify agent is a member of the team
        with self.db._lock:
            cursor = self.db._conn.execute(
                "SELECT 1 FROM team_members WHERE team_id = ? AND agent_id = ?",
                (team_id, agent_id),
            )
            if cursor.fetchone() is None:
                raise ValueError(
                    f"Agent '{agent_id}' is not a member of team '{team_id}'"
                )

        def _do(conn):
            conn.execute(
                "UPDATE teams SET lead_agent_id = ? WHERE id = ?",
                (agent_id, team_id),
            )

        self.db._execute_write(_do)
        return self.get_team(team_id)  # type: ignore[return-value]

    # ── Membership ──

    def add_member(self, team_id: str, agent_id: str) -> Dict[str, Any]:
        """Add agent to team. Idempotent if already a member.

        Raises:
            ValueError: If team doesn't exist or agent doesn't exist in agents table.
        """
        team = self.get_team(team_id)
        if team is None:
            raise ValueError(f"Team '{team_id}' does not exist")

        # Verify agent exists in agents table
        with self.db._lock:
            cursor = self.db._conn.execute(
                "SELECT 1 FROM agents WHERE id = ?",
                (agent_id,),
            )
            if cursor.fetchone() is None:
                raise ValueError(f"Agent '{agent_id}' does not exist")

        now = time.time()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO team_members (team_id, agent_id, joined_at, status)
                VALUES (?, ?, ?, 'active')
                ON CONFLICT(team_id, agent_id) DO UPDATE SET
                    status = 'active',
                    joined_at = excluded.joined_at
                """,
                (team_id, agent_id, now),
            )

        self.db._execute_write(_do)

        with self.db._lock:
            cursor = self.db._conn.execute(
                "SELECT team_id, agent_id, joined_at, status FROM team_members "
                "WHERE team_id = ? AND agent_id = ?",
                (team_id, agent_id),
            )
            row = cursor.fetchone()
        return dict(row)

    def remove_member(self, team_id: str, agent_id: str) -> bool:
        """Remove agent from team.

        Returns True if removed, False if wasn't a member.
        If the removed agent was the lead, lead_agent_id is set to None.
        """
        # Check if agent is the lead before removing
        team = self.get_team(team_id)
        if team is None:
            return False

        was_lead = team.get("lead_agent_id") == agent_id

        def _do(conn):
            cursor = conn.execute(
                "DELETE FROM team_members WHERE team_id = ? AND agent_id = ?",
                (team_id, agent_id),
            )
            removed = cursor.rowcount > 0

            if removed and was_lead:
                conn.execute(
                    "UPDATE teams SET lead_agent_id = NULL WHERE id = ?",
                    (team_id,),
                )

            return removed

        return self.db._execute_write(_do)

    def get_members(
        self, team_id: str, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return team members with agent details (role, status)."""
        if status is not None:
            query = (
                "SELECT tm.agent_id, a.role, tm.status, tm.joined_at "
                "FROM team_members tm JOIN agents a ON tm.agent_id = a.id "
                "WHERE tm.team_id = ? AND tm.status = ?"
            )
            params: tuple = (team_id, status)
        else:
            query = (
                "SELECT tm.agent_id, a.role, tm.status, tm.joined_at "
                "FROM team_members tm JOIN agents a ON tm.agent_id = a.id "
                "WHERE tm.team_id = ?"
            )
            params = (team_id,)

        with self.db._lock:
            cursor = self.db._conn.execute(query, params)
            rows = cursor.fetchall()
        return [dict(r) for r in rows]

    def get_members_excluding_lead(self, team_id: str) -> List[Dict[str, Any]]:
        """Return active members of *team_id* minus the team lead.

        Used by ``consult_team(include_lead=False)`` to fan out to
        non-lead members only.

        Returns:
            List of dicts with keys: agent_id, role, status, joined_at.
            Empty list if the team does not exist or has no non-lead members.
        """
        with self.db._lock:
            rows = self.db._conn.execute(
                """
                SELECT tm.agent_id, a.role, tm.status, tm.joined_at
                FROM team_members tm
                JOIN agents a ON tm.agent_id = a.id
                WHERE tm.team_id = ?
                  AND tm.status = 'active'
                  AND tm.agent_id != COALESCE(
                      (SELECT lead_agent_id FROM teams WHERE id = ?), ''
                  )
                ORDER BY tm.joined_at
                """,
                (team_id, team_id),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Composite info ──

    def get_team_info(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Full team info: team dict + members list + lead agent info."""
        team = self.get_team(team_id)
        if team is None:
            return None

        members = self.get_members(team_id)

        lead_info: Optional[Dict[str, Any]] = None
        lead_agent_id = team.get("lead_agent_id")
        if lead_agent_id is not None:
            with self.db._lock:
                cursor = self.db._conn.execute(
                    "SELECT id, role, status, created_at FROM agents WHERE id = ?",
                    (lead_agent_id,),
                )
                row = cursor.fetchone()
            if row is not None:
                lead_info = dict(row)

        return {
            "team": team,
            "members": members,
            "lead": lead_info,
        }


# ── Singleton ──

_team_manager: Optional[TeamManager] = None


def get_team_manager(db: Optional[SessionDB] = None) -> TeamManager:
    """Return the process-global TeamManager singleton."""
    global _team_manager
    if _team_manager is None:
        _team_manager = TeamManager(db)
    return _team_manager