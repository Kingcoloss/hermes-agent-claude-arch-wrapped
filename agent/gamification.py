"""KPI tracking and XP/Level engine for role-based multimodal agent platform.

Uses the existing SessionDB class from hermes_state.py for persistence.
Tables (agent_kpi, agent_skills_xp, agent_achievements) are defined in
hermes_state.py schema version 12. Per-agent rows added in v12.
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from hermes_state import SessionDB

logger = logging.getLogger(__name__)

#: Base XP required per level. Configurable at the instance level.
DEFAULT_XP_PER_LEVEL = 100


class KPITracker:
    """Tracks key performance indicators and role-based XP/levels.

    Metrics tracked:
    - task_success_rate
    - avg_tokens_per_task
    - tool_diversity_score
    - error_recovery_rate
    - role_proficiency_score
    """

    def __init__(self, db: Optional[SessionDB] = None, xp_per_level: int = DEFAULT_XP_PER_LEVEL):
        """Initialize the tracker with a SessionDB instance.

        Args:
            db: SessionDB instance. If None, a new one is created at the default path.
            xp_per_level: XP threshold for each level. Defaults to 100.
        """
        self.db = db or SessionDB()
        self.xp_per_level = xp_per_level

    # ── KPI recording ──

    def record_session_metrics(
        self,
        session_id: str,
        role: Optional[str],
        metrics_dict: Dict[str, float],
        agent_id: Optional[str] = None,
    ) -> None:
        """Record per-session metrics into the agent_kpi table.

        Args:
            session_id: The session identifier.
            role: The agent role for these metrics.
            metrics_dict: Mapping of metric name to numeric value.
            agent_id: Optional persistent agent identity (v12+).
        """
        timestamp = time.time()

        def _do(conn):
            for metric_name, metric_value in metrics_dict.items():
                conn.execute(
                    """
                    INSERT INTO agent_kpi (session_id, role, metric_name, metric_value, timestamp, agent_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, role, metric_name, float(metric_value), timestamp, agent_id),
                )
            logger.debug(
                "Recorded %d KPIs for session %s (role=%s, agent=%s)",
                len(metrics_dict),
                session_id,
                role,
                agent_id,
            )

        self.db._execute_write(_do)

    def get_kpi_summary(
        self,
        role: Optional[str] = None,
        days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Aggregate KPI metrics from the agent_kpi table.

        Args:
            role: Filter by role. If None, aggregates across all roles.
            days: Only include records from the last N days. If None, all time.

        Returns:
            Dict with average values for each tracked metric, plus record count.
        """
        conditions = []
        params: List[Any] = []

        if role is not None:
            conditions.append("role = ?")
            params.append(role)
        if days is not None:
            cutoff = time.time() - (days * 86400)
            conditions.append("timestamp >= ?")
            params.append(cutoff)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT metric_name, AVG(metric_value) as avg_value, COUNT(*) as cnt
            FROM agent_kpi
            {where_clause}
            GROUP BY metric_name
        """

        with self.db._lock:
            cursor = self.db._conn.execute(query, params)
            rows = cursor.fetchall()

        summary: Dict[str, Any] = {
            "record_count": 0,
            "task_success_rate": None,
            "avg_tokens_per_task": None,
            "tool_diversity_score": None,
            "error_recovery_rate": None,
            "role_proficiency_score": None,
        }

        total_records = 0
        for row in rows:
            metric_name = row["metric_name"]
            avg_value = row["avg_value"]
            cnt = row["cnt"]
            if metric_name in summary:
                summary[metric_name] = avg_value
            total_records += cnt

        summary["record_count"] = total_records
        return summary

    # ── XP / Level system ──

    def add_xp(self, skill_name: str, xp_delta: float, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Add XP to a skill/role and update its level.

        Args:
            skill_name: The skill or role name.
            xp_delta: Amount of XP to add (can be negative for penalties).
            agent_id: Optional persistent agent identity (v12+). If set, also
                writes a per-agent row with skill_name = "<agent_id>/<skill_name>".

        Returns:
            Dict with keys: skill_name, old_xp, new_xp, old_level, new_level,
            xp_to_next, leveled_up.
        """
        now = time.time()

        # Fetch current state
        with self.db._lock:
            cursor = self.db._conn.execute(
                "SELECT xp, level FROM agent_skills_xp WHERE skill_name = ?",
                (skill_name,),
            )
            row = cursor.fetchone()

        old_xp = row["xp"] if row else 0.0
        old_level = row["level"] if row else 1

        new_xp = max(0.0, old_xp + xp_delta)
        new_level = math.floor(new_xp / self.xp_per_level) + 1
        xp_to_next = (new_level * self.xp_per_level) - new_xp

        def _do(conn):
            conn.execute(
                """
                INSERT INTO agent_skills_xp (skill_name, xp, level, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(skill_name) DO UPDATE SET
                    xp = excluded.xp,
                    level = excluded.level,
                    last_updated = excluded.last_updated
                """,
                (skill_name, new_xp, new_level, now),
            )
            if agent_id:
                agent_skill = f"{agent_id}/{skill_name}"
                conn.execute(
                    """
                    INSERT INTO agent_skills_xp (skill_name, xp, level, last_updated, agent_id)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(skill_name) DO UPDATE SET
                        xp = excluded.xp,
                        level = excluded.level,
                        last_updated = excluded.last_updated
                    """,
                    (agent_skill, new_xp, new_level, now, agent_id),
                )

        self.db._execute_write(_do)

        leveled_up = new_level > old_level
        if leveled_up:
            logger.info(
                "Level up! %s went from level %d to level %d",
                skill_name,
                old_level,
                new_level,
            )

        return {
            "skill_name": skill_name,
            "old_xp": old_xp,
            "new_xp": new_xp,
            "old_level": old_level,
            "new_level": new_level,
            "xp_to_next": xp_to_next,
            "leveled_up": leveled_up,
        }

    def get_level(self, skill_name: str) -> Dict[str, Any]:
        """Return the current level, XP, and XP needed for next level.

        Args:
            skill_name: The skill or role name.

        Returns:
            Dict with keys: skill_name, level, xp, xp_to_next.
        """
        with self.db._lock:
            cursor = self.db._conn.execute(
                "SELECT xp, level FROM agent_skills_xp "
                "WHERE skill_name = ? AND agent_id IS NULL",
                (skill_name,),
            )
            row = cursor.fetchone()

        if not row:
            return {
                "skill_name": skill_name,
                "level": 1,
                "xp": 0.0,
                "xp_to_next": float(self.xp_per_level),
            }

        xp = row["xp"]
        level = math.floor(xp / self.xp_per_level) + 1
        xp_to_next = (level * self.xp_per_level) - xp

        return {
            "skill_name": skill_name,
            "level": level,
            "xp": xp,
            "xp_to_next": xp_to_next,
        }

    # ── Achievements ──

    def unlock_achievement(
        self,
        achievement_id: str,
        name: str,
        description: Optional[str] = None,
        role: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> bool:
        """Unlock an achievement.

        Args:
            achievement_id: Unique identifier for the achievement.
            name: Human-readable name.
            description: Optional description.
            role: Optional role association.
            agent_id: Optional persistent agent identity (v12+).

        Returns:
            True if newly unlocked, False if already existed.
        """
        now = time.time()

        def _do(conn):
            try:
                conn.execute(
                    """
                    INSERT INTO agent_achievements (achievement_id, name, description, role, unlocked_at, agent_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (achievement_id, name, description, role, now, agent_id),
                )
                logger.info("Unlocked achievement '%s' (%s)", name, achievement_id)
                return True
            except Exception as exc:
                # SQLite UNIQUE constraint violation or similar
                err_msg = str(exc).lower()
                if "unique" in err_msg or "constraint" in err_msg:
                    logger.debug(
                        "Achievement '%s' already unlocked, skipping",
                        achievement_id,
                    )
                    return False
                raise

        return self.db._execute_write(_do)

    def get_achievements(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """List unlocked achievements, optionally filtered by role.

        Args:
            role: Filter by role. If None, returns all achievements.

        Returns:
            List of achievement dicts with keys: id, achievement_id, name,
            description, role, unlocked_at.
        """
        if role is not None:
            query = """
                SELECT id, achievement_id, name, description, role, unlocked_at
                FROM agent_achievements
                WHERE role = ?
                ORDER BY unlocked_at DESC
            """
            params = (role,)
        else:
            query = """
                SELECT id, achievement_id, name, description, role, unlocked_at
                FROM agent_achievements
                ORDER BY unlocked_at DESC
            """
            params = ()

        with self.db._lock:
            cursor = self.db._conn.execute(query, params)
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_leaderboard(self, role: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the top skills/roles by XP (global rows only, agent_id IS NULL).

        Args:
            role: Filter by skill/role name. If None, returns all global skills.
            limit: Maximum number of entries to return.

        Returns:
            List of dicts with keys: rank, skill_name, level, xp.
        """
        if role is not None:
            query = """
                SELECT skill_name, level, xp FROM agent_skills_xp
                WHERE skill_name = ? AND agent_id IS NULL
                ORDER BY xp DESC
                LIMIT ?
            """
            params = (role, limit)
        else:
            query = """
                SELECT skill_name, level, xp FROM agent_skills_xp
                WHERE agent_id IS NULL
                ORDER BY xp DESC
                LIMIT ?
            """
            params = (limit,)

        with self.db._lock:
            cursor = self.db._conn.execute(query, params)
            rows = cursor.fetchall()

        results = [dict(row) for row in rows]
        for i, entry in enumerate(results, start=1):
            entry["rank"] = i
        return results

    def get_agent_leaderboard(self, agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the top skills for a specific agent by XP.

        Args:
            agent_id: The agent whose skill rows to query.
            limit: Maximum number of entries to return.

        Returns:
            List of dicts with keys: rank, skill_name, level, xp.
            The ``<agent_id>/`` prefix is stripped from skill_name.
        """
        prefix = f"{agent_id}/"
        query = """
            SELECT skill_name, level, xp FROM agent_skills_xp
            WHERE agent_id = ?
            ORDER BY xp DESC
            LIMIT ?
        """
        with self.db._lock:
            cursor = self.db._conn.execute(query, (agent_id, limit))
            rows = cursor.fetchall()

        results = []
        for i, row in enumerate(rows, start=1):
            entry = dict(row)
            # Strip the "<agent_id>/" prefix so callers see clean skill names.
            if entry["skill_name"].startswith(prefix):
                entry["skill_name"] = entry["skill_name"][len(prefix):]
            entry["rank"] = i
            results.append(entry)
        return results

