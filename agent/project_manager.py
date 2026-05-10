"""CRUD for projects and team assignments.

Projects track work across teams. Each project has a status lifecycle
(proposed -> active -> completed/cancelled) and can have zero or more
teams assigned via the project_teams junction table.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from hermes_state import SessionDB

logger = logging.getLogger(__name__)

#: Valid project_id format: lowercase alphanumeric, hyphens, underscores.
_PROJECT_ID_RE = re.compile(r"^[a-z0-9_-]+$")

#: Allowed project statuses.
_VALID_STATUSES = ("proposed", "active", "completed", "cancelled")

#: Fields that update_project is allowed to touch.
_UPDATABLE_FIELDS = {"status", "repo_path", "target_date"}


class ProjectManager:
    """CRUD for projects and team assignments."""

    def __init__(self, db: Optional[SessionDB] = None):
        self.db = db or SessionDB()

    # ── Project CRUD ──

    def create_project(
        self,
        project_id: str,
        name: str,
        repo_path: Optional[str] = None,
    ) -> Dict[str, object]:
        """Create a new project.

        Args:
            project_id: Unique identifier (must match ``^[a-z0-9_-]+$``).
            name: Human-readable project name (must be unique).
            repo_path: Repository path. Defaults to ``os.getcwd()``.

        Returns:
            Project dict.

        Raises:
            ValueError: If project_id format is invalid or id/name is taken.
        """
        if not _PROJECT_ID_RE.match(project_id):
            raise ValueError(
                f"Invalid project_id '{project_id}': must match ^[a-z0-9_-]+$"
            )
        if repo_path is None:
            repo_path = os.getcwd()
        else:
            repo_path = str(Path(repo_path).expanduser().resolve())
        now = time.time()

        def _do(conn):
            try:
                conn.execute(
                    "INSERT INTO projects (id, name, status, repo_path, target_date, created_at) "
                    "VALUES (?, ?, 'proposed', ?, NULL, ?)",
                    (project_id, name, repo_path, now),
                )
            except Exception as exc:
                err = str(exc).lower()
                if "unique" in err:
                    # Distinguish id vs name conflict for better error messages.
                    existing = conn.execute(
                        "SELECT id FROM projects WHERE id = ?", (project_id,)
                    ).fetchone()
                    if existing:
                        raise ValueError(f"Project '{project_id}' already exists")
                    raise ValueError(f"Project name '{name}' is already taken")
                raise

        self.db._execute_write(_do)
        return self.get_project(project_id)

    def get_project(self, project_id: str) -> Optional[Dict[str, object]]:
        """Return project dict or None."""
        with self.db._lock:
            cursor = self.db._conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def list_projects(self, status: Optional[str] = None) -> List[Dict[str, object]]:
        """List projects, optionally filtered by status."""
        if status is not None:
            with self.db._lock:
                cursor = self.db._conn.execute(
                    "SELECT * FROM projects WHERE status = ? ORDER BY created_at",
                    (status,),
                )
                rows = cursor.fetchall()
        else:
            with self.db._lock:
                cursor = self.db._conn.execute(
                    "SELECT * FROM projects ORDER BY created_at"
                )
                rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def update_project(self, project_id: str, **fields) -> Dict[str, object]:
        """Update project fields.

        Only ``status``, ``repo_path``, and ``target_date`` may be updated.
        Returns the updated project dict.

        Raises:
            ValueError: If an invalid field name or status value is supplied.
        """
        for key in fields:
            if key not in _UPDATABLE_FIELDS:
                raise ValueError(f"Cannot update field '{key}'")
        if "status" in fields and fields["status"] not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{fields['status']}': must be one of {_VALID_STATUSES}"
            )
        if "repo_path" in fields and fields["repo_path"] is not None:
            fields["repo_path"] = str(Path(fields["repo_path"]).expanduser().resolve())

        set_clauses = []
        params: list = []
        for key, value in fields.items():
            set_clauses.append(f"{key} = ?")
            params.append(value)
        params.append(project_id)

        def _do(conn):
            conn.execute(
                f"UPDATE projects SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )

        self.db._execute_write(_do)
        result = self.get_project(project_id)
        return result

    # ── Team assignments ──

    def assign_team(self, project_id: str, team_id: str) -> bool:
        """Assign a team to a project. Idempotent.

        Raises:
            ValueError: If team_id does not exist in the teams table.

        Returns:
            True on success.
        """
        # Verify team exists.
        with self.db._lock:
            row = self.db._conn.execute(
                "SELECT id FROM teams WHERE id = ?", (team_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Team '{team_id}' does not exist")

        def _do(conn):
            conn.execute(
                "INSERT OR IGNORE INTO project_teams (project_id, team_id) VALUES (?, ?)",
                (project_id, team_id),
            )

        self.db._execute_write(_do)
        return True

    def unassign_team(self, project_id: str, team_id: str) -> bool:
        """Remove a team from a project.

        Returns:
            True if the assignment existed and was removed, False otherwise.
        """
        def _do(conn):
            cursor = conn.execute(
                "DELETE FROM project_teams WHERE project_id = ? AND team_id = ?",
                (project_id, team_id),
            )
            return cursor.rowcount > 0

        return self.db._execute_write(_do)

    def get_project_teams(self, project_id: str) -> List[Dict[str, object]]:
        """Return teams assigned to this project, with team names."""
        with self.db._lock:
            cursor = self.db._conn.execute(
                "SELECT pt.project_id, pt.team_id, t.name, t.lead_agent_id, t.created_at "
                "FROM project_teams pt "
                "JOIN teams t ON t.id = pt.team_id "
                "WHERE pt.project_id = ?",
                (project_id,),
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    # ── Lifecycle ──

    def complete_project(self, project_id: str) -> Dict[str, object]:
        """Set status='completed' and set completed_at.

        Raises:
            ValueError: If the project is already completed or does not exist.

        Returns:
            Updated project dict.
        """
        project = self.get_project(project_id)
        if project is None:
            raise ValueError(f"Project '{project_id}' does not exist")
        if project["status"] == "completed":
            raise ValueError(f"Project '{project_id}' is already completed")
        now = time.time()

        def _do(conn):
            conn.execute(
                "UPDATE projects SET status = 'completed', completed_at = ? WHERE id = ?",
                (now, project_id),
            )

        self.db._execute_write(_do)
        return self.get_project(project_id)

    def get_project_info(self, project_id: str) -> Optional[Dict[str, object]]:
        """Full project info: project dict + assigned teams list."""
        project = self.get_project(project_id)
        if project is None:
            return None
        teams = self.get_project_teams(project_id)
        return {**project, "teams": teams}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_project_manager: Optional[ProjectManager] = None


def get_project_manager(db: Optional[SessionDB] = None) -> ProjectManager:
    """Return the process-global ProjectManager singleton."""
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager(db)
    return _project_manager