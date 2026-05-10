"""Release lifecycle manager for project shipping.

Manages the release pipeline: draft -> testing -> shipped.
Tracks release metadata, check gates (e.g. test suites), and git tag
creation.  Persists to the ``releases`` and ``release_checks`` tables
defined in hermes_state.py schema version 12.
"""

import logging
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from hermes_state import SessionDB

logger = logging.getLogger(__name__)

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+")

#: Safe git ref characters: must start with alphanumeric and contain only safe chars.
_GIT_REF_SAFE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def _validate_git_tag(tag: str) -> None:
    """Raise ValueError if *tag* is unsafe for use as a git tag name."""
    if tag.startswith("-"):
        raise ValueError(f"Invalid git tag '{tag}': must not start with '-'")
    if tag.startswith("/"):
        raise ValueError(f"Invalid git tag '{tag}': must not start with '/'")
    if ".." in tag:
        raise ValueError(f"Invalid git tag '{tag}': must not contain '..'")
    if any(c in tag for c in (" ", "\t", "\n", "\r")):
        raise ValueError(f"Invalid git tag '{tag}': must not contain whitespace")
    if not _GIT_REF_SAFE_RE.match(tag):
        raise ValueError(
            f"Invalid git tag '{tag}': contains characters not safe for git refs"
        )


class ReleaseManager:
    """Release lifecycle: draft -> testing -> shipped.

    Each release belongs to a project and carries a semver version string.
    Check gates (test runs, lints, etc.) are recorded in release_checks.
    A release can only be shipped once all checks have passed.
    """

    def __init__(self, db: Optional[SessionDB] = None):
        self.db = db or SessionDB()

    # ── Create / Read ──

    def create_release(self, project_id: str, version: str) -> dict:
        """Create a release draft.

        Args:
            project_id: The project this release belongs to.
            version: Semver string (e.g. "1.2.3").

        Returns:
            Release dict with keys: id, project_id, version, status,
            git_tag, created_at, shipped_at.

        Raises:
            ValueError: If project_id not found, version format invalid,
                or project+version combination already exists.
        """
        # Validate project exists
        with self.db._lock:
            row = self.db._conn.execute(
                "SELECT id FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        if not row:
            raise ValueError(f"Project '{project_id}' not found")

        # Validate semver-ish format
        if not _SEMVER_RE.match(version):
            raise ValueError(
                f"Invalid version '{version}': must match X.Y.Z"
            )

        # Check for duplicate project+version
        with self.db._lock:
            existing = self.db._conn.execute(
                "SELECT id FROM releases WHERE project_id = ? AND version = ?",
                (project_id, version),
            ).fetchone()
        if existing:
            raise ValueError(
                f"Release already exists for project '{project_id}' "
                f"with version '{version}'"
            )

        release_id = uuid4().hex[:12]
        now = time.time()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO releases (id, project_id, version, status, git_tag,
                                      created_at, shipped_at)
                VALUES (?, ?, ?, 'draft', NULL, ?, NULL)
                """,
                (release_id, project_id, version, now),
            )

        self.db._execute_write(_do)
        logger.info(
            "Created release %s version %s for project %s",
            release_id, version, project_id,
        )

        return {
            "id": release_id,
            "project_id": project_id,
            "version": version,
            "status": "draft",
            "git_tag": None,
            "created_at": now,
            "shipped_at": None,
        }

    def get_release(self, release_id: str) -> Optional[dict]:
        """Return release dict or None."""
        with self.db._lock:
            row = self.db._conn.execute(
                """
                SELECT id, project_id, version, status, git_tag,
                       created_at, shipped_at
                FROM releases WHERE id = ?
                """,
                (release_id,),
            ).fetchone()
        if not row:
            return None
        return dict(row)

    def list_releases(self, project_id: Optional[str] = None) -> list[dict]:
        """List releases, optionally filtered by project."""
        if project_id is not None:
            query = """
                SELECT id, project_id, version, status, git_tag,
                       created_at, shipped_at
                FROM releases WHERE project_id = ?
                ORDER BY created_at DESC
            """
            params = (project_id,)
        else:
            query = """
                SELECT id, project_id, version, status, git_tag,
                       created_at, shipped_at
                FROM releases ORDER BY created_at DESC
            """
            params = ()

        with self.db._lock:
            cursor = self.db._conn.execute(query, params)
            rows = cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Check gates ──

    def run_check(
        self, release_id: str, check_type: str = "tests",
        command: Optional[List[str]] = None,
    ) -> dict:
        """Run a check gate and store the result.

        Args:
            release_id: The release to check.
            check_type: Type of check. Only 'tests' is supported in M1.
            command: Command to run as a list of strings. Defaults to
                ``["scripts/run_tests.sh"]`` if not provided.

        Returns:
            Check dict with keys: id, release_id, check_type, status,
            result_text, checked_at.

        Raises:
            ValueError: If release not found.
        """
        if command is None:
            command = ["scripts/run_tests.sh"]

        release = self.get_release(release_id)
        if not release:
            raise ValueError(f"Release '{release_id}' not found")

        # Get project info for repo_path
        with self.db._lock:
            proj_row = self.db._conn.execute(
                "SELECT id, name, repo_path FROM projects WHERE id = ?",
                (release["project_id"],),
            ).fetchone()
        if not proj_row:
            raise ValueError(
                f"Project '{release['project_id']}' not found"
            )

        repo_path = proj_row["repo_path"] or os.getcwd()

        # Run subprocess check
        try:
            result = subprocess.run(
                command,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            status = "passed" if result.returncode == 0 else "failed"
            combined = (result.stdout or "") + (result.stderr or "")
            result_text = combined[:500]
        except subprocess.TimeoutExpired:
            status = "failed"
            result_text = "Check timed out after 300s"
        except Exception as exc:
            status = "failed"
            result_text = str(exc)[:500]

        check_id = uuid4().hex[:12]
        now = time.time()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO release_checks
                    (id, release_id, check_type, status, result_text, checked_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (check_id, release_id, check_type, status, result_text, now),
            )
            # Transition release from draft -> testing on first check
            conn.execute(
                "UPDATE releases SET status = 'testing' "
                "WHERE id = ? AND status = 'draft'",
                (release_id,),
            )

        self.db._execute_write(_do)

        logger.info(
            "Check %s (%s) for release %s: %s",
            check_id, check_type, release_id, status,
        )

        return {
            "id": check_id,
            "release_id": release_id,
            "check_type": check_type,
            "status": status,
            "result_text": result_text,
            "checked_at": now,
        }

    def get_checks(self, release_id: str) -> list[dict]:
        """Return all checks for a release."""
        with self.db._lock:
            cursor = self.db._conn.execute(
                """
                SELECT id, release_id, check_type, status,
                       result_text, checked_at
                FROM release_checks WHERE release_id = ?
                ORDER BY checked_at
                """,
                (release_id,),
            )
            rows = cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Ship ──

    def ship_release(self, release_id: str) -> dict:
        """Ship a release if all checks pass.

        Creates a git tag in the project's repo and transitions the release
        to 'shipped' status.

        Args:
            release_id: The release to ship.

        Returns:
            Updated release dict.

        Raises:
            ValueError: If checks missing/failed, wrong status, or
                project not found.
            RuntimeError: If git tag creation fails.
        """
        release = self.get_release(release_id)
        if not release:
            raise ValueError(f"Release '{release_id}' not found")

        status = release["status"]
        if status == "draft":
            raise ValueError("Cannot ship: run checks first")
        if status == "shipped":
            raise ValueError("Cannot ship: already shipped")

        # Verify all checks passed
        checks = self.get_checks(release_id)
        if not checks:
            raise ValueError("Cannot ship: no checks have been run")

        failed = [c for c in checks if c["status"] != "passed"]
        if failed:
            failed_types = ", ".join(
                f"{c['check_type']}={c['status']}" for c in failed
            )
            raise ValueError(
                f"Cannot ship: checks not passed: {failed_types}"
            )

        # Get project info for tag name and repo path
        with self.db._lock:
            proj_row = self.db._conn.execute(
                "SELECT id, name, repo_path, status FROM projects "
                "WHERE id = ?",
                (release["project_id"],),
            ).fetchone()
        if not proj_row:
            raise ValueError(
                f"Project '{release['project_id']}' not found"
            )

        project_name = proj_row["name"]
        repo_path = proj_row["repo_path"] or os.getcwd()
        tag = f"{project_name}/v{release['version']}"
        _validate_git_tag(tag)

        # Create git tag
        tag_result = subprocess.run(
            ["git", "tag", tag],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if tag_result.returncode != 0:
            raise RuntimeError(
                f"git tag failed: {tag_result.stderr.strip()}"
            )

        now = time.time()

        def _do(conn):
            conn.execute(
                """
                UPDATE releases
                SET status = 'shipped', git_tag = ?, shipped_at = ?
                WHERE id = ?
                """,
                (tag, now, release_id),
            )
            # First ship activates the project
            conn.execute(
                "UPDATE projects SET status = 'active' WHERE id = ?",
                (release["project_id"],),
            )

        try:
            self.db._execute_write(_do)
        except Exception:
            # Tag was created but DB write failed — clean up the orphaned tag.
            cleanup = subprocess.run(
                ["git", "tag", "-d", tag],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if cleanup.returncode == 0:
                logger.warning(
                    "DB write failed after git tag creation; "
                    "rolled back tag '%s'",
                    tag,
                )
            else:
                logger.warning(
                    "DB write failed after git tag creation AND tag rollback "
                    "also failed (tag='%s', git stderr: %s)",
                    tag,
                    cleanup.stderr.strip(),
                )
            raise

        logger.info(
            "Shipped release %s (%s) with tag %s",
            release_id, release["version"], tag,
        )

        release["status"] = "shipped"
        release["git_tag"] = tag
        release["shipped_at"] = now
        return release

    # ── Aggregate info ──

    def get_release_info(self, release_id: str) -> Optional[dict]:
        """Full release info: release dict + checks + project info."""
        release = self.get_release(release_id)
        if not release:
            return None

        checks = self.get_checks(release_id)

        with self.db._lock:
            proj_row = self.db._conn.execute(
                "SELECT id, name, status, repo_path, target_date, "
                "created_at, completed_at FROM projects WHERE id = ?",
                (release["project_id"],),
            ).fetchone()

        project = dict(proj_row) if proj_row else None

        return {
            "release": release,
            "checks": checks,
            "project": project,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_release_manager: Optional[ReleaseManager] = None


def get_release_manager(db: Optional[SessionDB] = None) -> ReleaseManager:
    """Return the process-global ReleaseManager singleton."""
    global _release_manager
    if _release_manager is None:
        _release_manager = ReleaseManager(db)
    return _release_manager