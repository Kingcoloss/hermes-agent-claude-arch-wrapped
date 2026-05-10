"""Tests for agent/release_manager.py — release lifecycle management."""

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.release_manager import ReleaseManager, get_release_manager
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(db: SessionDB, project_id: str = "proj1",
                  name: str = "hermes-v2",
                  repo_path: str = "/tmp/repo",
                  status: str = "proposed") -> None:
    """Insert a project row directly."""
    def _do(conn):
        conn.execute(
            "INSERT INTO projects (id, name, status, repo_path, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, name, status, repo_path, time.time()),
        )
    db._execute_write(_do)


def _make_release(rm: ReleaseManager, project_id: str = "proj1",
                  version: str = "0.1.0") -> dict:
    """Create a release via ReleaseManager (project must exist)."""
    return rm.create_release(project_id, version)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateRelease:
    """create_release: draft creation, validation, duplicate check."""

    def test_create_release(self, tmp_path):
        """Creates a draft release; get_release returns it."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db)

        release = rm.create_release("proj1", "0.1.0")
        assert release["status"] == "draft"
        assert release["version"] == "0.1.0"
        assert release["project_id"] == "proj1"
        assert release["git_tag"] is None
        assert release["shipped_at"] is None
        assert release["created_at"] is not None

        # Retrieve it
        got = rm.get_release(release["id"])
        assert got is not None
        assert got["id"] == release["id"]
        assert got["version"] == "0.1.0"

    def test_create_release_bad_version(self, tmp_path):
        """'abc' raises ValueError."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db)

        with pytest.raises(ValueError, match="Invalid version"):
            rm.create_release("proj1", "abc")

    def test_create_release_duplicate_version(self, tmp_path):
        """Same project+version raises ValueError."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db)

        rm.create_release("proj1", "0.1.0")
        with pytest.raises(ValueError, match="already exists"):
            rm.create_release("proj1", "0.1.0")

    def test_create_release_project_not_found(self, tmp_path):
        """Missing project_id raises ValueError."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)

        with pytest.raises(ValueError, match="not found"):
            rm.create_release("nonexistent", "0.1.0")


class TestRunCheck:
    """run_check: subprocess execution, status tracking."""

    def test_run_check_pass(self, tmp_path):
        """Mock subprocess returncode=0; check status='passed',
        release status transitions from 'draft' to 'testing'."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db)
        release = _make_release(rm)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "All tests passed"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            check = rm.run_check(release["id"], "tests")

        assert check["status"] == "passed"
        assert check["check_type"] == "tests"
        assert check["release_id"] == release["id"]

        # Release status should now be 'testing'
        updated = rm.get_release(release["id"])
        assert updated["status"] == "testing"

    def test_run_check_fail(self, tmp_path):
        """Mock subprocess returncode=1; check status='failed'."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db)
        release = _make_release(rm)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "2 tests failed"

        with patch("subprocess.run", return_value=mock_result):
            check = rm.run_check(release["id"], "tests")

        assert check["status"] == "failed"
        assert "failed" in check["result_text"]


class TestShipRelease:
    """ship_release: gating, git tagging, project activation."""

    def test_ship_release(self, tmp_path):
        """All checks pass; release shipped, git_tag set, project -> active."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db)
        release = _make_release(rm)

        # Mock check subprocess (passing)
        mock_check = MagicMock()
        mock_check.returncode = 0
        mock_check.stdout = "OK"
        mock_check.stderr = ""

        # Mock git tag subprocess (success)
        mock_tag = MagicMock()
        mock_tag.returncode = 0
        mock_tag.stdout = ""
        mock_tag.stderr = ""

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [mock_check, mock_tag]
            rm.run_check(release["id"], "tests")
            result = rm.ship_release(release["id"])

        assert result["status"] == "shipped"
        assert result["git_tag"] == "hermes-v2/v0.1.0"
        assert result["shipped_at"] is not None

        # Project should now be active
        with db._lock:
            proj = db._conn.execute(
                "SELECT status FROM projects WHERE id = 'proj1'"
            ).fetchone()
        assert proj["status"] == "active"

    def test_ship_without_checks_raises(self, tmp_path):
        """No checks run -> ValueError."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db)
        release = _make_release(rm)

        # Move to 'testing' status without running checks (simulated)
        def _do(conn):
            conn.execute(
                "UPDATE releases SET status = 'testing' WHERE id = ?",
                (release["id"],),
            )
        db._execute_write(_do)

        with pytest.raises(ValueError, match="no checks"):
            rm.ship_release(release["id"])

    def test_ship_with_failed_check_raises(self, tmp_path):
        """One check failed -> ValueError."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db)
        release = _make_release(rm)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "FAIL"

        with patch("subprocess.run", return_value=mock_result):
            rm.run_check(release["id"], "tests")

        with pytest.raises(ValueError, match="not passed"):
            rm.ship_release(release["id"])


class TestGetReleaseInfo:
    """get_release_info: aggregate release + checks + project."""

    def test_get_release_info(self, tmp_path):
        """Returns release dict + checks list + project dict."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db)
        release = _make_release(rm)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "All good"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            rm.run_check(release["id"], "tests")

        info = rm.get_release_info(release["id"])
        assert info is not None
        assert info["release"]["id"] == release["id"]
        assert len(info["checks"]) == 1
        assert info["checks"][0]["status"] == "passed"
        assert info["project"]["name"] == "hermes-v2"

    def test_get_release_info_not_found(self, tmp_path):
        """Returns None for nonexistent release."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        assert rm.get_release_info("nonexistent") is None


class TestListReleases:
    """list_releases: filtering by project."""

    def test_list_releases_filtered(self, tmp_path):
        """Only releases for the given project are returned."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db, project_id="proj1", name="alpha")
        _make_project(db, project_id="proj2", name="beta",
                      repo_path="/tmp/beta")

        rm.create_release("proj1", "0.1.0")
        rm.create_release("proj2", "0.2.0")

        only_proj1 = rm.list_releases(project_id="proj1")
        assert len(only_proj1) == 1
        assert only_proj1[0]["project_id"] == "proj1"


class TestSingleton:
    """get_release_manager returns a singleton."""

    def test_singleton(self, tmp_path):
        """Repeated calls return the same instance."""
        db = SessionDB(db_path=tmp_path / "test.db")
        # Reset singleton for test isolation
        import agent.release_manager as mod
        mod._release_manager = None

        rm1 = get_release_manager(db)
        rm2 = get_release_manager(db)
        assert rm1 is rm2

        # Cleanup
        mod._release_manager = None


class TestGitTagInjection:
    """Fix #2 — git tag injection: invalid project names must raise before subprocess."""

    def test_invalid_project_name_raises_before_subprocess(self, tmp_path):
        """Project name that starts with '-' must raise ValueError, never calling git."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        # Insert a project with a name that would produce a dangerous tag.
        _make_project(db, project_id="danger", name="-rf myproject",
                      repo_path=str(tmp_path))
        release = _make_release(rm, project_id="danger")

        # Move to testing status by inserting a fake passed check.
        def _do(conn):
            conn.execute(
                "UPDATE releases SET status = 'testing' WHERE id = ?",
                (release["id"],),
            )
            conn.execute(
                "INSERT INTO release_checks "
                "(id, release_id, check_type, status, result_text, checked_at) "
                "VALUES ('ckfake', ?, 'tests', 'passed', 'ok', ?)",
                (release["id"], time.time()),
            )
        db._execute_write(_do)

        with patch("subprocess.run") as mock_run:
            with pytest.raises(ValueError, match="Invalid git tag"):
                rm.ship_release(release["id"])
            # subprocess must NOT have been called for git tag creation.
            mock_run.assert_not_called()

    def test_tag_with_dotdot_raises(self, tmp_path):
        """Project name containing '..' produces an invalid tag — must raise."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db, project_id="dotdot", name="my..project",
                      repo_path=str(tmp_path))
        release = _make_release(rm, project_id="dotdot")

        def _do(conn):
            conn.execute(
                "UPDATE releases SET status = 'testing' WHERE id = ?",
                (release["id"],),
            )
            conn.execute(
                "INSERT INTO release_checks "
                "(id, release_id, check_type, status, result_text, checked_at) "
                "VALUES ('ckfake2', ?, 'tests', 'passed', 'ok', ?)",
                (release["id"], time.time()),
            )
        db._execute_write(_do)

        with patch("subprocess.run") as mock_run:
            with pytest.raises(ValueError, match="Invalid git tag"):
                rm.ship_release(release["id"])
            mock_run.assert_not_called()


class TestRunCheckCustomCommand:
    """Fix #3 — custom command for run_check."""

    def test_run_check_custom_command(self, tmp_path):
        """Passing command=['pytest'] uses that command, not run_tests.sh."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db)
        release = _make_release(rm)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "pytest output"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            check = rm.run_check(release["id"], "tests", command=["pytest"])

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["pytest"]
        assert check["status"] == "passed"

    def test_run_check_default_command(self, tmp_path):
        """When command=None, defaults to ['scripts/run_tests.sh']."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db)
        release = _make_release(rm)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            rm.run_check(release["id"], "tests")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["scripts/run_tests.sh"]


class TestShipReleaseRollback:
    """Fix #6 — orphaned tag cleanup on DB write failure."""

    def test_db_failure_triggers_tag_delete(self, tmp_path):
        """If DB write fails after git tag succeeds, 'git tag -d' is called."""
        db = SessionDB(db_path=tmp_path / "test.db")
        rm = ReleaseManager(db=db)
        _make_project(db)
        release = _make_release(rm)

        # Move to testing with a passed check.
        def _do(conn):
            conn.execute(
                "UPDATE releases SET status = 'testing' WHERE id = ?",
                (release["id"],),
            )
            conn.execute(
                "INSERT INTO release_checks "
                "(id, release_id, check_type, status, result_text, checked_at) "
                "VALUES ('ckroll', ?, 'tests', 'passed', 'ok', ?)",
                (release["id"], time.time()),
            )
        db._execute_write(_do)

        # First subprocess.run call is git tag (success); second is git tag -d (success).
        mock_tag_ok = MagicMock()
        mock_tag_ok.returncode = 0
        mock_tag_ok.stdout = ""
        mock_tag_ok.stderr = ""

        mock_delete_ok = MagicMock()
        mock_delete_ok.returncode = 0
        mock_delete_ok.stdout = ""
        mock_delete_ok.stderr = ""

        with patch("subprocess.run", side_effect=[mock_tag_ok, mock_delete_ok]) as mock_run:
            with patch.object(db, "_execute_write", side_effect=RuntimeError("DB down")):
                with pytest.raises(RuntimeError, match="DB down"):
                    rm.ship_release(release["id"])

        # Verify git tag -d was called with the expected tag.
        assert mock_run.call_count == 2
        delete_call = mock_run.call_args_list[1]
        cmd = delete_call[0][0]
        assert cmd[:3] == ["git", "tag", "-d"]
        assert cmd[3] == "hermes-v2/v0.1.0"