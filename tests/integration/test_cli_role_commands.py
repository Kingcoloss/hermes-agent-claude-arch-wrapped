"""Integration tests for CLI role, KPI, and leaderboard slash commands."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI, _cprint
from hermes_cli.commands import resolve_command


def _make_cli():
    """Return a minimal HermesCLI stub suitable for handler tests."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = "sess-123"
    cli_obj._pending_input = MagicMock()
    cli_obj._agent_running = False
    return cli_obj


def _make_role(name, description="A role.", tool_count=3):
    """Return a mock RoleProfile-like object."""
    role = MagicMock()
    role.name = name
    role.description = description
    role.resolved_tools = [f"tool-{i}" for i in range(tool_count)]
    return role


@pytest.fixture
def capture_cprint(monkeypatch):
    """Patch cli._cprint and return a list that collects every printed line."""
    lines = []
    monkeypatch.setattr("cli._cprint", lines.append)
    return lines


# ────────────────────────────
# Registry / dispatch tests
# ────────────────────────────


class TestCommandRegistry:
    def test_role_command_is_resolvable(self):
        cmd = resolve_command("role")
        assert cmd is not None
        assert cmd.gateway_only is False

    def test_kpi_command_is_resolvable(self):
        cmd = resolve_command("kpi")
        assert cmd is not None
        assert cmd.gateway_only is False

    def test_leaderboard_command_is_resolvable(self):
        cmd = resolve_command("leaderboard")
        assert cmd is not None
        assert cmd.gateway_only is False

    def test_role_aliases_not_present(self):
        # role has no aliases; ensure we don't accidentally resolve something else
        assert resolve_command("role") is not None
        assert resolve_command("r") is None


# ────────────────────────────
# /role list
# ────────────────────────────


class TestRoleListCommand:
    def test_role_list_shows_empty_when_no_roles(self, capture_cprint):
        cli_obj = _make_cli()

        with patch("agent.role_manager.get_role_manager") as mock_rm:
            mgr = MagicMock()
            mgr.list_roles.return_value = []
            mock_rm.return_value = mgr

            cli_obj._handle_role_command("/role list")

        assert any("No roles configured." in line for line in capture_cprint)

    def test_role_list_shows_roles_with_tool_counts(self, capture_cprint):
        cli_obj = _make_cli()

        with patch("agent.role_manager.get_role_manager") as mock_rm:
            mgr = MagicMock()
            mgr.list_roles.return_value = ["devops", "quant-trader"]
            mgr.get_role.side_effect = lambda n: _make_role(
                n, description=f"{n} desc", tool_count=3 if n == "devops" else 5
            )
            mock_rm.return_value = mgr

            cli_obj._handle_role_command("/role list")

        output = "\n".join(capture_cprint)
        assert "Available Roles" in output
        assert "devops" in output
        assert "quant-trader" in output
        assert "devops desc" in output
        assert "3 tools" in output
        assert "5 tools" in output
        assert "/role switch <name>" in output

    def test_role_list_highlights_active_role(self, capture_cprint):
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(role="devops")

        with patch("agent.role_manager.get_role_manager") as mock_rm:
            mgr = MagicMock()
            mgr.list_roles.return_value = ["devops", "quant-trader"]
            mgr.get_role.side_effect = lambda n: _make_role(n, description=f"{n} desc")
            mock_rm.return_value = mgr

            cli_obj._handle_role_command("/role list")

        output = "\n".join(capture_cprint)
        # The active role gets a "●" marker
        assert "●" in output

    def test_role_list_defaults_when_no_subcommand(self, capture_cprint):
        cli_obj = _make_cli()

        with patch("agent.role_manager.get_role_manager") as mock_rm:
            mgr = MagicMock()
            mgr.list_roles.return_value = ["fullstack-dev"]
            mgr.get_role.side_effect = lambda n: _make_role(n, description="fs desc")
            mock_rm.return_value = mgr

            # /role with no subcommand should default to list
            cli_obj._handle_role_command("/role")

        assert any("fullstack-dev" in line for line in capture_cprint)


# ────────────────────────────
# /role switch
# ────────────────────────────


class TestRoleSwitchCommand:
    def test_role_switch_sets_agent_role(self, capture_cprint):
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(role=None)

        with patch("agent.role_manager.get_role_manager") as mock_rm:
            mgr = MagicMock()
            mgr.get_role.return_value = _make_role("devops", tool_count=7)
            mock_rm.return_value = mgr

            cli_obj._handle_role_command("/role switch devops")

        assert cli_obj.agent.role == "devops"
        output = "\n".join(capture_cprint)
        assert "Role set to: devops" in output
        assert "Tools available: 7" in output

    def test_role_switch_rejects_unknown_role(self, capture_cprint):
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(role=None)

        with patch("agent.role_manager.get_role_manager") as mock_rm:
            mgr = MagicMock()
            mgr.get_role.return_value = None
            mgr.list_roles.return_value = ["devops", "quant-trader"]
            mock_rm.return_value = mgr

            cli_obj._handle_role_command("/role switch unknown-role")

        assert cli_obj.agent.role is None
        output = "\n".join(capture_cprint)
        assert "Unknown role: unknown-role" in output
        assert "devops" in output
        assert "quant-trader" in output

    def test_role_switch_rejects_missing_argument(self, capture_cprint):
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(role="existing")

        with patch("agent.role_manager.get_role_manager") as mock_rm:
            mock_rm.return_value = MagicMock()
            cli_obj._handle_role_command("/role switch")

        assert cli_obj.agent.role == "existing"  # unchanged
        assert any("Usage: /role switch <name>" in line for line in capture_cprint)

    def test_role_switch_requires_active_agent(self, capture_cprint):
        cli_obj = _make_cli()
        cli_obj.agent = None

        with patch("agent.role_manager.get_role_manager") as mock_rm:
            mock_rm.return_value = MagicMock()
            cli_obj._handle_role_command("/role switch devops")

        assert any("No active agent" in line for line in capture_cprint)


# ────────────────────────────
# /kpi
# ────────────────────────────


class TestKPICommand:
    def test_kpi_with_explicit_role_shows_metrics(self, capture_cprint):
        cli_obj = _make_cli()

        with patch("agent.gamification.KPITracker") as mock_kpi_cls:
            tracker = MagicMock()
            tracker.get_kpi_summary.return_value = {
                "record_count": 12,
                "task_success_rate": 0.85,
                "avg_tokens_per_task": 1450.0,
                "tool_diversity_score": 3.5,
                "error_recovery_rate": 0.92,
                "role_proficiency_score": 2.8,
            }
            mock_kpi_cls.return_value = tracker

            cli_obj._show_kpi("/kpi devops")

        output = "\n".join(capture_cprint)
        assert "KPI Summary: devops" in output
        assert "12" in output  # record count
        assert "0.85" in output or "0.92" in output
        tracker.get_kpi_summary.assert_called_once_with(role="devops")

    def test_kpi_with_no_data_shows_empty_message(self, capture_cprint):
        cli_obj = _make_cli()

        with patch("agent.gamification.KPITracker") as mock_kpi_cls:
            tracker = MagicMock()
            tracker.get_kpi_summary.return_value = {"record_count": 0}
            mock_kpi_cls.return_value = tracker

            cli_obj._show_kpi("/kpi devops")

        assert any("No KPI data recorded yet for 'devops'." in line for line in capture_cprint)

    def test_kpi_without_arg_uses_agent_role(self, capture_cprint):
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(role="quant-trader")

        with patch("agent.gamification.KPITracker") as mock_kpi_cls:
            tracker = MagicMock()
            tracker.get_kpi_summary.return_value = {
                "record_count": 5,
                "task_success_rate": 0.77,
            }
            mock_kpi_cls.return_value = tracker

            cli_obj._show_kpi("/kpi")

        tracker.get_kpi_summary.assert_called_once_with(role="quant-trader")
        assert any("KPI Summary: quant-trader" in line for line in capture_cprint)

    def test_kpi_without_arg_and_no_agent_shows_usage(self, capture_cprint):
        cli_obj = _make_cli()
        cli_obj.agent = None

        cli_obj._show_kpi("/kpi")

        assert any("No active agent" in line for line in capture_cprint)
        assert any("/kpi <role>" in line for line in capture_cprint)


# ────────────────────────────
# /leaderboard
# ────────────────────────────


class TestLeaderboardCommand:
    def test_leaderboard_shows_ranked_rows(self, capture_cprint):
        cli_obj = _make_cli()

        with patch("agent.gamification.KPITracker") as mock_kpi_cls:
            tracker = MagicMock()
            tracker.get_leaderboard.return_value = [
                {"rank": 1, "skill_name": "devops", "level": 5, "xp": 420.0},
                {"rank": 2, "skill_name": "quant-trader", "level": 3, "xp": 210.0},
            ]
            mock_kpi_cls.return_value = tracker

            cli_obj._show_leaderboard("/leaderboard")

        output = "\n".join(capture_cprint)
        assert "Leaderboard" in output
        assert "devops" in output
        assert "quant-trader" in output
        assert "420" in output
        assert "210" in output
        tracker.get_leaderboard.assert_called_once_with(role=None, limit=10)

    def test_leaderboard_with_role_filter(self, capture_cprint):
        cli_obj = _make_cli()

        with patch("agent.gamification.KPITracker") as mock_kpi_cls:
            tracker = MagicMock()
            tracker.get_leaderboard.return_value = [
                {"rank": 1, "skill_name": "devops", "level": 2, "xp": 150.0},
            ]
            mock_kpi_cls.return_value = tracker

            cli_obj._show_leaderboard("/leaderboard devops")

        output = "\n".join(capture_cprint)
        assert "Leaderboard: devops" in output
        tracker.get_leaderboard.assert_called_once_with(role="devops", limit=10)

    def test_leaderboard_empty_no_filter(self, capture_cprint):
        cli_obj = _make_cli()

        with patch("agent.gamification.KPITracker") as mock_kpi_cls:
            tracker = MagicMock()
            tracker.get_leaderboard.return_value = []
            mock_kpi_cls.return_value = tracker

            cli_obj._show_leaderboard("/leaderboard")

        assert any("No leaderboard data yet." in line for line in capture_cprint)

    def test_leaderboard_empty_with_filter(self, capture_cprint):
        cli_obj = _make_cli()

        with patch("agent.gamification.KPITracker") as mock_kpi_cls:
            tracker = MagicMock()
            tracker.get_leaderboard.return_value = []
            mock_kpi_cls.return_value = tracker

            cli_obj._show_leaderboard("/leaderboard unknown-role")

        assert any("No leaderboard data for 'unknown-role' yet." in line for line in capture_cprint)

    def test_leaderboard_highlights_active_role(self, capture_cprint):
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(role="devops")

        with patch("agent.gamification.KPITracker") as mock_kpi_cls:
            tracker = MagicMock()
            tracker.get_leaderboard.return_value = [
                {"rank": 1, "skill_name": "devops", "level": 4, "xp": 300.0},
            ]
            mock_kpi_cls.return_value = tracker

            cli_obj._show_leaderboard("/leaderboard")

        output = "\n".join(capture_cprint)
        # Active role gets a "●" marker before the rank
        assert "●" in output


# ────────────────────────────
# End-to-end dispatch via process_command
# ────────────────────────────


class TestProcessCommandDispatch:
    """Verify that process_command routes to the correct handlers."""

    def test_process_command_role_dispatches_to_handler(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_handle_role_command") as mock_handler:
            result = cli_obj.process_command("/role list")
        assert result is True
        mock_handler.assert_called_once_with("/role list")

    def test_process_command_kpi_dispatches_to_handler(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_show_kpi") as mock_handler:
            result = cli_obj.process_command("/kpi devops")
        assert result is True
        mock_handler.assert_called_once_with("/kpi devops")

    def test_process_command_leaderboard_dispatches_to_handler(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_show_leaderboard") as mock_handler:
            result = cli_obj.process_command("/leaderboard")
        assert result is True
        mock_handler.assert_called_once_with("/leaderboard")

    def test_process_command_unknown_role_name_gives_error(self, capture_cprint):
        """Full integration: process_command → _handle_role_command → mock RM."""
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(role="existing")

        with patch("agent.role_manager.get_role_manager") as mock_rm:
            mgr = MagicMock()
            mgr.get_role.return_value = None
            mgr.list_roles.return_value = ["devops"]
            mock_rm.return_value = mgr

            result = cli_obj.process_command("/role switch not-a-role")

        assert result is True
        assert any("Unknown role: not-a-role" in line for line in capture_cprint)

    def test_process_command_kpi_no_data_path(self, capture_cprint):
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(role="test-role")

        with patch("agent.gamification.KPITracker") as mock_kpi_cls:
            tracker = MagicMock()
            tracker.get_kpi_summary.return_value = {"record_count": 0}
            mock_kpi_cls.return_value = tracker

            result = cli_obj.process_command("/kpi")

        assert result is True
        assert any("No KPI data recorded yet for 'test-role'." in line for line in capture_cprint)
