"""Integration tests for Web Server REST API endpoints for role/KPI management."""

import time
from unittest.mock import patch, MagicMock

import pytest

# Skip entire module if fastapi/starlette is not installed
try:
    from starlette.testclient import TestClient
except ImportError:
    pytest.skip("fastapi/starlette not installed", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers / mocks
# ---------------------------------------------------------------------------

class MockRoleProfile:
    """Minimal stand-in for agent.role_manager.RoleProfile."""

    def __init__(self, name, description, toolsets, default_model, skin, kpi_weights):
        self.name = name
        self.description = description
        self.toolsets = toolsets
        self.default_model = default_model
        self.skin = skin
        self.kpi_weights = kpi_weights

    @property
    def resolved_tools(self):
        return ["mock_tool_a", "mock_tool_b"]


class MockRoleManager:
    """Minimal stand-in for agent.role_manager.RoleManager."""

    def __init__(self, roles):
        self._roles = roles

    def list_roles(self):
        return sorted(self._roles.keys())

    def get_role(self, name):
        return self._roles.get(name)


def _make_mock_role_manager():
    return MockRoleManager({
        "fullstack-dev": MockRoleProfile(
            name="fullstack-dev",
            description="Frontend, backend, database, API, testing, and deployment.",
            toolsets=["fullstack-dev"],
            default_model="anthropic/claude-sonnet-4",
            skin="default",
            kpi_weights={
                "task_success_rate": 1.2,
                "avg_tokens_per_task": 0.8,
                "tool_diversity_score": 1.0,
                "error_recovery_rate": 1.2,
                "role_proficiency_score": 1.2,
            },
        ),
        "quant-trader": MockRoleProfile(
            name="quant-trader",
            description="Statistical arbitrage and backtesting.",
            toolsets=["quant-trader"],
            default_model=None,
            skin=None,
            kpi_weights={
                "task_success_rate": 1.2,
                "avg_tokens_per_task": 0.3,
                "tool_diversity_score": 0.8,
                "error_recovery_rate": 1.0,
                "role_proficiency_score": 1.5,
            },
        ),
    })


# ---------------------------------------------------------------------------
# Role endpoint tests
# ---------------------------------------------------------------------------

class TestRoleEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, _isolate_hermes_home):
        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"

    def test_get_roles_returns_200(self, monkeypatch):
        """GET /api/roles should return 200 with a list of roles."""
        import agent.role_manager as rm_module
        monkeypatch.setattr(rm_module, "get_role_manager", _make_mock_role_manager)

        resp = self.client.get("/api/roles")
        assert resp.status_code == 200
        data = resp.json()
        assert "roles" in data
        assert isinstance(data["roles"], list)
        assert len(data["roles"]) == 2

    def test_get_roles_schema(self, monkeypatch):
        """Verify each role object has the expected fields."""
        import agent.role_manager as rm_module
        monkeypatch.setattr(rm_module, "get_role_manager", _make_mock_role_manager)

        resp = self.client.get("/api/roles")
        data = resp.json()
        role = data["roles"][0]
        assert "name" in role
        assert "description" in role
        assert "toolsets" in role
        assert "default_model" in role
        assert "skin" in role
        assert "kpi_weights" in role
        assert "resolved_tool_count" in role
        assert isinstance(role["toolsets"], list)
        assert isinstance(role["kpi_weights"], dict)
        assert isinstance(role["resolved_tool_count"], int)

    def test_get_role_by_name_returns_200(self, monkeypatch):
        """GET /api/roles/{name} should return a single role profile."""
        import agent.role_manager as rm_module
        monkeypatch.setattr(rm_module, "get_role_manager", _make_mock_role_manager)

        resp = self.client.get("/api/roles/fullstack-dev")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "fullstack-dev"
        assert data["description"] == "Frontend, backend, database, API, testing, and deployment."
        assert data["default_model"] == "anthropic/claude-sonnet-4"
        assert data["skin"] == "default"
        assert data["resolved_tool_count"] == 2

    def test_get_role_by_name_returns_404(self, monkeypatch):
        """GET /api/roles/{name} should return 404 for an unknown role."""
        import agent.role_manager as rm_module
        monkeypatch.setattr(rm_module, "get_role_manager", _make_mock_role_manager)

        resp = self.client.get("/api/roles/nonexistent-role")
        assert resp.status_code == 404
        assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# KPI endpoint tests
# ---------------------------------------------------------------------------

class TestKPIEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, _isolate_hermes_home):
        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"

    @pytest.fixture
    def _seed_kpi_data(self):
        """Insert KPI, XP, and achievement records into the isolated DB."""
        from hermes_state import SessionDB
        db = SessionDB()
        now = time.time()

        try:
            # Create parent sessions to satisfy the FK constraint on agent_kpi.session_id
            db.create_session("sess-1", "cli")
            db.create_session("sess-2", "cli")

            # Seed KPI data
            db._conn.executemany(
                """
                INSERT INTO agent_kpi (session_id, role, metric_name, metric_value, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    ("sess-1", "fullstack-dev", "task_success_rate", 0.95, now - 86400),
                    ("sess-1", "fullstack-dev", "avg_tokens_per_task", 120.5, now - 86400),
                    ("sess-2", "quant-trader", "task_success_rate", 0.88, now - 172800),
                    ("sess-2", "quant-trader", "role_proficiency_score", 1.5, now - 172800),
                ],
            )

            # Seed XP data
            db._conn.executemany(
                """
                INSERT INTO agent_skills_xp (skill_name, xp, level, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(skill_name) DO UPDATE SET
                    xp = excluded.xp,
                    level = excluded.level,
                    last_updated = excluded.last_updated
                """,
                [
                    ("fullstack-dev", 250.0, 3, now),
                    ("quant-trader", 180.0, 2, now),
                    ("devops", 95.0, 1, now),
                ],
            )

            # Seed achievements
            db._conn.executemany(
                """
                INSERT INTO agent_achievements (achievement_id, name, description, role, unlocked_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    ("ach-1", "First Commit", "Made first commit", "fullstack-dev", now - 86400),
                    ("ach-2", "Bug Hunter", "Found 10 bugs", "fullstack-dev", now - 43200),
                    ("ach-3", "Market Analyst", "Analyzed 100 trades", "quant-trader", now - 172800),
                ],
            )
            db._conn.commit()
        finally:
            db.close()

    def test_get_kpi_summary_no_params(self, _seed_kpi_data):
        """GET /api/kpi with no params should aggregate all KPIs."""
        resp = self.client.get("/api/kpi")
        assert resp.status_code == 200
        data = resp.json()
        assert "record_count" in data
        assert data["record_count"] == 4
        assert data["task_success_rate"] is not None

    def test_get_kpi_summary_with_role_and_days(self, _seed_kpi_data):
        """GET /api/kpi?role=fullstack-dev&days=7 should filter correctly."""
        resp = self.client.get("/api/kpi?role=fullstack-dev&days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert data["record_count"] == 2
        assert data["task_success_rate"] == 0.95
        assert data["avg_tokens_per_task"] == 120.5

    def test_get_kpi_invalid_days_returns_422(self):
        """GET /api/kpi with non-int days should return 422."""
        resp = self.client.get("/api/kpi?days=notanumber")
        assert resp.status_code == 422

    def test_get_xp_all_skills(self, _seed_kpi_data):
        """GET /api/xp should return all skills ordered by XP desc."""
        resp = self.client.get("/api/xp")
        assert resp.status_code == 200
        data = resp.json()
        assert "skills" in data
        assert len(data["skills"]) == 3
        # Ordered by xp DESC
        assert data["skills"][0]["skill_name"] == "fullstack-dev"
        assert data["skills"][0]["xp"] == 250.0
        assert data["skills"][0]["level"] == 3

    def test_get_xp_single_skill(self, _seed_kpi_data):
        """GET /api/xp?skill=quant-trader should return level info for that skill."""
        resp = self.client.get("/api/xp?skill=quant-trader")
        assert resp.status_code == 200
        data = resp.json()
        assert data["skill_name"] == "quant-trader"
        assert data["level"] == 2
        assert data["xp"] == 180.0
        assert data["xp_to_next"] == 20.0  # level 3 needs 200, has 180

    def test_get_xp_unknown_skill(self):
        """GET /api/xp?skill=unknown should return default level 1."""
        resp = self.client.get("/api/xp?skill=unknown")
        assert resp.status_code == 200
        data = resp.json()
        assert data["skill_name"] == "unknown"
        assert data["level"] == 1
        assert data["xp"] == 0.0

    def test_get_achievements_all(self, _seed_kpi_data):
        """GET /api/achievements should return all achievements."""
        resp = self.client.get("/api/achievements")
        assert resp.status_code == 200
        data = resp.json()
        assert "achievements" in data
        assert len(data["achievements"]) == 3

    def test_get_achievements_by_role(self, _seed_kpi_data):
        """GET /api/achievements?role=fullstack-dev should filter by role."""
        resp = self.client.get("/api/achievements?role=fullstack-dev")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["achievements"]) == 2
        for ach in data["achievements"]:
            assert ach["role"] == "fullstack-dev"

    def test_get_achievements_unknown_role(self, _seed_kpi_data):
        """GET /api/achievements?role=unknown should return empty list."""
        resp = self.client.get("/api/achievements?role=unknown")
        assert resp.status_code == 200
        data = resp.json()
        assert data["achievements"] == []

    def test_get_leaderboard_default(self, _seed_kpi_data):
        """GET /api/leaderboard should return ranked skills by XP."""
        resp = self.client.get("/api/leaderboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "leaderboard" in data
        assert len(data["leaderboard"]) == 3
        # Check rank assignment
        assert data["leaderboard"][0]["rank"] == 1
        assert data["leaderboard"][0]["skill_name"] == "fullstack-dev"
        assert data["leaderboard"][1]["rank"] == 2
        assert data["leaderboard"][2]["rank"] == 3

    def test_get_leaderboard_with_role_and_limit(self, _seed_kpi_data):
        """GET /api/leaderboard?role=devops&limit=1 should filter and limit."""
        resp = self.client.get("/api/leaderboard?role=devops&limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["leaderboard"]) == 1
        assert data["leaderboard"][0]["skill_name"] == "devops"

    def test_get_leaderboard_invalid_limit_returns_422(self):
        """GET /api/leaderboard?limit=notanumber should return 422."""
        resp = self.client.get("/api/leaderboard?limit=notanumber")
        assert resp.status_code == 422
