"""Tests for tools/simulation_tool.py — deterministic simulation handlers."""

import json
import pytest

from tools.simulation_tool import (
    _sim_run_handler,
    _sim_monte_carlo_option_handler,
    _sim_save_state_handler,
    _sim_load_state_handler,
    _check_simulation_requirements,
    _MAX_RESULT_ARRAY_LEN,
)

try:
    import numpy as np  # noqa: F401
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False


# =========================================================================
# Requirements / availability
# =========================================================================

class TestSimulationRequirements:
    def test_check_requirements_when_numpy_present(self):
        assert _check_simulation_requirements() == NUMPY_AVAILABLE


# =========================================================================
# sim_run — Monte Carlo
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
class TestSimRunMonteCarlo:
    def test_monte_carlo_normal_reproducible(self):
        args = {
            "kind": "monte_carlo",
            "steps": 100,
            "seed": 42,
            "config": {
                "initial_value": 100.0,
                "drift": 0.01,
                "volatility": 0.2,
                "distribution": "normal",
            },
        }
        result1 = json.loads(_sim_run_handler(args))
        result2 = json.loads(_sim_run_handler(args))

        assert "error" not in result1
        assert "error" not in result2
        # Reproducibility: same seed -> same results
        assert result1["results"] == result2["results"]
        assert result1["seed_used"] == 42
        assert "summary" in result1
        assert "elapsed_ms" in result1

    def test_monte_carlo_uniform(self):
        args = {
            "kind": "monte_carlo",
            "steps": 50,
            "seed": 7,
            "config": {
                "initial_value": 10.0,
                "drift": 0.0,
                "volatility": 1.0,
                "distribution": "uniform",
            },
        }
        result = json.loads(_sim_run_handler(args))
        assert "error" not in result
        assert result["kind"] == "monte_carlo"
        assert len(result["results"]) == 51  # initial + steps

    def test_monte_carlo_summary_stats_shape(self):
        args = {
            "kind": "monte_carlo",
            "steps": 1000,
            "seed": 1,
            "config": {
                "initial_value": 50.0,
                "drift": 0.0,
                "volatility": 0.1,
                "distribution": "normal",
            },
        }
        result = json.loads(_sim_run_handler(args))
        summary = result["summary"]
        assert all(k in summary for k in ("mean", "std", "min", "max", "final"))
        assert isinstance(summary["mean"], float)
        assert isinstance(summary["std"], float)

    def test_auto_seed_when_none_provided(self):
        args = {
            "kind": "monte_carlo",
            "steps": 10,
            "config": {"initial_value": 0.0},
        }
        result = json.loads(_sim_run_handler(args))
        assert "seed_used" in result
        assert isinstance(result["seed_used"], int)

    def test_invalid_kind_rejected(self):
        args = {"kind": "invalid", "steps": 10, "seed": 1}
        result = json.loads(_sim_run_handler(args))
        assert result.get("success") is False
        assert "kind" in result["error"].lower()

    def test_zero_or_negative_steps_rejected(self):
        for steps in (0, -5):
            args = {"kind": "monte_carlo", "steps": steps, "seed": 1}
            result = json.loads(_sim_run_handler(args))
            assert result.get("success") is False

    def test_invalid_distribution_rejected(self):
        args = {
            "kind": "monte_carlo",
            "steps": 10,
            "seed": 1,
            "config": {"distribution": "exponential"},
        }
        result = json.loads(_sim_run_handler(args))
        assert result.get("success") is False
        assert "distribution" in result["error"].lower()

    def test_large_result_array_trimmed(self):
        args = {
            "kind": "monte_carlo",
            "steps": _MAX_RESULT_ARRAY_LEN + 500,
            "seed": 1,
            "config": {"initial_value": 0.0},
        }
        result = json.loads(_sim_run_handler(args))
        assert len(result["results"]) == _MAX_RESULT_ARRAY_LEN
        assert "_note" in result
        assert "trimmed" in result["_note"].lower()


# =========================================================================
# sim_run — Discrete Event
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
class TestSimRunDiscreteEvent:
    def test_discrete_event_basic(self):
        args = {
            "kind": "discrete_event",
            "steps": 20,
            "seed": 99,
            "config": {
                "arrival_rate": 2.0,
                "service_rate": 3.0,
                "servers": 2,
            },
        }
        result = json.loads(_sim_run_handler(args))
        assert "error" not in result
        assert result["kind"] == "discrete_event"
        assert len(result["results"]) == 20
        first = result["results"][0]
        assert all(k in first for k in ("time", "in_service", "queue_length", "servers_busy"))

    def test_discrete_event_reproducible(self):
        args = {
            "kind": "discrete_event",
            "steps": 50,
            "seed": 123,
            "config": {"arrival_rate": 1.0, "service_rate": 2.0, "servers": 1},
        }
        result1 = json.loads(_sim_run_handler(args))
        result2 = json.loads(_sim_run_handler(args))
        assert result1["results"] == result2["results"]

    def test_discrete_event_summary_stats(self):
        args = {
            "kind": "discrete_event",
            "steps": 100,
            "seed": 1,
            "config": {"arrival_rate": 1.0, "service_rate": 2.0, "servers": 1},
        }
        result = json.loads(_sim_run_handler(args))
        summary = result["summary"]
        assert all(k in summary for k in ("mean", "std", "min", "max", "final"))
        # Queue length should be non-negative
        assert summary["min"] >= 0

    def test_invalid_servers_rejected(self):
        args = {
            "kind": "discrete_event",
            "steps": 10,
            "seed": 1,
            "config": {"arrival_rate": 1.0, "service_rate": 1.0, "servers": 0},
        }
        result = json.loads(_sim_run_handler(args))
        assert result.get("success") is False
        assert "servers" in result["error"].lower()


# =========================================================================
# sim_monte_carlo_option
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
class TestSimMonteCarloOption:
    def test_call_option_pricing(self):
        args = {
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "num_paths": 5000,
            "seed": 42,
            "option_type": "call",
        }
        result = json.loads(_sim_monte_carlo_option_handler(args))
        assert "error" not in result
        assert result["option_type"] == "call"
        assert result["paths_simulated"] == 5000
        assert result["seed_used"] == 42
        assert result["price_estimate"] > 0
        assert result["standard_error"] >= 0
        ci = result["confidence_interval_95"]
        assert len(ci) == 2
        assert ci[0] <= result["price_estimate"] <= ci[1]

    def test_put_option_pricing(self):
        args = {
            "spot": 100.0,
            "strike": 110.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "num_paths": 5000,
            "seed": 42,
            "option_type": "put",
        }
        result = json.loads(_sim_monte_carlo_option_handler(args))
        assert "error" not in result
        assert result["option_type"] == "put"
        assert result["price_estimate"] > 0

    def test_reproducible_with_same_seed(self):
        args = {
            "spot": 100.0,
            "strike": 105.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "num_paths": 2000,
            "seed": 77,
            "option_type": "call",
        }
        result1 = json.loads(_sim_monte_carlo_option_handler(args))
        result2 = json.loads(_sim_monte_carlo_option_handler(args))
        assert result1["price_estimate"] == result2["price_estimate"]
        assert result1["standard_error"] == result2["standard_error"]

    def test_auto_seed_when_none(self):
        args = {
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "num_paths": 1000,
        }
        result = json.loads(_sim_monte_carlo_option_handler(args))
        assert "seed_used" in result
        assert isinstance(result["seed_used"], int)

    def test_invalid_option_type_rejected(self):
        args = {
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "num_paths": 1000,
            "seed": 1,
            "option_type": "straddle",
        }
        result = json.loads(_sim_monte_carlo_option_handler(args))
        assert result.get("success") is False
        assert "option_type" in result["error"].lower()

    def test_invalid_inputs_rejected(self):
        for bad in (
            {"spot": -100, "strike": 100, "maturity": 1, "risk_free_rate": 0.05, "volatility": 0.2},
            {"spot": 100, "strike": 0, "maturity": 1, "risk_free_rate": 0.05, "volatility": 0.2},
            {"spot": 100, "strike": 100, "maturity": 0, "risk_free_rate": 0.05, "volatility": 0.2},
            {"spot": 100, "strike": 100, "maturity": 1, "risk_free_rate": 0.05, "volatility": -0.1},
        ):
            bad.setdefault("num_paths", 100)
            bad.setdefault("seed", 1)
            result = json.loads(_sim_monte_carlo_option_handler(bad))
            assert result.get("success") is False, f"Expected failure for args {bad}"

    def test_default_num_paths(self):
        args = {
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "seed": 1,
        }
        result = json.loads(_sim_monte_carlo_option_handler(args))
        assert result["paths_simulated"] == 10000


# =========================================================================
# sim_save_state / sim_load_state
# =========================================================================

class TestSimSaveLoadState:
    def test_save_returns_sha256_id(self):
        state = {"foo": "bar", "count": 3}
        result = json.loads(_sim_save_state_handler({"state": state}))
        assert result["success"] is True
        assert len(result["state_id"]) == 64
        assert result["saved_at"].startswith("20")

    def test_save_and_load_roundtrip(self):
        state = {"matrix": [[1, 2], [3, 4]], "meta": {"seed": 42}}
        saved = json.loads(_sim_save_state_handler({"state": state}))
        state_id = saved["state_id"]

        loaded = json.loads(_sim_load_state_handler({"state_id": state_id}))
        assert loaded["success"] is True
        assert loaded["state"] == state

    def test_load_missing_state(self):
        result = json.loads(_sim_load_state_handler({"state_id": "nonexistent"}))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_load_invalid_state_id_rejected(self):
        for bad_id in ("../etc/passwd", "foo/bar", ".hidden", "foo\\bar"):
            result = json.loads(_sim_load_state_handler({"state_id": bad_id}))
            assert result.get("success") is False
            assert "invalid" in result["error"].lower()

    def test_save_state_must_be_dict(self):
        result = json.loads(_sim_save_state_handler({"state": "not a dict"}))
        assert result.get("success") is False
        assert "dict" in result["error"].lower()

    def test_save_state_requires_state_key(self):
        result = json.loads(_sim_save_state_handler({}))
        assert result.get("success") is False

    def test_load_state_requires_state_id(self):
        result = json.loads(_sim_load_state_handler({}))
        assert result.get("success") is False

    def test_save_is_deterministic_same_state_same_id(self):
        state = {"a": 1, "b": [2, 3]}
        r1 = json.loads(_sim_save_state_handler({"state": state}))
        r2 = json.loads(_sim_save_state_handler({"state": state}))
        assert r1["state_id"] == r2["state_id"]

    def test_save_different_states_different_ids(self):
        r1 = json.loads(_sim_save_state_handler({"state": {"a": 1}}))
        r2 = json.loads(_sim_save_state_handler({"state": {"a": 2}}))
        assert r1["state_id"] != r2["state_id"]

    def test_state_id_is_sha256_of_sorted_json(self):
        import hashlib
        state = {"z": 1, "a": 2}
        expected_json = json.dumps(state, sort_keys=True, ensure_ascii=False)
        expected_id = hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
        result = json.loads(_sim_save_state_handler({"state": state}))
        assert result["state_id"] == expected_id
