"""Simulation Tools for Hermes Agent.

Deterministic simulation tools with seeded random number generation for
reproducible Monte Carlo and discrete-event experiments.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# Maximum number of raw step results to include in the output.
_MAX_RESULT_ARRAY_LEN = 1000


def _trim_results(results: List[Any]) -> List[Any]:
    """Trim a large result array to a safe size for model context."""
    if len(results) <= _MAX_RESULT_ARRAY_LEN:
        return results
    return results[:_MAX_RESULT_ARRAY_LEN]


def _summary_stats(values: List[float]) -> Dict[str, float]:
    """Return mean, std, min, max, and final value for a list of floats."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "final": float(arr[-1]) if arr.size else 0.0,
    }


# ---------------------------------------------------------------------------
# sim_run
# ---------------------------------------------------------------------------

def _sim_run_handler(args: Dict[str, Any], **kwargs) -> str:
    """Run a deterministic simulation."""
    kind = (args.get("kind") or "").strip().lower()
    steps = int(args.get("steps", 0))
    seed = args.get("seed")
    config = args.get("config") or {}

    if kind not in ("monte_carlo", "discrete_event"):
        return tool_error("kind must be 'monte_carlo' or 'discrete_event'", success=False)

    if steps <= 0:
        return tool_error("steps must be a positive integer", success=False)

    # Resolve seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
    else:
        seed = int(seed)

    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        start_ms = time.monotonic()

        if kind == "monte_carlo":
            initial_value = float(config.get("initial_value", 0.0))
            drift = float(config.get("drift", 0.0))
            volatility = float(config.get("volatility", 0.1))
            distribution = (config.get("distribution") or "normal").strip().lower()

            if distribution not in ("normal", "uniform"):
                return tool_error("distribution must be 'normal' or 'uniform'", success=False)

            results: List[float] = [float(initial_value)]
            for _ in range(steps):
                if distribution == "normal":
                    shock = rng.normal(loc=drift, scale=volatility)
                else:
                    # uniform centered around drift with width volatility
                    low = drift - volatility
                    high = drift + volatility
                    shock = rng.uniform(low=low, high=high)
                next_val = results[-1] + shock
                results.append(next_val)

        else:  # discrete_event
            arrival_rate = float(config.get("arrival_rate", 1.0))
            service_rate = float(config.get("service_rate", 1.0))
            servers = int(config.get("servers", 1))

            if servers <= 0:
                return tool_error("servers must be a positive integer", success=False)

            # Simple M/M/c queue simulation
            time_now = 0.0
            queue: List[float] = []
            busy_until: List[float] = [0.0] * servers
            results = []

            for _ in range(steps):
                # Next arrival
                inter_arrival = rng.exponential(1.0 / arrival_rate)
                time_now += inter_arrival

                # Try to assign to a server
                assigned = False
                for i in range(servers):
                    if busy_until[i] <= time_now:
                        service_time = rng.exponential(1.0 / service_rate)
                        busy_until[i] = time_now + service_time
                        assigned = True
                        break

                if not assigned:
                    queue.append(time_now)

                # Snapshot system state
                in_service = sum(1 for b in busy_until if b > time_now)
                results.append(
                    {
                        "time": round(time_now, 6),
                        "in_service": in_service,
                        "queue_length": len(queue),
                        "servers_busy": in_service,
                    }
                )

                # Drain queue if possible
                new_queue = []
                for arrival_time in queue:
                    served = False
                    for i in range(servers):
                        if busy_until[i] <= arrival_time:
                            service_time = rng.exponential(1.0 / service_rate)
                            busy_until[i] = arrival_time + service_time
                            served = True
                            break
                    if not served:
                        new_queue.append(arrival_time)
                queue = new_queue

        elapsed_ms = round((time.monotonic() - start_ms) * 1000, 3)

        # For discrete_event we return object steps; compute numeric summary
        # over a key metric (e.g., queue_length) so stats are still meaningful.
        if kind == "monte_carlo":
            summary = _summary_stats(results)
            trimmed = _trim_results(results)
            payload = {
                "results": trimmed,
                "summary": summary,
                "seed_used": seed,
                "elapsed_ms": elapsed_ms,
                "kind": kind,
                "steps": steps,
            }
        else:
            queue_lengths = [step["queue_length"] for step in results]
            summary = _summary_stats(queue_lengths)
            trimmed = _trim_results(results)
            payload = {
                "results": trimmed,
                "summary": summary,
                "seed_used": seed,
                "elapsed_ms": elapsed_ms,
                "kind": kind,
                "steps": steps,
            }

        if len(results) > _MAX_RESULT_ARRAY_LEN:
            payload["_note"] = (
                f"Results array trimmed from {len(results)} to {_MAX_RESULT_ARRAY_LEN} entries. "
                "Summary statistics are computed over the full simulation."
            )

        return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        logger.exception("sim_run failed: %s", e)
        return tool_error(str(e), success=False)


SIM_RUN_SCHEMA = {
    "name": "sim_run",
    "description": (
        "Run a deterministic simulation with seeded random number generation. "
        "Supports Monte Carlo random walks and discrete-event queue simulations. "
        "Results are reproducible when the same seed is provided."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["monte_carlo", "discrete_event"],
                "description": "Type of simulation to run",
            },
            "steps": {
                "type": "integer",
                "description": "Number of simulation steps",
                "minimum": 1,
            },
            "seed": {
                "type": "integer",
                "description": "Optional random seed for reproducibility. Auto-generated if omitted.",
            },
            "config": {
                "type": "object",
                "description": (
                    "Simulation-specific configuration. For monte_carlo: "
                    "{initial_value, drift, volatility, distribution ('normal'|'uniform')}. "
                    "For discrete_event: {arrival_rate, service_rate, servers}."
                ),
            },
        },
        "required": ["kind", "steps"],
    },
}


# ---------------------------------------------------------------------------
# sim_monte_carlo_option
# ---------------------------------------------------------------------------

def _sim_monte_carlo_option_handler(args: Dict[str, Any], **kwargs) -> str:
    """Price an option via Monte Carlo simulation of geometric Brownian motion."""
    spot = float(args.get("spot", 0.0))
    strike = float(args.get("strike", 0.0))
    maturity = float(args.get("maturity", 0.0))
    risk_free_rate = float(args.get("risk_free_rate", 0.0))
    volatility = float(args.get("volatility", 0.0))
    num_paths = int(args.get("num_paths", 10000))
    seed = args.get("seed")
    option_type = (args.get("option_type") or "call").strip().lower()

    if spot <= 0 or strike <= 0 or maturity <= 0 or volatility < 0:
        return tool_error("spot, strike, maturity must be positive and volatility must be non-negative", success=False)

    if option_type not in ("call", "put"):
        return tool_error("option_type must be 'call' or 'put'", success=False)

    if num_paths <= 0:
        return tool_error("num_paths must be a positive integer", success=False)

    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
    else:
        seed = int(seed)

    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        start_ms = time.monotonic()

        # Geometric Brownian Motion: dS = rS dt + sigma S dW
        dt = maturity / 252.0  # daily steps assuming 252 trading days/year
        n_steps = max(1, int(maturity * 252))

        # Antithetic variates for variance reduction: generate half the paths and mirror them
        half_paths = num_paths // 2
        if half_paths == 0:
            half_paths = num_paths

        Z = rng.standard_normal((half_paths, n_steps))

        # Drift and diffusion per step
        drift_per_step = (risk_free_rate - 0.5 * volatility**2) * dt
        diffusion_per_step = volatility * np.sqrt(dt)

        log_returns = drift_per_step + diffusion_per_step * Z
        # Antithetic paths
        log_returns_anti = drift_per_step + diffusion_per_step * (-Z)

        # Cumulative sum to get log(S_T / S_0)
        cum_log_returns = np.cumsum(log_returns, axis=1)
        cum_log_returns_anti = np.cumsum(log_returns_anti, axis=1)

        S_T = spot * np.exp(cum_log_returns[:, -1])
        S_T_anti = spot * np.exp(cum_log_returns_anti[:, -1])

        # Combine original and antithetic
        all_S_T = np.concatenate([S_T, S_T_anti])
        # If num_paths is odd, drop the extra antithetic path
        all_S_T = all_S_T[:num_paths]

        if option_type == "call":
            payoffs = np.maximum(all_S_T - strike, 0.0)
        else:
            payoffs = np.maximum(strike - all_S_T, 0.0)

        # Discount to present value
        discounted = np.exp(-risk_free_rate * maturity) * payoffs

        price_estimate = float(np.mean(discounted))
        standard_error = float(np.std(discounted, ddof=1) / np.sqrt(num_paths))
        ci_lower = float(price_estimate - 1.96 * standard_error)
        ci_upper = float(price_estimate + 1.96 * standard_error)

        elapsed_ms = round((time.monotonic() - start_ms) * 1000, 3)

        payload = {
            "price_estimate": price_estimate,
            "standard_error": standard_error,
            "confidence_interval_95": [ci_lower, ci_upper],
            "paths_simulated": num_paths,
            "seed_used": seed,
            "elapsed_ms": elapsed_ms,
            "option_type": option_type,
            "spot": spot,
            "strike": strike,
            "maturity": maturity,
            "risk_free_rate": risk_free_rate,
            "volatility": volatility,
        }

        return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        logger.exception("sim_monte_carlo_option failed: %s", e)
        return tool_error(str(e), success=False)


SIM_MONTE_CARLO_OPTION_SCHEMA = {
    "name": "sim_monte_carlo_option",
    "description": (
        "Price a European option using Monte Carlo simulation of geometric Brownian motion. "
        "Simulates asset price paths under the risk-neutral measure and discounts the average payoff."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "spot": {
                "type": "number",
                "description": "Current underlying asset price",
            },
            "strike": {
                "type": "number",
                "description": "Option strike price",
            },
            "maturity": {
                "type": "number",
                "description": "Time to maturity in years",
            },
            "risk_free_rate": {
                "type": "number",
                "description": "Annual risk-free interest rate (e.g., 0.05 for 5%)",
            },
            "volatility": {
                "type": "number",
                "description": "Annual volatility of the underlying (e.g., 0.2 for 20%)",
            },
            "num_paths": {
                "type": "integer",
                "description": "Number of Monte Carlo paths to simulate",
                "default": 10000,
                "minimum": 1,
            },
            "seed": {
                "type": "integer",
                "description": "Optional random seed for reproducibility",
            },
            "option_type": {
                "type": "string",
                "enum": ["call", "put"],
                "description": "Option type",
                "default": "call",
            },
        },
        "required": ["spot", "strike", "maturity", "risk_free_rate", "volatility"],
    },
}


# ---------------------------------------------------------------------------
# sim_save_state / sim_load_state
# ---------------------------------------------------------------------------

def _sim_states_dir() -> Path:
    """Return the directory used to persist simulation states."""
    return get_hermes_home() / "sim_states"


def _sim_save_state_handler(args: Dict[str, Any], **kwargs) -> str:
    """Serialize and save a simulation state to disk."""
    state = args.get("state")
    if state is None or not isinstance(state, dict):
        return tool_error("state must be a dict", success=False)

    try:
        payload_json = json.dumps(state, sort_keys=True, ensure_ascii=False)
        state_id = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

        states_dir = _sim_states_dir()
        states_dir.mkdir(parents=True, exist_ok=True)

        state_path = states_dir / f"{state_id}.json"
        state_path.write_text(payload_json, encoding="utf-8")

        return json.dumps(
            {
                "state_id": state_id,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "success": True,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("sim_save_state failed: %s", e)
        return tool_error(str(e), success=False)


def _sim_load_state_handler(args: Dict[str, Any], **kwargs) -> str:
    """Load a previously saved simulation state from disk."""
    state_id = args.get("state_id")
    if not state_id or not isinstance(state_id, str):
        return tool_error("state_id is required and must be a string", success=False)

    # Sanitize state_id to prevent directory traversal
    safe_id = state_id.strip()
    if "/" in safe_id or "\\" in safe_id or safe_id.startswith("."):
        return tool_error("Invalid state_id format", success=False)

    try:
        states_dir = _sim_states_dir()
        state_path = states_dir / f"{safe_id}.json"

        if not state_path.exists():
            return json.dumps(
                {"error": f"State '{safe_id}' not found.", "success": False},
                ensure_ascii=False,
            )

        raw = state_path.read_text(encoding="utf-8")
        state = json.loads(raw)

        return json.dumps({"state": state, "success": True}, ensure_ascii=False)
    except Exception as e:
        logger.exception("sim_load_state failed: %s", e)
        return tool_error(str(e), success=False)


SIM_SAVE_STATE_SCHEMA = {
    "name": "sim_save_state",
    "description": "Serialize and save a simulation state to disk. Returns a SHA-256 state_id that can be used to reload the state later.",
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "object",
                "description": "Arbitrary JSON-serializable simulation state dict to persist",
            },
        },
        "required": ["state"],
    },
}

SIM_LOAD_STATE_SCHEMA = {
    "name": "sim_load_state",
    "description": "Load a previously saved simulation state from disk by its state_id.",
    "parameters": {
        "type": "object",
        "properties": {
            "state_id": {
                "type": "string",
                "description": "SHA-256 state_id returned by sim_save_state",
            },
        },
        "required": ["state_id"],
    },
}


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_simulation_requirements() -> bool:
    """Simulation tools are always available when numpy is installed."""
    try:
        import numpy as np  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

registry.register(
    name="sim_run",
    toolset="simulation",
    schema=SIM_RUN_SCHEMA,
    handler=_sim_run_handler,
    check_fn=_check_simulation_requirements,
    description="Run a deterministic simulation (Monte Carlo or discrete event)",
    emoji="🎲",
)

registry.register(
    name="sim_monte_carlo_option",
    toolset="simulation",
    schema=SIM_MONTE_CARLO_OPTION_SCHEMA,
    handler=_sim_monte_carlo_option_handler,
    check_fn=_check_simulation_requirements,
    description="Price a European option via Monte Carlo simulation",
    emoji="📈",
)

registry.register(
    name="sim_save_state",
    toolset="simulation",
    schema=SIM_SAVE_STATE_SCHEMA,
    handler=_sim_save_state_handler,
    check_fn=lambda: True,
    description="Serialize and save a simulation state to disk",
    emoji="💾",
)

registry.register(
    name="sim_load_state",
    toolset="simulation",
    schema=SIM_LOAD_STATE_SCHEMA,
    handler=_sim_load_state_handler,
    check_fn=lambda: True,
    description="Load a previously saved simulation state from disk",
    emoji="📂",
)
