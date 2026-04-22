#!/usr/bin/env python3
"""Quantitative Math Tool — financial mathematics for trading and portfolio analysis.

Registered tools:
- ``quant_black_scholes``     — option pricing + Greeks
- ``quant_var``               — Value at Risk
- ``quant_sharpe_ratio``      — Sharpe ratio
- ``quant_portfolio_optimize`` — mean-variance optimization
- ``quant_correlation_matrix`` — correlation matrix from series
- ``quant_drawdown``          — maximum drawdown analysis
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    try:
        from scipy.stats import norm
        return float(norm.cdf(x))
    except Exception:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    try:
        from scipy.stats import norm
        return float(norm.pdf(x))
    except Exception:
        return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _z_score(confidence: float) -> float:
    """Return the z-score for a two-tailed confidence level."""
    try:
        from scipy.stats import norm
        return float(norm.ppf(confidence))
    except Exception:
        # Common confidence levels
        lookup = {
            0.90: 1.2816,
            0.95: 1.6449,
            0.99: 2.3263,
        }
        return lookup.get(confidence, 1.6449)


def _to_numpy_floats(values: List[float]) -> Any:
    import numpy as np
    return np.asarray(values, dtype=float)


# ---------------------------------------------------------------------------
# quant_black_scholes
# ---------------------------------------------------------------------------


def _black_scholes_handler(args: Dict[str, Any], **kwargs) -> str:
    """Calculate Black-Scholes option price and Greeks."""
    try:
        spot = float(args["spot"])
        strike = float(args["strike"])
        maturity = float(args["maturity"])
        risk_free_rate = float(args["risk_free_rate"])
        volatility = float(args["volatility"])
        option_type = (args.get("option_type") or "call").lower().strip()
    except (KeyError, TypeError, ValueError) as e:
        return tool_error(f"Invalid or missing parameter: {e}", success=False)

    if option_type not in ("call", "put"):
        return tool_error("option_type must be 'call' or 'put'", success=False)

    if spot <= 0 or strike <= 0 or maturity <= 0 or volatility <= 0:
        return tool_error("spot, strike, maturity, and volatility must be positive", success=False)

    S = spot
    K = strike
    T = maturity
    r = risk_free_rate
    sigma = volatility

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    n_d1 = _norm_cdf(-d1)
    n_d2 = _norm_cdf(-d2)
    phi_d1 = _norm_pdf(d1)

    if option_type == "call":
        price = S * nd1 - K * math.exp(-r * T) * nd2
        delta = nd1
    else:
        price = K * math.exp(-r * T) * n_d2 - S * n_d1
        delta = -n_d1

    gamma = phi_d1 / (S * sigma * math.sqrt(T))
    theta = -(S * phi_d1 * sigma) / (2 * math.sqrt(T))
    if option_type == "call":
        theta -= r * K * math.exp(-r * T) * nd2
    else:
        theta += r * K * math.exp(-r * T) * n_d2
    theta = theta / 365.0  # annual to daily

    vega = S * phi_d1 * math.sqrt(T) / 100.0  # per 1% change
    rho = K * T * math.exp(-r * T) * (nd2 if option_type == "call" else -n_d2) / 100.0

    return tool_result(
        success=True,
        price=round(price, 6),
        delta=round(delta, 6),
        gamma=round(gamma, 6),
        theta=round(theta, 6),
        vega=round(vega, 6),
        rho=round(rho, 6),
        inputs={
            "spot": spot,
            "strike": strike,
            "maturity": maturity,
            "risk_free_rate": risk_free_rate,
            "volatility": volatility,
            "option_type": option_type,
        },
    )


registry.register(
    name="quant_black_scholes",
    toolset="quant_math",
    schema={
        "type": "function",
        "function": {
            "name": "quant_black_scholes",
            "description": "Calculate Black-Scholes option price and Greeks (delta, gamma, theta, vega, rho).",
            "parameters": {
                "type": "object",
                "properties": {
                    "spot": {"type": "number", "description": "Current spot price of the underlying"},
                    "strike": {"type": "number", "description": "Strike price of the option"},
                    "maturity": {"type": "number", "description": "Time to maturity in years"},
                    "risk_free_rate": {"type": "number", "description": "Annual risk-free interest rate (e.g. 0.05 for 5%)"},
                    "volatility": {"type": "number", "description": "Annual volatility (e.g. 0.2 for 20%)"},
                    "option_type": {"type": "string", "enum": ["call", "put"], "description": "Option type"},
                },
                "required": ["spot", "strike", "maturity", "risk_free_rate", "volatility", "option_type"],
            },
        },
    },
    handler=_black_scholes_handler,
    check_fn=lambda: True,
    description="Black-Scholes option pricing and Greeks",
    emoji="📈",
)


# ---------------------------------------------------------------------------
# quant_var
# ---------------------------------------------------------------------------


def _var_handler(args: Dict[str, Any], **kwargs) -> str:
    """Calculate Value at Risk."""
    try:
        returns = list(args["returns"])
        confidence = float(args.get("confidence", 0.95))
        method = (args.get("method") or "historical").lower().strip()
    except (KeyError, TypeError, ValueError) as e:
        return tool_error(f"Invalid or missing parameter: {e}", success=False)

    if len(returns) == 0:
        return tool_error("returns must not be empty", success=False)

    if method not in ("historical", "parametric"):
        return tool_error("method must be 'historical' or 'parametric'", success=False)

    if not 0 < confidence < 1:
        return tool_error("confidence must be between 0 and 1", success=False)

    import numpy as np

    arr = np.asarray(returns, dtype=float)

    if method == "historical":
        var = float(np.percentile(arr, (1 - confidence) * 100))
    else:
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        z = _z_score(confidence)
        var = mean - z * std

    return tool_result(
        success=True,
        var=round(var, 6),
        method=method,
        confidence=confidence,
        num_returns=len(returns),
    )


registry.register(
    name="quant_var",
    toolset="quant_math",
    schema={
        "type": "function",
        "function": {
            "name": "quant_var",
            "description": "Calculate Value at Risk (VaR) from a list of returns using historical or parametric method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "returns": {"type": "array", "items": {"type": "number"}, "description": "List of historical returns"},
                    "confidence": {"type": "number", "default": 0.95, "description": "Confidence level (e.g. 0.95 for 95%)"},
                    "method": {"type": "string", "enum": ["historical", "parametric"], "default": "historical", "description": "Calculation method"},
                },
                "required": ["returns"],
            },
        },
    },
    handler=_var_handler,
    check_fn=lambda: True,
    description="Value at Risk calculation",
    emoji="📉",
)


# ---------------------------------------------------------------------------
# quant_sharpe_ratio
# ---------------------------------------------------------------------------


def _sharpe_ratio_handler(args: Dict[str, Any], **kwargs) -> str:
    """Calculate Sharpe ratio and annualized metrics."""
    try:
        returns = list(args["returns"])
        risk_free_rate = float(args.get("risk_free_rate", 0.02))
        periods_per_year = int(args.get("periods_per_year", 252))
    except (KeyError, TypeError, ValueError) as e:
        return tool_error(f"Invalid or missing parameter: {e}", success=False)

    if len(returns) < 2:
        return tool_error("returns must contain at least 2 values", success=False)

    import numpy as np

    arr = np.asarray(returns, dtype=float)
    mean_return = float(np.mean(arr))
    std_return = float(np.std(arr))

    if std_return == 0:
        return tool_error("returns have zero standard deviation", success=False)

    sharpe = (mean_return - risk_free_rate / periods_per_year) / std_return
    annualized_return = mean_return * periods_per_year
    annualized_volatility = std_return * math.sqrt(periods_per_year)
    sharpe_annualized = (annualized_return - risk_free_rate) / annualized_volatility

    return tool_result(
        success=True,
        sharpe_ratio=round(sharpe, 6),
        sharpe_ratio_annualized=round(sharpe_annualized, 6),
        annualized_return=round(annualized_return, 6),
        annualized_volatility=round(annualized_volatility, 6),
        mean_return=round(mean_return, 6),
        std_return=round(std_return, 6),
        num_periods=len(returns),
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )


registry.register(
    name="quant_sharpe_ratio",
    toolset="quant_math",
    schema={
        "type": "function",
        "function": {
            "name": "quant_sharpe_ratio",
            "description": "Calculate Sharpe ratio and annualized return/volatility from a series of returns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "returns": {"type": "array", "items": {"type": "number"}, "description": "List of periodic returns"},
                    "risk_free_rate": {"type": "number", "default": 0.02, "description": "Annual risk-free rate"},
                    "periods_per_year": {"type": "integer", "default": 252, "description": "Number of periods per year (252 for daily, 12 for monthly)"},
                },
                "required": ["returns"],
            },
        },
    },
    handler=_sharpe_ratio_handler,
    check_fn=lambda: True,
    description="Sharpe ratio and annualized metrics",
    emoji="⚖️",
)


# ---------------------------------------------------------------------------
# quant_portfolio_optimize
# ---------------------------------------------------------------------------


def _portfolio_optimize_handler(args: Dict[str, Any], **kwargs) -> str:
    """Mean-variance portfolio optimization."""
    try:
        expected_returns = list(args["expected_returns"])
        covariance_matrix = list(args["covariance_matrix"])
        target_return = args.get("target_return")
        risk_free_rate = float(args.get("risk_free_rate", 0.02))
    except (KeyError, TypeError, ValueError) as e:
        return tool_error(f"Invalid or missing parameter: {e}", success=False)

    n = len(expected_returns)
    if n == 0:
        return tool_error("expected_returns must not be empty", success=False)

    import numpy as np

    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance_matrix, dtype=float)

    if cov.shape != (n, n):
        return tool_error(f"covariance_matrix must be {n}x{n}, got {cov.shape}", success=False)

    # Try scipy.optimize first; fall back to simple closed form for 2 assets
    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    def _portfolio_metrics(w: np.ndarray) -> tuple:
        ret = float(np.dot(w, mu))
        vol = float(math.sqrt(np.dot(w, np.dot(cov, w))))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n)]

    if target_return is not None:
        target = float(target_return)
        constraints.append({"type": "eq", "fun": lambda w: np.dot(w, mu) - target})

    if minimize is not None:
        # Maximize Sharpe → minimize negative Sharpe
        def neg_sharpe(w: np.ndarray) -> float:
            _, vol, sharpe = _portfolio_metrics(w)
            return -sharpe

        x0 = np.ones(n) / n
        result = minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        optimal_weights = result.x.tolist()
    else:
        # Fallback: equal weight if scipy unavailable
        optimal_weights = (np.ones(n) / n).tolist()

    opt_ret, opt_vol, opt_sharpe = _portfolio_metrics(np.asarray(optimal_weights))

    # Efficient frontier: sample portfolios along the frontier
    frontier_points = []
    if minimize is not None and n <= 5:
        target_returns = np.linspace(mu.min(), mu.max(), 20)
        for tr in target_returns:
            c = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w: np.dot(w, mu) - tr},
            ]
            res = minimize(
                lambda w: float(np.dot(w, np.dot(cov, w))),
                np.ones(n) / n,
                method="SLSQP",
                bounds=bounds,
                constraints=c,
                options={"maxiter": 1000},
            )
            if res.success:
                w = res.x.tolist()
                r, v, s = _portfolio_metrics(np.asarray(w))
                frontier_points.append({
                    "target_return": round(float(tr), 6),
                    "expected_return": round(r, 6),
                    "volatility": round(v, 6),
                    "weights": [round(x, 4) for x in w],
                })

    return tool_result(
        success=True,
        optimal_weights=[round(x, 6) for x in optimal_weights],
        expected_return=round(opt_ret, 6),
        expected_volatility=round(opt_vol, 6),
        sharpe_ratio=round(opt_sharpe, 6),
        efficient_frontier_points=frontier_points,
        risk_free_rate=risk_free_rate,
        num_assets=n,
        solver=("scipy" if minimize is not None else "fallback_equal_weight"),
    )


registry.register(
    name="quant_portfolio_optimize",
    toolset="quant_math",
    schema={
        "type": "function",
        "function": {
            "name": "quant_portfolio_optimize",
            "description": "Mean-variance portfolio optimization. Returns optimal weights, expected return, volatility, and Sharpe ratio. Uses scipy.optimize when available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expected_returns": {"type": "array", "items": {"type": "number"}, "description": "Expected returns for each asset"},
                    "covariance_matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Covariance matrix (n x n)"},
                    "target_return": {"type": "number", "description": "Optional target return constraint"},
                    "risk_free_rate": {"type": "number", "default": 0.02, "description": "Risk-free rate for Sharpe calculation"},
                },
                "required": ["expected_returns", "covariance_matrix"],
            },
        },
    },
    handler=_portfolio_optimize_handler,
    check_fn=lambda: True,
    description="Mean-variance portfolio optimization",
    emoji="🎯",
)


# ---------------------------------------------------------------------------
# quant_correlation_matrix
# ---------------------------------------------------------------------------


def _correlation_matrix_handler(args: Dict[str, Any], **kwargs) -> str:
    """Calculate correlation matrix from price/return series."""
    try:
        series = dict(args["series"])
    except (KeyError, TypeError, ValueError) as e:
        return tool_error(f"Invalid or missing parameter: {e}", success=False)

    if len(series) < 2:
        return tool_error("series must contain at least 2 assets", success=False)

    import numpy as np

    # Ensure all series have the same length
    lengths = {len(v) for v in series.values()}
    if len(lengths) != 1:
        return tool_error(f"All series must have the same length; found lengths {lengths}", success=False)

    data = np.column_stack([np.asarray(v, dtype=float) for v in series.values()])
    corr = np.corrcoef(data, rowvar=False)

    assets = list(series.keys())
    n = len(assets)

    # Build human-readable matrix dict
    matrix_dict = {}
    for i, a in enumerate(assets):
        matrix_dict[a] = {}
        for j, b in enumerate(assets):
            matrix_dict[a][b] = round(float(corr[i, j]), 6)

    return tool_result(
        success=True,
        correlation_matrix=matrix_dict,
        assets=assets,
        num_observations=data.shape[0],
    )


registry.register(
    name="quant_correlation_matrix",
    toolset="quant_math",
    schema={
        "type": "function",
        "function": {
            "name": "quant_correlation_matrix",
            "description": "Calculate correlation matrix from a dict of asset price or return series.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series": {
                        "type": "object",
                        "description": "Dict mapping asset name to list of float values",
                    },
                },
                "required": ["series"],
            },
        },
    },
    handler=_correlation_matrix_handler,
    check_fn=lambda: True,
    description="Correlation matrix from asset series",
    emoji="🔗",
)


# ---------------------------------------------------------------------------
# quant_drawdown
# ---------------------------------------------------------------------------


def _drawdown_handler(args: Dict[str, Any], **kwargs) -> str:
    """Calculate maximum drawdown and drawdown series."""
    try:
        prices = list(args["prices"])
    except (KeyError, TypeError, ValueError) as e:
        return tool_error(f"Invalid or missing parameter: {e}", success=False)

    if len(prices) == 0:
        return tool_error("prices must not be empty", success=False)

    import numpy as np

    arr = np.asarray(prices, dtype=float)
    if arr[0] == 0:
        return tool_error("First price cannot be zero", success=False)

    running_max = np.maximum.accumulate(arr)
    drawdowns = (arr - running_max) / running_max
    max_dd_idx = int(np.argmin(drawdowns))
    max_dd = float(drawdowns[max_dd_idx])

    # Find peak before trough
    peak_idx = 0
    for i in range(max_dd_idx, -1, -1):
        if arr[i] == running_max[max_dd_idx]:
            peak_idx = i
            break

    # Duration = trough index - peak index
    duration = max_dd_idx - peak_idx

    dd_series = [round(float(v), 6) for v in drawdowns.tolist()]

    return tool_result(
        success=True,
        max_drawdown=round(max_dd, 6),
        max_drawdown_duration=duration,
        peak_index=peak_idx,
        trough_index=max_dd_idx,
        drawdown_series=dd_series,
        num_prices=len(prices),
    )


registry.register(
    name="quant_drawdown",
    toolset="quant_math",
    schema={
        "type": "function",
        "function": {
            "name": "quant_drawdown",
            "description": "Calculate maximum drawdown and drawdown series from a price history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prices": {"type": "array", "items": {"type": "number"}, "description": "List of asset prices in chronological order"},
                },
                "required": ["prices"],
            },
        },
    },
    handler=_drawdown_handler,
    check_fn=lambda: True,
    description="Maximum drawdown analysis",
    emoji="📉",
)
