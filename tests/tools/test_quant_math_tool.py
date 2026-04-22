"""Tests for tools/quant_math_tool.py — quantitative finance math handlers."""

import json
import math
import pytest

from tools.quant_math_tool import (
    _black_scholes_handler,
    _var_handler,
    _sharpe_ratio_handler,
    _portfolio_optimize_handler,
    _correlation_matrix_handler,
    _drawdown_handler,
)

try:
    import numpy as np  # noqa: F401
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False


# =========================================================================
# quant_black_scholes
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
class TestBlackScholes:
    def test_call_price_basic(self):
        args = {
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "option_type": "call",
        }
        result = json.loads(_black_scholes_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert result["price"] > 0
        assert 0 < result["delta"] <= 1
        assert result["gamma"] > 0
        assert result["vega"] > 0

    def test_put_price_basic(self):
        args = {
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "option_type": "put",
        }
        result = json.loads(_black_scholes_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert result["price"] > 0
        assert -1 <= result["delta"] < 0

    def test_put_call_parity(self):
        """C - P = S - K * e^(-rT)"""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        call = json.loads(_black_scholes_handler({
            "spot": S, "strike": K, "maturity": T,
            "risk_free_rate": r, "volatility": sigma, "option_type": "call",
        }))
        put = json.loads(_black_scholes_handler({
            "spot": S, "strike": K, "maturity": T,
            "risk_free_rate": r, "volatility": sigma, "option_type": "put",
        }))
        lhs = call["price"] - put["price"]
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 0.01

    def test_invalid_option_type(self):
        args = {
            "spot": 100, "strike": 100, "maturity": 1,
            "risk_free_rate": 0.05, "volatility": 0.2, "option_type": "straddle",
        }
        result = json.loads(_black_scholes_handler(args))
        assert "error" in result

    def test_missing_param(self):
        result = json.loads(_black_scholes_handler({"spot": 100}))
        assert "error" in result

    def test_negative_volatility(self):
        args = {
            "spot": 100, "strike": 100, "maturity": 1,
            "risk_free_rate": 0.05, "volatility": -0.1, "option_type": "call",
        }
        result = json.loads(_black_scholes_handler(args))
        assert "error" in result


# =========================================================================
# quant_var
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
class TestVaR:
    def test_historical_var(self):
        returns = [-0.02, -0.01, 0.005, 0.01, 0.015, -0.005, 0.02, -0.03, 0.01, 0.005]
        args = {"returns": returns, "confidence": 0.95, "method": "historical"}
        result = json.loads(_var_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert result["method"] == "historical"
        assert result["confidence"] == 0.95
        assert isinstance(result["var"], float)

    def test_parametric_var(self):
        returns = [-0.02, -0.01, 0.005, 0.01, 0.015, -0.005, 0.02, -0.03, 0.01, 0.005]
        args = {"returns": returns, "confidence": 0.95, "method": "parametric"}
        result = json.loads(_var_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert result["method"] == "parametric"

    def test_empty_returns(self):
        args = {"returns": []}
        result = json.loads(_var_handler(args))
        assert "error" in result

    def test_invalid_method(self):
        args = {"returns": [0.01, -0.01], "method": "monte_carlo"}
        result = json.loads(_var_handler(args))
        assert "error" in result

    def test_invalid_confidence(self):
        args = {"returns": [0.01, -0.01], "confidence": 1.5}
        result = json.loads(_var_handler(args))
        assert "error" in result


# =========================================================================
# quant_sharpe_ratio
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
class TestSharpeRatio:
    def test_basic_sharpe(self):
        returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.005, 0.008, -0.002, 0.012, 0.003]
        args = {"returns": returns, "risk_free_rate": 0.02, "periods_per_year": 252}
        result = json.loads(_sharpe_ratio_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert isinstance(result["sharpe_ratio"], float)
        assert isinstance(result["annualized_return"], float)
        assert isinstance(result["annualized_volatility"], float)

    def test_zero_std_error(self):
        args = {"returns": [0.01, 0.01, 0.01], "risk_free_rate": 0.02}
        result = json.loads(_sharpe_ratio_handler(args))
        assert "error" in result

    def test_single_return_error(self):
        args = {"returns": [0.01]}
        result = json.loads(_sharpe_ratio_handler(args))
        assert "error" in result


# =========================================================================
# quant_portfolio_optimize
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
class TestPortfolioOptimize:
    def test_two_asset_optimization(self):
        args = {
            "expected_returns": [0.10, 0.15],
            "covariance_matrix": [
                [0.04, 0.01],
                [0.01, 0.09],
            ],
            "risk_free_rate": 0.02,
        }
        result = json.loads(_portfolio_optimize_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert len(result["optimal_weights"]) == 2
        assert abs(sum(result["optimal_weights"]) - 1.0) < 1e-6
        assert result["expected_return"] > 0
        assert result["expected_volatility"] >= 0
        assert isinstance(result["sharpe_ratio"], float)
        assert len(result["efficient_frontier_points"]) > 0

    def test_three_asset_with_target(self):
        args = {
            "expected_returns": [0.08, 0.12, 0.10],
            "covariance_matrix": [
                [0.04, 0.01, 0.005],
                [0.01, 0.09, 0.01],
                [0.005, 0.01, 0.06],
            ],
            "target_return": 0.10,
            "risk_free_rate": 0.02,
        }
        result = json.loads(_portfolio_optimize_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert len(result["optimal_weights"]) == 3
        assert abs(sum(result["optimal_weights"]) - 1.0) < 1e-6

    def test_covariance_shape_mismatch(self):
        args = {
            "expected_returns": [0.10, 0.15],
            "covariance_matrix": [[0.04]],
        }
        result = json.loads(_portfolio_optimize_handler(args))
        assert "error" in result

    def test_empty_returns(self):
        args = {"expected_returns": [], "covariance_matrix": []}
        result = json.loads(_portfolio_optimize_handler(args))
        assert "error" in result


# =========================================================================
# quant_correlation_matrix
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
class TestCorrelationMatrix:
    def test_basic_correlation(self):
        args = {
            "series": {
                "AAPL": [150.0, 152.0, 151.0, 153.0, 155.0],
                "MSFT": [300.0, 302.0, 301.0, 304.0, 306.0],
            }
        }
        result = json.loads(_correlation_matrix_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert "AAPL" in result["correlation_matrix"]
        assert "MSFT" in result["correlation_matrix"]
        # Diagonal should be ~1.0
        assert abs(result["correlation_matrix"]["AAPL"]["AAPL"] - 1.0) < 1e-6

    def test_single_asset_error(self):
        args = {"series": {"AAPL": [150, 151, 152]}}
        result = json.loads(_correlation_matrix_handler(args))
        assert "error" in result

    def test_mismatched_lengths(self):
        args = {
            "series": {
                "A": [1, 2, 3],
                "B": [1, 2],
            }
        }
        result = json.loads(_correlation_matrix_handler(args))
        assert "error" in result


# =========================================================================
# quant_drawdown
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
class TestDrawdown:
    def test_basic_drawdown(self):
        prices = [100.0, 110.0, 105.0, 90.0, 95.0, 80.0, 85.0]
        args = {"prices": prices}
        result = json.loads(_drawdown_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert result["max_drawdown"] < 0
        assert result["max_drawdown_duration"] >= 0
        assert result["peak_index"] < result["trough_index"]
        assert len(result["drawdown_series"]) == len(prices)

    def test_no_drawdown(self):
        prices = [100.0, 105.0, 110.0, 115.0, 120.0]
        args = {"prices": prices}
        result = json.loads(_drawdown_handler(args))
        assert "error" not in result
        assert result["max_drawdown"] == 0.0 or result["max_drawdown"] > -1e-9

    def test_empty_prices(self):
        args = {"prices": []}
        result = json.loads(_drawdown_handler(args))
        assert "error" in result

    def test_zero_first_price(self):
        args = {"prices": [0, 100, 90]}
        result = json.loads(_drawdown_handler(args))
        assert "error" in result
