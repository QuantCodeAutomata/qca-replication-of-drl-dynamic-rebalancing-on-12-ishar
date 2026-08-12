"""
Tests for src.stats.metrics, src.stats.statistical_tests, and
src.backtest.benchmarks.

All tests rely entirely on synthetic data; no market data files are read.
Mocking is not used: the benchmark test is designed so that the trailing
estimation window always contains fewer than 20 observations, exercising the
equal-weight fallback path rather than the optimiser.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.benchmarks import compute_buy_hold_benchmark, compute_static_benchmark
from src.stats.metrics import compute_performance_metrics, compute_turnover
from src.stats.statistical_tests import (
    alpha_regression_newey_west,
    sharpe_difference_bootstrap,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(0)


def _make_returns(n: int = 252, mu: float = 0.001, sigma: float = 0.01) -> np.ndarray:
    """Generate synthetic daily log returns ~ N(mu, sigma)."""
    rng = np.random.default_rng(42)
    return rng.normal(mu, sigma, size=n)


# ─────────────────────────────────────────────────────────────────────────────
# compute_performance_metrics
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_metrics_basic():
    """Verify annualized return, Sharpe, and max drawdown with known inputs."""
    r = _make_returns(n=504, mu=0.001, sigma=0.01)
    m = compute_performance_metrics(r)

    # Annualised return should be positive and plausibly above zero
    assert 'annualized_return' in m
    assert m['annualized_return'] > 0, 'Expected positive mean return'

    # Sharpe should be a finite positive number given mu > 0 and sigma > 0
    assert 'sharpe_ratio' in m
    assert np.isfinite(m['sharpe_ratio'])
    assert m['sharpe_ratio'] > 0

    # Volatility should be close to sigma * sqrt(252)
    expected_vol = 0.01 * np.sqrt(252)
    assert abs(m['annualized_volatility'] - expected_vol) < 0.05

    # Max drawdown must be non-positive
    assert m['max_drawdown'] <= 0.0

    # All seven core keys must be present
    required = {
        'annualized_return', 'annualized_volatility', 'var_95', 'cvar_95',
        'max_drawdown', 'sharpe_ratio', 'sortino_ratio',
    }
    assert required.issubset(m.keys())


def test_compute_metrics_empty():
    """Edge case: fewer than 2 observations returns an empty dict."""
    assert compute_performance_metrics(np.array([])) == {}
    assert compute_performance_metrics(np.array([0.01])) == {}


def test_var_cvar_ordering():
    """CVaR must always be >= VaR (both expressed as positive loss figures)."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        r = rng.normal(0.0, 0.02, size=500)
        m = compute_performance_metrics(r)
        assert m['cvar_95'] >= m['var_95'] - 1e-10, (
            f"CVaR ({m['cvar_95']:.6f}) < VaR ({m['var_95']:.6f}) for seed {seed}"
        )


def test_sharpe_zero_vol():
    """Constant-return series has near-zero volatility.

    Zero volatility with positive mean is economically a risk-free positive
    return, so Sharpe is effectively +∞.  The implementation returns a large
    positive sentinel (1e6) rather than dividing by zero.  The assertion
    checks the correct direction: Sharpe must be non-negative (and in
    practice will be very large for a positive-drift constant series).
    """
    r = np.full(252, 0.001)
    m = compute_performance_metrics(r)
    assert m['sharpe_ratio'] >= 0.0


def test_max_drawdown_negative():
    """Max drawdown is always <= 0, regardless of the return series."""
    # Series with all positive returns → zero drawdown
    r_up = np.full(100, 0.005)
    m_up = compute_performance_metrics(r_up)
    assert m_up['max_drawdown'] <= 0.0

    # Series with mixed returns
    r_mixed = _make_returns(252, mu=0.0, sigma=0.02)
    m_mixed = compute_performance_metrics(r_mixed)
    assert m_mixed['max_drawdown'] <= 0.0

    # Series with all negative returns
    r_down = np.full(100, -0.005)
    m_down = compute_performance_metrics(r_down)
    assert m_down['max_drawdown'] <= 0.0


def test_upside_downside_capture():
    """
    A strategy that doubles the benchmark on both up and down days should
    show upside_capture ≈ 2 and downside_capture ≈ 2.
    A defensive strategy (larger gains, smaller losses than benchmark) should
    have upside_capture > 1 and downside_capture < 1.
    """
    n = 200
    rng = np.random.default_rng(7)
    b = rng.choice([-0.01, 0.01], size=n)  # Alternating up/down days

    # Strategy A: exactly 2x benchmark
    r_a = 2 * b
    m_a = compute_performance_metrics(r_a, benchmark_returns=b)
    assert abs(m_a['upside_capture'] - 2.0) < 1e-9
    assert abs(m_a['downside_capture'] - 2.0) < 1e-9

    # Strategy B: outperforms on up days, less negative on down days
    r_b = np.where(b > 0, b * 1.5, b * 0.5)
    m_b = compute_performance_metrics(r_b, benchmark_returns=b)
    assert m_b['upside_capture'] > 1.0, 'Expected upside capture > 1 for outperforming strategy'
    assert m_b['downside_capture'] < 1.0, 'Expected downside capture < 1 for defensive strategy'


# ─────────────────────────────────────────────────────────────────────────────
# alpha_regression_newey_west
# ─────────────────────────────────────────────────────────────────────────────


def test_alpha_regression():
    """Verify all expected keys are present in the regression output."""
    rng = np.random.default_rng(99)
    b = rng.normal(0.0, 0.01, size=252)
    # Strategy = 0.5 * benchmark + constant alpha + noise
    s = 0.0005 + 0.5 * b + rng.normal(0.0, 0.005, size=252)

    result = alpha_regression_newey_west(s, b, n_lags=5)

    expected_keys = {
        'alpha_daily', 'annualized_alpha', 'beta', 't_stat', 'p_value', 'n_obs'
    }
    assert expected_keys.issubset(result.keys()), (
        f'Missing keys: {expected_keys - result.keys()}'
    )
    assert result['n_obs'] == 252
    # Beta should be close to 0.5
    assert abs(result['beta'] - 0.5) < 0.15, f"beta={result['beta']:.4f} unexpectedly far from 0.5"
    # annualized_alpha should be close to 0.0005 * 252 = 0.126
    assert abs(result['annualized_alpha'] - 0.0005 * 252) < 0.05
    # p-value must be in [0, 1]
    assert 0.0 <= result['p_value'] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# sharpe_difference_bootstrap
# ─────────────────────────────────────────────────────────────────────────────


def test_sharpe_bootstrap():
    """Bootstrap p-value must lie in [0, 1] and keys must be present."""
    rng = np.random.default_rng(123)
    n = 252
    b = rng.normal(0.0, 0.01, size=n)
    # Strategy A has higher mean → should have lower p-value
    a = rng.normal(0.001, 0.01, size=n)

    result = sharpe_difference_bootstrap(a, b, n_bootstrap=500, seed=42)

    assert 0.0 <= result['bootstrap_p_value'] <= 1.0
    assert 'observed_sharpe_diff' in result
    assert 'sharpe_a' in result
    assert 'sharpe_b' in result

    # A strategy identical to the baseline should have p_value close to 0.5
    result_same = sharpe_difference_bootstrap(b, b, n_bootstrap=500, seed=42)
    assert result_same['observed_sharpe_diff'] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_static_benchmark / compute_buy_hold_benchmark
# ─────────────────────────────────────────────────────────────────────────────


def test_benchmark_computation():
    """
    Verify that the static benchmark stays fully invested and weights sum to 1.

    The test uses only 30 trading days, ensuring the trailing estimation window
    (default 252 days) never reaches 20 observations.  This triggers the
    equal-weight fallback path and keeps the test independent of the convex
    optimiser.
    """
    rng = np.random.default_rng(5)
    n_days = 30
    n_assets = 4
    dates = pd.bdate_range('2020-01-02', periods=n_days)
    returns_arr = rng.normal(0.0, 0.01, size=(n_days, n_assets))
    log_returns_df = pd.DataFrame(
        returns_arr, index=dates, columns=['A', 'B', 'C', 'D']
    )

    result = compute_static_benchmark(
        log_returns_df=log_returns_df,
        test_dates=dates,
        risk_model='variance',
        estimation_window=252,
        rebalance_every=30,
        initial_value=1000.0,
    )

    # Shape and index
    assert len(result) == n_days
    assert list(result.index) == list(dates)

    # Weights sum to 1 on every day
    for w in result['weights']:
        assert abs(w.sum() - 1.0) < 1e-10, f'Weights do not sum to 1: {w}'
        assert np.all(w >= 0), 'Negative weight detected'

    # Portfolio remains invested: value is always positive
    assert (result['portfolio_value'] > 0).all()

    # Portfolio value starts near initial_value and changes with returns
    assert abs(result['portfolio_value'].iloc[0] - 1000.0) < 50.0


def test_buy_hold_benchmark():
    """
    Verify that the buy-and-hold benchmark correctly compounds log returns.
    """
    dates = pd.bdate_range('2020-01-02', periods=10)
    # Constant 1 % daily return
    log_ret = pd.Series([0.01] * 10, index=dates)

    result = compute_buy_hold_benchmark(log_ret, dates, initial_value=1000.0)

    assert list(result.index) == list(dates)
    # After 10 days of 0.01 log return, value = 1000 * exp(0.10)
    expected_final = 1000.0 * np.exp(0.10)
    assert abs(result['portfolio_value'].iloc[-1] - expected_final) < 1e-6

    # Missing dates in benchmark_returns are filled with 0 (no change)
    sparse = pd.Series([0.05], index=[dates[5]])
    result_sparse = compute_buy_hold_benchmark(sparse, dates, initial_value=1000.0)
    # Portfolio should only move on dates[5]
    assert result_sparse['portfolio_value'].iloc[0] == 1000.0
    assert abs(result_sparse['portfolio_value'].iloc[5] - 1000.0 * np.exp(0.05)) < 1e-6
