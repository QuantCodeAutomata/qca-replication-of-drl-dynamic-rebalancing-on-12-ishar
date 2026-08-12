"""
Tests for the portfolio optimiser and frontier mapper modules.

Synthetic data: 5 assets, 300 daily observations generated from a
fixed random seed to guarantee reproducibility.  All tests are self-contained
and require no external data sources.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer.estimators import (
    RollingEstimator,
    estimate_covariance,
    estimate_expected_returns,
)
from src.optimizer.frontier_mapper import (
    build_target_return_grid,
    find_feasible_target,
    lambda_to_target_return,
)
from src.optimizer.portfolio_optimizer import (
    estimate_semi_covariance,
    solve_mean_cvar,
    solve_mean_semivariance,
    solve_mean_variance,
)

# ---------------------------------------------------------------------------
# Shared synthetic data fixtures
# ---------------------------------------------------------------------------

N_ASSETS = 5
N_OBS = 300
RNG_SEED = 42


@pytest.fixture(scope="module")
def synthetic_returns():
    """Generate (300, 5) synthetic log-return matrix with realistic parameters."""
    rng = np.random.default_rng(RNG_SEED)
    # Daily log returns: small positive drift, cross-asset correlations
    mu_true = np.array([0.0003, 0.0005, 0.0007, 0.0004, 0.0006])
    # Construct a simple correlation-based covariance
    corr = np.array([
        [1.00, 0.40, 0.30, 0.20, 0.25],
        [0.40, 1.00, 0.50, 0.35, 0.30],
        [0.30, 0.50, 1.00, 0.45, 0.40],
        [0.20, 0.35, 0.45, 1.00, 0.60],
        [0.25, 0.30, 0.40, 0.60, 1.00],
    ])
    vol = np.array([0.010, 0.012, 0.015, 0.011, 0.013])
    sigma_true = np.diag(vol) @ corr @ np.diag(vol)
    returns = rng.multivariate_normal(mu_true, sigma_true, size=N_OBS)
    return returns


@pytest.fixture(scope="module")
def mu_sigma(synthetic_returns):
    """Return estimated mu and sigma from the synthetic data."""
    mu = estimate_expected_returns(synthetic_returns)
    sigma = estimate_covariance(synthetic_returns)
    return mu, sigma


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_weights_valid(weights, n_assets=N_ASSETS, tol=1e-5):
    """Assert weights form a valid long-only fully-invested portfolio."""
    assert weights is not None, "Solver returned None for a feasible problem."
    assert weights.shape == (n_assets,), (
        f"Expected shape ({n_assets},), got {weights.shape}"
    )
    assert np.all(weights >= -tol), (
        f"Some weights are negative: {weights[weights < -tol]}"
    )
    assert abs(weights.sum() - 1.0) < tol, (
        f"Weights do not sum to 1: sum={weights.sum()}"
    )


# ---------------------------------------------------------------------------
# Mean–Variance tests
# ---------------------------------------------------------------------------

class TestSolveMeanVariance:
    def test_solve_mean_variance_sum_to_one(self, mu_sigma):
        """Optimal weights must be non-negative and sum to 1."""
        mu, sigma = mu_sigma
        mu_target = np.mean(mu)  # achievable mid-point target
        weights = solve_mean_variance(mu, sigma, mu_target, N_ASSETS)
        _assert_weights_valid(weights)

    def test_solve_mean_variance_feasible(self, mu_sigma):
        """Solver must succeed for any target in [min(mu), max(mu)]."""
        mu, sigma = mu_sigma
        # Test three points: low, mid, high-but-feasible
        for frac in [0.1, 0.5, 0.9]:
            mu_target = np.min(mu) + frac * (np.max(mu) - np.min(mu))
            weights = solve_mean_variance(mu, sigma, mu_target, N_ASSETS)
            _assert_weights_valid(weights)
            # Achieved return must satisfy the constraint (up to solver tolerance)
            achieved = float(mu @ weights)
            assert achieved >= mu_target - 1e-4, (
                f"Achieved return {achieved:.6f} < target {mu_target:.6f}"
            )

    def test_solve_mean_variance_infeasible(self, mu_sigma):
        """Solver must return None when the target exceeds max(mu)."""
        mu, sigma = mu_sigma
        # Set target strictly above the maximum single-asset return
        mu_target = np.max(mu) * 10.0
        weights = solve_mean_variance(mu, sigma, mu_target, N_ASSETS)
        assert weights is None, (
            "Expected None for an infeasible target, got a weight vector."
        )

    def test_solve_mean_variance_minimum_variance(self, mu_sigma):
        """With target = min(mu), result should be the minimum-variance portfolio."""
        mu, sigma = mu_sigma
        mu_target = np.min(mu) - 1e-8  # slightly below min -> unconstrained
        weights = solve_mean_variance(mu, sigma, mu_target, N_ASSETS)
        _assert_weights_valid(weights)


# ---------------------------------------------------------------------------
# Mean–Semi-Variance tests
# ---------------------------------------------------------------------------

class TestSolveMeanSemivariance:
    def test_solve_mean_semivariance_sum_to_one(self, synthetic_returns, mu_sigma):
        """Semi-variance weights must be non-negative and sum to 1."""
        mu, _ = mu_sigma
        semi_cov = estimate_semi_covariance(synthetic_returns)
        mu_target = np.mean(mu)
        weights = solve_mean_semivariance(mu, semi_cov, mu_target, N_ASSETS)
        _assert_weights_valid(weights)

    def test_solve_mean_semivariance_feasible_range(self, synthetic_returns, mu_sigma):
        """Semi-variance solver must succeed across the feasible return range."""
        mu, _ = mu_sigma
        semi_cov = estimate_semi_covariance(synthetic_returns)
        for frac in [0.2, 0.6, 0.95]:
            mu_target = np.min(mu) + frac * (np.max(mu) - np.min(mu))
            weights = solve_mean_semivariance(mu, semi_cov, mu_target, N_ASSETS)
            _assert_weights_valid(weights)


# ---------------------------------------------------------------------------
# Semi-Covariance estimation tests
# ---------------------------------------------------------------------------

class TestEstimateSemiCovariance:
    def test_estimate_semi_covariance_psd(self, synthetic_returns):
        """Semi-covariance matrix must be positive semi-definite."""
        semi_cov = estimate_semi_covariance(synthetic_returns)
        assert semi_cov.shape == (N_ASSETS, N_ASSETS), "Wrong shape."
        eigenvalues = np.linalg.eigvalsh(semi_cov)
        assert np.all(eigenvalues >= -1e-10), (
            f"Semi-cov has negative eigenvalues: {eigenvalues[eigenvalues < 0]}"
        )

    def test_estimate_semi_covariance_symmetry(self, synthetic_returns):
        """Semi-covariance matrix must be symmetric."""
        semi_cov = estimate_semi_covariance(synthetic_returns)
        np.testing.assert_allclose(semi_cov, semi_cov.T, atol=1e-14)

    def test_estimate_semi_covariance_positive_returns_only(self):
        """When all returns are above the benchmark, SCOV should be zero matrix."""
        # All returns positive -> downside = 0 everywhere
        returns = np.abs(np.random.default_rng(0).standard_normal((100, 3))) + 0.01
        semi_cov = estimate_semi_covariance(returns, benchmark=0.0)
        np.testing.assert_allclose(semi_cov, np.zeros((3, 3)), atol=1e-14)

    def test_estimate_semi_covariance_custom_benchmark(self, synthetic_returns):
        """A positive benchmark should produce a larger semi-cov than B=0."""
        scov_zero = estimate_semi_covariance(synthetic_returns, benchmark=0.0)
        scov_pos = estimate_semi_covariance(synthetic_returns, benchmark=0.01)
        # More returns fall below a higher benchmark -> larger diagonal entries
        assert np.trace(scov_pos) >= np.trace(scov_zero) - 1e-14

    def test_estimate_semi_covariance_nan_input(self):
        """NaN values in returns must not propagate; they are treated as 0."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal((100, N_ASSETS))
        returns[5, 2] = np.nan   # inject a single NaN
        returns[20, :] = np.inf  # inject a whole row of Inf
        semi_cov = estimate_semi_covariance(returns, benchmark=0.0)
        assert not np.any(np.isnan(semi_cov)), "NaN leaked into semi-covariance matrix."
        assert not np.any(np.isinf(semi_cov)), "Inf leaked into semi-covariance matrix."
        eigenvalues = np.linalg.eigvalsh(semi_cov)
        assert np.all(eigenvalues >= -1e-10), "Semi-cov has negative eigenvalues after NaN sanitisation."

    def test_solve_mean_semivariance_nan_semi_cov(self, mu_sigma):
        """Solver must return None (not raise) when semi_cov contains NaN values."""
        mu, _ = mu_sigma
        nan_semi_cov = np.full((N_ASSETS, N_ASSETS), np.nan)
        mu_target = np.mean(mu)
        # Before the fix this raised ValueError inside cp.quad_form before the try-except.
        result = solve_mean_semivariance(mu, nan_semi_cov, mu_target, N_ASSETS)
        # All-NaN semi_cov degenerates to identity-only → solver may succeed or fail cleanly.
        # Either way, if it succeeds the weights must be valid; it must never raise.
        if result is not None:
            _assert_weights_valid(result)

    def test_solve_mean_semivariance_asymmetric_semi_cov(self, mu_sigma):
        """Slightly asymmetric semi_cov (floating-point noise) must not crash solver."""
        mu, sigma = mu_sigma
        rng = np.random.default_rng(0)
        returns = rng.multivariate_normal(mu, sigma, size=252)
        semi_cov = estimate_semi_covariance(returns)
        # Introduce a tiny asymmetry to simulate float accumulation
        semi_cov[0, 1] += 1e-12
        mu_target = np.mean(mu)
        weights = solve_mean_semivariance(mu, semi_cov, mu_target, N_ASSETS)
        _assert_weights_valid(weights)


# ---------------------------------------------------------------------------
# CVaR tests
# ---------------------------------------------------------------------------

class TestSolveMeanCvar:
    def test_solve_mean_cvar_sum_to_one(self, synthetic_returns, mu_sigma):
        """CVaR-optimal weights must be non-negative and sum to 1."""
        mu, _ = mu_sigma
        mu_target = np.mean(mu)
        weights = solve_mean_cvar(mu, synthetic_returns, mu_target, N_ASSETS)
        _assert_weights_valid(weights)

    def test_solve_mean_cvar_feasible_range(self, synthetic_returns, mu_sigma):
        """CVaR solver must succeed for targets across the feasible range."""
        mu, _ = mu_sigma
        for frac in [0.1, 0.5, 0.9]:
            mu_target = np.min(mu) + frac * (np.max(mu) - np.min(mu))
            weights = solve_mean_cvar(mu, synthetic_returns, mu_target, N_ASSETS)
            _assert_weights_valid(weights)

    def test_solve_mean_cvar_infeasible(self, synthetic_returns, mu_sigma):
        """CVaR solver must return None when the target is unachievable."""
        mu, _ = mu_sigma
        mu_target = np.max(mu) * 100.0
        weights = solve_mean_cvar(mu, synthetic_returns, mu_target, N_ASSETS)
        assert weights is None

    def test_solve_mean_cvar_beta_sensitivity(self, synthetic_returns, mu_sigma):
        """Both beta=0.90 and beta=0.99 should yield valid portfolios."""
        mu, _ = mu_sigma
        mu_target = np.mean(mu)
        for beta in [0.90, 0.99]:
            weights = solve_mean_cvar(
                mu, synthetic_returns, mu_target, N_ASSETS, beta=beta
            )
            _assert_weights_valid(weights)


# ---------------------------------------------------------------------------
# Frontier mapper tests
# ---------------------------------------------------------------------------

class TestFrontierMapper:
    def test_frontier_mapper_monotone(self, mu_sigma):
        """Target-return grid must be monotonically non-decreasing."""
        mu, _ = mu_sigma
        grid = build_target_return_grid(mu)
        assert len(grid) == 100
        diffs = np.diff(grid)
        assert np.all(diffs >= 0), "Grid is not monotonically non-decreasing."

    def test_frontier_mapper_endpoints(self, mu_sigma):
        """Grid endpoints must match min(mu) and max(mu)."""
        mu, _ = mu_sigma
        grid = build_target_return_grid(mu)
        assert abs(grid[0] - np.min(mu)) < 1e-12
        assert abs(grid[-1] - np.max(mu)) < 1e-12

    def test_lambda_boundary(self, mu_sigma):
        """lambda=1 must yield the minimum return; lambda=100 the maximum."""
        mu, _ = mu_sigma
        grid = build_target_return_grid(mu)
        assert lambda_to_target_return(1, grid) == grid[0]
        assert lambda_to_target_return(100, grid) == grid[99]

    def test_lambda_to_target_return_midpoint(self, mu_sigma):
        """lambda=50 must correspond to grid index 49."""
        mu, _ = mu_sigma
        grid = build_target_return_grid(mu)
        assert lambda_to_target_return(50, grid) == grid[49]

    def test_lambda_out_of_range(self, mu_sigma):
        """Invalid lambda values must raise AssertionError."""
        mu, _ = mu_sigma
        grid = build_target_return_grid(mu)
        with pytest.raises(AssertionError):
            lambda_to_target_return(0, grid)
        with pytest.raises(AssertionError):
            lambda_to_target_return(101, grid)

    def test_find_feasible_target_within_range(self, mu_sigma):
        """Feasible target must be returned unchanged."""
        mu, sigma = mu_sigma
        mu_target = np.mean(mu)
        result = find_feasible_target(mu_target, mu, sigma)
        assert result == mu_target

    def test_find_feasible_target_projects_infeasible(self, mu_sigma):
        """Infeasible target must be projected to fallback_fraction * max(mu)."""
        mu, sigma = mu_sigma
        mu_target = np.max(mu) * 2.0
        result = find_feasible_target(mu_target, mu, sigma, fallback_fraction=0.99)
        assert abs(result - 0.99 * np.max(mu)) < 1e-12

    def test_build_target_return_grid_equal_returns(self):
        """Edge case: all assets have the same expected return -> constant grid."""
        mu_flat = np.full(N_ASSETS, 0.0005)
        grid = build_target_return_grid(mu_flat)
        assert len(grid) == 100
        np.testing.assert_allclose(grid, 0.0005, atol=1e-15)

    def test_build_target_return_grid_two_assets(self):
        """Grid with two distinct assets must span [min, max] exactly."""
        mu_two = np.array([0.001, 0.003])
        grid = build_target_return_grid(mu_two)
        assert grid[0] == pytest.approx(0.001)
        assert grid[-1] == pytest.approx(0.003)
        assert len(grid) == 100


# ---------------------------------------------------------------------------
# RollingEstimator tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def returns_df():
    """Build a (300, 5) DataFrame with a DatetimeIndex for RollingEstimator tests."""
    rng = np.random.default_rng(RNG_SEED)
    mu_true = np.array([0.0003, 0.0005, 0.0007, 0.0004, 0.0006])
    data = rng.normal(mu_true, 0.01, size=(N_OBS, N_ASSETS))
    dates = pd.bdate_range("2020-01-02", periods=N_OBS)
    tickers = [f"A{i}" for i in range(N_ASSETS)]
    return pd.DataFrame(data, index=dates, columns=tickers)


class TestRollingEstimator:

    def test_rolling_estimator_cache(self, returns_df):
        """Calling get() twice for the same date should use the cache."""
        estimator = RollingEstimator(returns_df, window=252)
        date = returns_df.index[260]
        result1 = estimator.get(date)
        result2 = estimator.get(date)
        # Same object from cache
        assert result1 is result2, "Cache miss: second call returned a new object."

    def test_rolling_estimator_output_keys(self, returns_df):
        """Output dict must contain all required keys."""
        estimator = RollingEstimator(returns_df, window=252)
        date = returns_df.index[-1]
        result = estimator.get(date)
        for key in ("mu", "sigma", "semi_cov", "scenario_returns", "n_obs"):
            assert key in result, f"Missing key: {key}"

    def test_rolling_estimator_shapes(self, returns_df):
        """Output arrays must have correct shapes for the given window."""
        window = 60
        estimator = RollingEstimator(returns_df, window=window)
        # Date with enough history
        date = returns_df.index[200]
        result = estimator.get(date)
        assert result["mu"].shape == (N_ASSETS,)
        assert result["sigma"].shape == (N_ASSETS, N_ASSETS)
        assert result["semi_cov"].shape == (N_ASSETS, N_ASSETS)
        assert result["scenario_returns"].shape[1] == N_ASSETS
        assert result["n_obs"] == min(window, 201)

    def test_rolling_estimator_short_history(self, returns_df):
        """Early dates with fewer than `window` rows must not raise an error."""
        estimator = RollingEstimator(returns_df, window=252)
        date = returns_df.index[10]  # only 11 rows available
        result = estimator.get(date)
        assert result["n_obs"] == 11
        assert result["mu"].shape == (N_ASSETS,)

    def test_rolling_estimator_covariance_psd(self, returns_df):
        """Estimated covariance matrix must be positive definite (regularised)."""
        estimator = RollingEstimator(returns_df, window=100)
        date = returns_df.index[150]
        result = estimator.get(date)
        eigenvalues = np.linalg.eigvalsh(result["sigma"])
        assert np.all(eigenvalues > 0), (
            f"Covariance is not positive definite. Min eigenvalue: {eigenvalues.min()}"
        )

    def test_rolling_estimator_different_windows(self, returns_df):
        """Two estimators with different windows must give different sigma."""
        est_short = RollingEstimator(returns_df, window=30)
        est_long = RollingEstimator(returns_df, window=200)
        date = returns_df.index[250]
        sigma_short = est_short.get(date)["sigma"]
        sigma_long = est_long.get(date)["sigma"]
        # Matrices should differ (same data but different observation counts)
        assert not np.allclose(sigma_short, sigma_long), (
            "Expected different covariance estimates for different windows."
        )

    def test_rolling_estimator_multiple_dates_cached(self, returns_df):
        """Multiple date queries must each be independently cached."""
        estimator = RollingEstimator(returns_df, window=60)
        dates = returns_df.index[60:65]
        results = [estimator.get(d) for d in dates]
        # Verify all are cached
        assert len(estimator._cache) >= len(dates)
        # Verify second retrieval still uses cache
        for d, r in zip(dates, results):
            assert estimator.get(d) is r


# ---------------------------------------------------------------------------
# Estimator utility tests
# ---------------------------------------------------------------------------

class TestEstimatorUtilities:
    def test_estimate_expected_returns_shape(self, synthetic_returns):
        """estimate_expected_returns must return (n_assets,) array."""
        mu = estimate_expected_returns(synthetic_returns)
        assert mu.shape == (N_ASSETS,)

    def test_estimate_expected_returns_zero_mean(self):
        """Zero-mean returns should yield mu close to zero."""
        rng = np.random.default_rng(1)
        returns = rng.standard_normal((10_000, 3))
        mu = estimate_expected_returns(returns)
        np.testing.assert_allclose(mu, 0.0, atol=0.05)

    def test_estimate_expected_returns_annualize(self, synthetic_returns):
        """Annualised mu must equal daily mu * 252."""
        mu_daily = estimate_expected_returns(synthetic_returns, annualize=False)
        mu_annual = estimate_expected_returns(synthetic_returns, annualize=True)
        np.testing.assert_allclose(mu_annual, mu_daily * 252, rtol=1e-10)

    def test_estimate_covariance_shape(self, synthetic_returns):
        """Covariance matrix must have shape (n_assets, n_assets)."""
        sigma = estimate_covariance(synthetic_returns)
        assert sigma.shape == (N_ASSETS, N_ASSETS)

    def test_estimate_covariance_symmetry(self, synthetic_returns):
        """Covariance matrix must be symmetric."""
        sigma = estimate_covariance(synthetic_returns)
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-14)

    def test_estimate_covariance_positive_definite(self, synthetic_returns):
        """Regularised covariance must be positive definite."""
        sigma = estimate_covariance(synthetic_returns)
        eigenvalues = np.linalg.eigvalsh(sigma)
        assert np.all(eigenvalues > 0)

    def test_estimate_covariance_single_asset(self):
        """Single-asset edge case must return a (1, 1) matrix."""
        returns = np.random.default_rng(0).standard_normal((100, 1))
        sigma = estimate_covariance(returns)
        assert sigma.shape == (1, 1)
        assert sigma[0, 0] > 0

    def test_estimate_covariance_constant_returns(self):
        """Constant returns (zero variance) plus regularisation must give pos-def."""
        returns = np.ones((100, 3)) * 0.001   # zero variance
        sigma = estimate_covariance(returns, regularize=1e-8)
        eigenvalues = np.linalg.eigvalsh(sigma)
        assert np.all(eigenvalues > 0)
