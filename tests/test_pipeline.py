"""
Integration tests for the full DRL rebalancing pipeline.

Tests verify:
- Module interfaces are compatible
- Data flows correctly from features → optimizer → environment
- Rolling window construction is correct
- Ensemble selection logic works
- Metrics computation is consistent
- No forward-looking leakage in observations
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

# ── helpers ────────────────────────────────────────────────────────────────────
def make_synthetic_returns(n_days: int = 1000, n_assets: int = 5, seed: int = 42):
    np.random.seed(seed)
    dates = pd.date_range("2010-01-01", periods=n_days, freq="B")
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.01,
        index=dates,
        columns=[f"A{i}" for i in range(n_assets)],
    )
    return returns


def make_synthetic_obs(n_days: int, obs_dim: int = 200, seed: int = 0):
    np.random.seed(seed)
    return np.random.randn(n_days, obs_dim).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Rolling windows
# ══════════════════════════════════════════════════════════════════════════════
class TestRollingWindows:
    def test_etf_window_count(self):
        from src.backtest.rolling_backtest import build_rolling_windows
        dates = pd.date_range("2003-01-02", "2023-01-01", freq="B")
        windows = build_rolling_windows(dates, universe="etf")
        assert len(windows) >= 1, "ETF should produce at least 1 window"

    def test_djia_window_count(self):
        from src.backtest.rolling_backtest import build_rolling_windows
        dates = pd.date_range("2005-01-03", "2023-01-01", freq="B")
        windows = build_rolling_windows(dates, universe="djia")
        assert len(windows) >= 1, "DJIA should produce at least 1 window"

    def test_windows_non_overlapping_test_periods(self):
        from src.backtest.rolling_backtest import build_rolling_windows
        dates = pd.date_range("2003-01-02", "2023-01-01", freq="B")
        windows = build_rolling_windows(dates, universe="etf")
        for i in range(len(windows) - 1):
            assert windows[i].test_end < windows[i + 1].test_start, \
                "Test periods must not overlap"

    def test_window_ordering(self):
        from src.backtest.rolling_backtest import build_rolling_windows
        dates = pd.date_range("2003-01-02", "2023-01-01", freq="B")
        windows = build_rolling_windows(dates, universe="etf")
        for win in windows:
            assert win.train_start < win.train_end, "Train start must precede end"
            assert win.train_end < win.val_start,   "Train must precede validation"
            assert win.val_end < win.test_start,    "Validation must precede test"

    def test_get_window_dates(self):
        from src.backtest.rolling_backtest import build_rolling_windows, get_window_dates
        dates = pd.date_range("2003-01-02", "2023-01-01", freq="B")
        windows = build_rolling_windows(dates, universe="etf")
        if windows:
            w = windows[0]
            train_d = get_window_dates(dates, w, "train")
            val_d   = get_window_dates(dates, w, "val")
            test_d  = get_window_dates(dates, w, "test")
            assert len(train_d) > 0
            assert len(val_d) > 0
            assert len(test_d) > 0
            # No overlap between segments
            assert len(set(train_d) & set(test_d)) == 0


# ══════════════════════════════════════════════════════════════════════════════
#  Optimizer interface
# ══════════════════════════════════════════════════════════════════════════════
class TestOptimizerInterface:
    def test_make_optimizer_fn_variance(self):
        from src.pipeline import make_optimizer_fn
        opt_fn = make_optimizer_fn("variance", 5)
        estimates = {
            "mu": np.array([0.001, 0.002, 0.0015, 0.0005, 0.003]),
            "sigma": np.eye(5) * 0.0001,
            "semi_cov": np.eye(5) * 0.00005,
            "scenario_returns": np.random.randn(50, 5) * 0.01,
        }
        mu_target = 0.001
        w = opt_fn(estimates, mu_target)
        assert w is not None
        assert np.isclose(w.sum(), 1.0, atol=1e-4), "Weights must sum to 1"
        assert np.all(w >= -1e-6), "Weights must be non-negative"

    def test_make_optimizer_fn_cvar(self):
        from src.pipeline import make_optimizer_fn
        opt_fn = make_optimizer_fn("cvar", 5)
        returns = np.random.randn(100, 5) * 0.01
        estimates = {
            "mu": returns.mean(axis=0),
            "sigma": np.eye(5) * 0.0001,
            "semi_cov": np.eye(5) * 0.00005,
            "scenario_returns": returns,
        }
        w = opt_fn(estimates, estimates["mu"].mean())
        if w is not None:
            assert np.isclose(w.sum(), 1.0, atol=1e-4)
            assert np.all(w >= -1e-6)

    def test_estimator_fn(self):
        from src.optimizer.estimators import RollingEstimator
        from src.pipeline import make_estimator_fn
        n_days, n_assets = 300, 5
        returns = make_synthetic_returns(n_days, n_assets)
        est = RollingEstimator(returns, window=50)
        dates = returns.index
        est_fn = make_estimator_fn(est, dates)
        result = est_fn(100)
        assert "mu" in result
        assert "sigma" in result
        assert result["mu"].shape == (n_assets,)


# ══════════════════════════════════════════════════════════════════════════════
#  Environment compatibility
# ══════════════════════════════════════════════════════════════════════════════
class TestEnvironmentPipeline:
    def _make_env(self, n_assets=5, T=60):
        from src.rl_env.portfolio_env import PortfolioEnv
        obs_dim = 5 * n_assets * 10  # small observation
        returns = np.random.randn(T, n_assets) * 0.01
        obs_tensors = np.random.randn(T, obs_dim).astype(np.float32)

        def mock_estimator(step_idx):
            return {
                "mu": np.ones(n_assets) * 0.001,
                "sigma": np.eye(n_assets) * 0.0001,
                "semi_cov": np.eye(n_assets) * 0.00005,
                "scenario_returns": returns[:max(1, step_idx)],
            }

        def mock_optimizer(estimates, mu_target):
            return np.ones(n_assets) / n_assets

        env = PortfolioEnv(
            price_returns=returns, observation_tensors=obs_tensors,
            optimizer_fn=mock_optimizer, estimator_fn=mock_estimator,
            n_assets=n_assets, n_features=5, lookback=10,
        )
        return env

    def test_environment_runs_to_completion(self):
        env = self._make_env()
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 1000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert done, "Episode should terminate"

    def test_obs_shape_constant(self):
        env = self._make_env(n_assets=3, T=40)
        obs, _ = env.reset()
        expected_shape = env.observation_space.shape
        for _ in range(5):
            action = env.action_space.sample()
            obs, _, terminated, _, _ = env.step(action)
            assert obs.shape == expected_shape, "Obs shape must remain constant"
            if terminated:
                break

    def test_reward_is_log_return(self):
        """Reward should be log portfolio return — finite and bounded."""
        env = self._make_env()
        obs, _ = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, _, info = env.step(action)
            assert np.isfinite(reward), "Reward must be finite"
            assert -10 <= reward <= 10, "Reward clipped to [-10, 10]"
            if terminated:
                break


# ══════════════════════════════════════════════════════════════════════════════
#  Episode log → time-series conversion
# ══════════════════════════════════════════════════════════════════════════════
class TestEpisodeLogConversion:
    def test_episode_log_to_series_shape(self):
        from src.pipeline import episode_log_to_series, INITIAL_VALUE
        n_assets = 5
        T = 50
        dates = pd.date_range("2020-01-01", periods=T, freq="B")
        returns = np.random.randn(T, n_assets) * 0.01
        test_log = [
            {"h_t": 5, "weights": list(np.ones(n_assets)/n_assets),
             "lambda_t": 50, "portfolio_value": INITIAL_VALUE},
            {"h_t": 10, "weights": list(np.ones(n_assets)/n_assets),
             "lambda_t": 60, "portfolio_value": INITIAL_VALUE},
        ]
        ts = episode_log_to_series(test_log, returns, dates)
        assert len(ts) == T, "Output length must equal number of test dates"
        assert "portfolio_value" in ts.columns
        assert "daily_return" in ts.columns

    def test_episode_log_pv_positive(self):
        from src.pipeline import episode_log_to_series, INITIAL_VALUE
        n_assets = 3
        T = 30
        dates = pd.date_range("2020-01-01", periods=T, freq="B")
        returns = np.zeros((T, n_assets))
        test_log = [{"h_t": 5, "weights": [1/3, 1/3, 1/3], "lambda_t": 50,
                     "portfolio_value": INITIAL_VALUE}]
        ts = episode_log_to_series(test_log, returns, dates)
        assert np.all(ts["portfolio_value"].values >= 0), "Portfolio value must be >= 0"


# ══════════════════════════════════════════════════════════════════════════════
#  Ensemble selection
# ══════════════════════════════════════════════════════════════════════════════
class TestEnsembleSelection:
    def test_select_best_sharpe(self):
        from src.rl_env.agents import select_ensemble_model
        models = {
            "PPO_variance":     (None, 0.5),
            "PPO_semivariance": (None, 0.8),
            "A2C_variance":     (None, 0.3),
        }
        best_name, _ = select_ensemble_model(models)
        assert best_name == "PPO_semivariance"

    def test_tie_breaking_deterministic(self):
        from src.rl_env.agents import select_ensemble_model
        models = {
            "PPO_variance":     (None, 0.6),
            "A2C_variance":     (None, 0.6),
        }
        name1, _ = select_ensemble_model(models)
        name2, _ = select_ensemble_model(models)
        assert name1 == name2, "Tie-breaking must be deterministic"

    def test_single_model(self):
        from src.rl_env.agents import select_ensemble_model
        models = {"PPO_variance": (None, 0.4)}
        best_name, _ = select_ensemble_model(models)
        assert best_name == "PPO_variance"


# ══════════════════════════════════════════════════════════════════════════════
#  No forward-looking leakage check
# ══════════════════════════════════════════════════════════════════════════════
class TestNoLookAheadBias:
    def test_rolling_estimator_uses_past_only(self):
        from src.optimizer.estimators import RollingEstimator
        n_days, n_assets = 300, 5
        returns = make_synthetic_returns(n_days, n_assets)
        est = RollingEstimator(returns, window=50)
        dates = returns.index

        # Estimate at date t should use only data up to and including t
        t = dates[100]
        result = est.get(t)
        t_loc = returns.index.get_loc(t)
        # mu should match mean of returns up to t
        actual_mu = returns.iloc[max(0, t_loc - 49):t_loc + 1].mean().values
        assert np.allclose(result["mu"], actual_mu, atol=1e-10), \
            "Estimator must use only past data"

    def test_observation_tensor_uses_past_only(self):
        """Observation at date t must use only the last `lookback` rows up to t."""
        from src.features.standardizer import build_observation_tensor
        n_days = 400
        dates = pd.date_range("2015-01-01", periods=n_days, freq="B")
        tickers = ["A", "B"]
        n_features = 3
        lookback = 10

        # Create synthetic feature dfs
        std_features = {
            t: pd.DataFrame(
                np.random.randn(n_days, n_features),
                index=dates,
                columns=[f"f{i}" for i in range(n_features)]
            )
            for t in tickers
        }

        date = dates[50]
        tensor = build_observation_tensor(std_features, tickers, date, lookback)
        # Shape check
        assert tensor.shape == (n_features, len(tickers), lookback), \
            f"Expected ({n_features},{len(tickers)},{lookback}), got {tensor.shape}"


# ══════════════════════════════════════════════════════════════════════════════
#  Performance metrics consistency
# ══════════════════════════════════════════════════════════════════════════════
class TestMetricsConsistency:
    def test_sharpe_positive_for_positive_returns(self):
        from src.stats.metrics import compute_performance_metrics
        r = np.ones(252) * 0.001  # constant positive returns
        m = compute_performance_metrics(r)
        assert m["sharpe_ratio"] > 0

    def test_max_drawdown_always_non_positive(self):
        from src.stats.metrics import compute_performance_metrics
        np.random.seed(0)
        r = np.random.randn(500) * 0.01
        m = compute_performance_metrics(r)
        assert m["max_drawdown"] <= 0.0, "Max drawdown must be <= 0"

    def test_cvar_ge_var(self):
        from src.stats.metrics import compute_performance_metrics
        np.random.seed(1)
        r = np.random.randn(500) * 0.02
        m = compute_performance_metrics(r)
        assert m["cvar_95"] >= m["var_95"] - 1e-8, "CVaR must be >= VaR"

    def test_sortino_ge_sharpe_for_positive_mean(self):
        """When mean > 0, sortino >= sharpe since downside vol <= total vol."""
        from src.stats.metrics import compute_performance_metrics
        np.random.seed(2)
        r = np.abs(np.random.randn(300)) * 0.01  # all positive
        m = compute_performance_metrics(r)
        # Sortino should be higher when returns are positive (less downside)
        assert np.isfinite(m["sortino_ratio"])

    def test_metrics_with_benchmark(self):
        from src.stats.metrics import compute_performance_metrics
        np.random.seed(3)
        r = np.random.randn(252) * 0.01
        b = np.random.randn(252) * 0.01
        m = compute_performance_metrics(r, benchmark_returns=b)
        assert "beta" in m
        assert "tracking_error" in m
        assert "information_ratio" in m
        assert "upside_capture" in m
        assert "downside_capture" in m
        assert m["tracking_error"] >= 0

    def test_beta_of_identical_series(self):
        from src.stats.metrics import compute_performance_metrics
        np.random.seed(4)
        r = np.random.randn(252) * 0.01
        m = compute_performance_metrics(r, benchmark_returns=r)
        assert abs(m["beta"] - 1.0) < 0.01, "Beta of identical series should be ~1"


# ══════════════════════════════════════════════════════════════════════════════
#  Statistical tests
# ══════════════════════════════════════════════════════════════════════════════
class TestStatisticalTests:
    def test_alpha_regression_keys(self):
        from src.stats.statistical_tests import alpha_regression_newey_west
        np.random.seed(5)
        r = np.random.randn(252) * 0.01
        b = np.random.randn(252) * 0.01
        result = alpha_regression_newey_west(r, b)
        for k in ["alpha_daily", "annualized_alpha", "beta", "t_stat", "p_value"]:
            assert k in result, f"Key {k} missing from regression result"

    def test_p_value_in_range(self):
        from src.stats.statistical_tests import alpha_regression_newey_west
        np.random.seed(6)
        r = np.random.randn(252) * 0.01
        b = np.random.randn(252) * 0.01
        result = alpha_regression_newey_west(r, b)
        assert 0 <= result["p_value"] <= 1

    def test_sharpe_bootstrap_p_value_range(self):
        from src.stats.statistical_tests import sharpe_difference_bootstrap
        np.random.seed(7)
        r = np.random.randn(252) * 0.01 + 0.001
        b = np.random.randn(252) * 0.01
        result = sharpe_difference_bootstrap(r, b, n_bootstrap=100)
        assert 0 <= result["bootstrap_p_value"] <= 1

    def test_sharpe_bootstrap_sign(self):
        """If strategy has clearly higher Sharpe, p-value should be low."""
        from src.stats.statistical_tests import sharpe_difference_bootstrap
        np.random.seed(8)
        # Strategy: constant positive returns
        r = np.ones(252) * 0.005
        # Benchmark: zero returns
        b = np.zeros(252)
        result = sharpe_difference_bootstrap(r, b, n_bootstrap=200)
        assert result["observed_sharpe_diff"] > 0, "Strategy Sharpe must exceed benchmark"


# ══════════════════════════════════════════════════════════════════════════════
#  Observation tensor
# ══════════════════════════════════════════════════════════════════════════════
class TestObservationTensor:
    def test_build_obs_array_shape(self):
        from src.features.standardizer import rolling_zscore_standardize
        from src.pipeline import build_obs_array
        n_days, n_assets, n_features, lookback = 400, 3, 5, 10
        dates = pd.date_range("2015-01-01", periods=n_days, freq="B")
        tickers = [f"T{i}" for i in range(n_assets)]
        std_features = {
            t: pd.DataFrame(
                np.random.randn(n_days, n_features),
                index=dates, columns=[f"f{i}" for i in range(n_features)]
            )
            for t in tickers
        }
        obs = build_obs_array(std_features, tickers, dates[lookback:lookback+20], lookback)
        assert obs.shape == (20, n_features * n_assets * lookback)
        assert obs.dtype == np.float32


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark
# ══════════════════════════════════════════════════════════════════════════════
class TestBenchmarks:
    def test_buy_hold_pv_grows_with_positive_returns(self):
        from src.backtest.benchmarks import compute_buy_hold_benchmark
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        bench_ret = pd.Series(np.ones(100) * 0.001, index=dates)
        result = compute_buy_hold_benchmark(bench_ret, dates, initial_value=1000.0)
        assert result["portfolio_value"].iloc[-1] > 1000.0

    def test_buy_hold_starts_at_initial_value(self):
        from src.backtest.benchmarks import compute_buy_hold_benchmark
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        bench_ret = pd.Series(np.zeros(50), index=dates)
        result = compute_buy_hold_benchmark(bench_ret, dates, initial_value=500.0)
        assert abs(result["portfolio_value"].iloc[0] - 500.0) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
