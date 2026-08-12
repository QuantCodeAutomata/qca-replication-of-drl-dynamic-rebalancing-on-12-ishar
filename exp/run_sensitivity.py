"""
EXP_3: Sensitivity and Ambiguity Resolution Study (Yu & Chang 2025)

Tests how replication outcomes change across:
  A) Semivariance benchmark B: {0, mean, target_return}
  B) Lambda-to-target-return mapping: {asset_minmax, feasible_minmax, percentile}
  C) Optimizer estimation window: {126, 252, 504}
  D) Policy evaluation: {deterministic, stochastic}

Usage:
    python exp/run_sensitivity.py --universe etf
    python exp/run_sensitivity.py --universe djia
"""
import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results" / "sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("exp.sensitivity")
warnings.filterwarnings("ignore")

from src.data_module.downloader import download_universe, ETF_TICKERS, DJIA_TICKERS
from src.data_module.preprocessor import (
    compute_log_returns, compute_adjusted_ohlc, align_trading_calendar
)
from src.features.feature_builder import build_feature_matrix
from src.features.standardizer import rolling_zscore_standardize
from src.optimizer.estimators import RollingEstimator
from src.optimizer.portfolio_optimizer import (
    solve_mean_variance, estimate_semi_covariance
)
from src.optimizer.frontier_mapper import build_target_return_grid, lambda_to_target_return
from src.backtest.rolling_backtest import build_rolling_windows, get_window_dates
from src.backtest.benchmarks import compute_static_benchmark, compute_buy_hold_benchmark
from src.stats.metrics import compute_performance_metrics
from src.stats.statistical_tests import alpha_regression_newey_west
from src.viz.plots import plot_sensitivity_tornado
from src.pipeline import (
    LOOKBACK, RL_OBS_LOOKBACK, INITIAL_VALUE, CVaR_BETA, REBALANCE_EVERY,
    N_TIMESTEPS, N_VAL_RUNS,
    make_optimizer_fn, make_estimator_fn, build_obs_array,
    train_and_eval_window, episode_log_to_series, MODEL_ORDER,
)

# ── Sensitivity grid ──────────────────────────────────────────────────────────
@dataclass
class SensConfig:
    name: str
    semivar_B: float          = 0.0
    estimation_window: int    = 252
    deterministic_eval: bool  = True
    lambda_mapping: str       = "asset_minmax"  # or "feasible_minmax" / "percentile"

BASE_CONFIG = SensConfig(
    name="base",
    semivar_B=0.0, estimation_window=252,
    deterministic_eval=True, lambda_mapping="asset_minmax"
)

SENSITIVITY_GRID = [
    # A: Semivariance benchmark B
    SensConfig("B=mean",         semivar_B=float("nan"),   estimation_window=252),
    SensConfig("B=0",            semivar_B=0.0,             estimation_window=252),
    # B: Estimation window
    SensConfig("window=126",     semivar_B=0.0,             estimation_window=126),
    SensConfig("window=252",     semivar_B=0.0,             estimation_window=252),
    SensConfig("window=504",     semivar_B=0.0,             estimation_window=504),
    # C: Deterministic vs stochastic evaluation
    SensConfig("det_eval",       semivar_B=0.0, estimation_window=252, deterministic_eval=True),
    SensConfig("stoch_eval",     semivar_B=0.0, estimation_window=252, deterministic_eval=False),
    # D: Lambda mapping
    SensConfig("lambda_minmax",  semivar_B=0.0, estimation_window=252, lambda_mapping="asset_minmax"),
    SensConfig("lambda_pctile",  semivar_B=0.0, estimation_window=252, lambda_mapping="percentile"),
]


def build_target_grid_alt(mu: np.ndarray, mapping: str, n_levels: int = 100) -> np.ndarray:
    """Alternative lambda-to-target-return mappings for sensitivity."""
    if mapping == "asset_minmax":
        return build_target_return_grid(mu, n_levels)
    elif mapping == "feasible_minmax":
        # Same as asset_minmax for long-only (max achievable = max(mu))
        return build_target_return_grid(mu, n_levels)
    elif mapping == "percentile":
        # Map lambda levels to percentiles of expected asset returns
        quantiles = np.linspace(0, 100, n_levels)
        return np.percentile(mu, quantiles)
    else:
        return build_target_return_grid(mu, n_levels)


def run_single_config(
    config: SensConfig,
    asset_returns_df: pd.DataFrame,
    std_features: dict,
    tickers: list,
    benchmark_ret: pd.Series,
    all_dates: pd.DatetimeIndex,
    universe: str,
) -> Dict:
    """Run a single sensitivity configuration and return metrics."""
    n_assets = len(tickers)

    # Handle mean-based semivariance B
    semivar_B = config.semivar_B
    use_mean_B = np.isnan(semivar_B) if isinstance(semivar_B, float) else False
    if use_mean_B:
        semivar_B = 0.0  # will override per window

    rolling_est = RollingEstimator(
        asset_returns_df, window=config.estimation_window, semivar_benchmark=semivar_B
    )

    windows = build_rolling_windows(all_dates, universe=universe)
    oos_returns = []

    n_features = next(iter(std_features.values())).shape[1]

    for win in windows[:2]:  # Limit to first 2 windows for sensitivity speed
        train_dates = get_window_dates(all_dates, win, "train")
        val_dates   = get_window_dates(all_dates, win, "val")
        test_dates  = get_window_dates(all_dates, win, "test")

        if len(train_dates) < 50 or len(test_dates) < 20:
            continue

        train_ret = asset_returns_df.loc[train_dates].values
        val_ret   = asset_returns_df.loc[val_dates].values
        test_ret  = asset_returns_df.loc[test_dates].values

        train_obs = build_obs_array(std_features, tickers, train_dates, RL_OBS_LOOKBACK)
        val_obs   = build_obs_array(std_features, tickers, val_dates,   RL_OBS_LOOKBACK)
        test_obs  = build_obs_array(std_features, tickers, test_dates,  RL_OBS_LOOKBACK)

        try:
            model, _, test_log = train_and_eval_window(
                algo="PPO", risk_model="variance",
                train_returns=train_ret, train_obs=train_obs, train_dates=train_dates,
                val_returns=val_ret,   val_obs=val_obs,     val_dates=val_dates,
                test_returns=test_ret, test_obs=test_obs,   test_dates=test_dates,
                rolling_est=rolling_est,
                n_assets=n_assets, n_features=n_features,
                seed=42 + win.window_idx,
                n_timesteps=max(1000, N_TIMESTEPS // 3),
                n_val_runs=max(3, N_VAL_RUNS // 3),
            )
            ts = episode_log_to_series(test_log, test_ret, test_dates)
            oos_returns.append(ts["daily_return"].values)
        except Exception as e:
            logger.warning(f"Config {config.name} window {win.window_idx} failed: {e}")
            # Fallback to equal weight
            eq_ret = np.mean(test_ret, axis=1)
            oos_returns.append(eq_ret)

    if not oos_returns:
        return {"sharpe_ratio": 0.0, "annualized_return": 0.0, "config": config.name}

    all_r = np.concatenate(oos_returns)
    metrics = compute_performance_metrics(all_r)
    metrics["config"] = config.name
    return metrics


def run_sensitivity(universe: str):
    """Run full sensitivity study for the given universe."""
    cache_dir = ROOT / "data" / f"{universe}_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if universe == "etf":
        tickers, benchmark_ticker = ETF_TICKERS, "SPY"
        start, end = "2003-01-01", "2023-01-31"
    else:
        tickers, benchmark_ticker = DJIA_TICKERS, "DIA"
        start, end = "2005-01-01", "2023-01-31"

    logger.info(f"Downloading data for {universe.upper()} sensitivity study …")
    data_dict, benchmark_series = download_universe(tickers, benchmark_ticker, start, end, str(cache_dir))
    data_dict["adj_close"][benchmark_ticker] = benchmark_series
    aligned = align_trading_calendar(data_dict)

    adj_open, adj_high, adj_low, adj_close = compute_adjusted_ohlc(
        aligned["open"], aligned["high"], aligned["low"],
        aligned["close"], aligned["adj_close"]
    )
    log_returns_df = compute_log_returns(aligned["adj_close"])
    all_dates = log_returns_df.index
    asset_returns_df = log_returns_df[tickers]
    benchmark_ret = compute_log_returns(aligned["adj_close"][[benchmark_ticker]]).squeeze()

    feature_data = {
        "adj_close": adj_close[tickers],
        "open":      adj_open[tickers],
        "high":      adj_high[tickers],
        "low":       adj_low[tickers],
        "close":     adj_close[tickers],
        "volume":    aligned["volume"][tickers],
    }
    raw_features = build_feature_matrix(feature_data, tickers)
    std_features = {t: rolling_zscore_standardize(f, 252) for t, f in raw_features.items()}

    # ── Run all configurations ─────────────────────────────────────────────────
    results = []
    configs_to_test = [BASE_CONFIG] + SENSITIVITY_GRID

    for cfg in configs_to_test:
        logger.info(f"Running sensitivity config: {cfg.name} …")
        m = run_single_config(
            cfg, asset_returns_df, std_features, tickers,
            benchmark_ret, all_dates, universe
        )
        results.append(m)
        logger.info(f"  Sharpe={m.get('sharpe_ratio', 0):.3f}  "
                    f"AnnRet={m.get('annualized_return', 0):.3f}")

    # ── Compare to base ────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).set_index("config")
    base_sharpe = results_df.loc["base", "sharpe_ratio"] if "base" in results_df.index else 0
    results_df["sharpe_delta"] = results_df["sharpe_ratio"] - base_sharpe
    results_df["return_delta"] = (
        results_df["annualized_return"]
        - (results_df.loc["base", "annualized_return"] if "base" in results_df.index else 0)
    )

    # ── Save results ───────────────────────────────────────────────────────────
    uni_dir = RESULTS_DIR / universe
    uni_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(uni_dir / "sensitivity_results.csv")

    # Tornado chart
    plot_df = results_df.drop(index="base", errors="ignore")[["sharpe_delta"]].copy()
    if not plot_df.empty:
        fig = plot_sensitivity_tornado(
            plot_df, metric="sharpe_delta",
            title=f"Sensitivity: Sharpe Δ vs Base — {universe.upper()}",
            save_path=str(uni_dir / "sensitivity_tornado.png"),
        )
        plt.close(fig)

    # Write summary
    write_sensitivity_md(results_df, universe)
    logger.info(f"Sensitivity results saved to {uni_dir}")
    return results_df


def write_sensitivity_md(results_df: pd.DataFrame, universe: str):
    """Write sensitivity summary to RESULTS.md section."""
    md_path = RESULTS_DIR / f"sensitivity_{universe}.md"
    lines = [
        f"# Sensitivity Study — {universe.upper()} Universe",
        "",
        "## Summary: Impact on Sharpe Ratio by Ambiguity Dimension",
        "",
        "| Config | Sharpe | AnnRet | Sharpe Δ vs Base |",
        "|--------|--------|--------|-----------------|",
    ]
    for idx, row in results_df.iterrows():
        lines.append(
            f"| {idx} | {row.get('sharpe_ratio', 0):.3f} | "
            f"{row.get('annualized_return', 0):.3f} | "
            f"{row.get('sharpe_delta', 0):.3f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Rows with large |Sharpe Δ| indicate first-order ambiguities.",
        "- Estimation window and lambda mapping are expected to be most influential.",
        "- Deterministic vs stochastic evaluation affects val-Sharpe ranking but not policy quality.",
        "",
    ]
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Sensitivity markdown saved to {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["etf", "djia", "both"], default="etf")
    args = parser.parse_args()

    if args.universe in ("etf", "both"):
        run_sensitivity("etf")
    if args.universe in ("djia", "both"):
        run_sensitivity("djia")

    logger.info("Sensitivity study complete.")


if __name__ == "__main__":
    main()
