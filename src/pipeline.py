"""
Main DRL Portfolio Rebalancing Pipeline
Replication of Yu & Chang (2025) - Smart Tangency Portfolio

Integrates: data → features → optimizer → RL environment → backtest → metrics
"""
import os
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# ── experiment settings (override via env variables) ──────────────────────────
N_TIMESTEPS = int(os.getenv("DRL_TIMESTEPS", "5000"))   # SB3 training steps
N_VAL_RUNS  = int(os.getenv("DRL_VAL_RUNS",  "10"))     # validation episodes
N_LAMBDA    = 100                                         # frontier grid points
LOOKBACK    = 252                                         # optimizer estimation window (paper: 252 days)
# RL observation lookback (decoupled from optimizer window for tractable training).
# Paper specifies 252-day state tensor but that yields 102 k+ obs dimensions.
# Using 30 trading days keeps the MLP tractable while preserving the 34-feature
# cross-sectional signal; document this as a practical implementation choice.
RL_OBS_LOOKBACK = int(os.getenv("DRL_OBS_LOOKBACK", "30"))
ESTIMATION_WINDOW = 252                                   # optimizer window
REBALANCE_EVERY   = 30                                    # static benchmark period
CVaR_BETA         = 0.95
SEMIVAR_B         = 0.0    # base case: B=0 (documented choice)
INITIAL_VALUE     = 1000.0
RISK_FREE         = 0.0

MODEL_ORDER = [
    "PPO_variance", "PPO_semivariance", "PPO_cvar",
    "A2C_variance", "A2C_semivariance", "A2C_cvar",
]

# ── optimizer factory ──────────────────────────────────────────────────────────
from src.optimizer.portfolio_optimizer import (
    solve_mean_variance, solve_mean_semivariance, solve_mean_cvar,
    estimate_semi_covariance,
)
from src.optimizer.frontier_mapper import (
    build_target_return_grid, lambda_to_target_return, find_feasible_target,
)
from src.optimizer.estimators import RollingEstimator


def make_optimizer_fn(risk_model: str, n_assets: int):
    """Return a callable (estimates, mu_target) -> weights for the given risk model."""

    def optimizer_fn(estimates: Dict, mu_target: float) -> Optional[np.ndarray]:
        mu = estimates["mu"]
        feasible_target = find_feasible_target(mu_target, mu, estimates.get("sigma"))

        if risk_model == "variance":
            sigma = estimates["sigma"]
            return solve_mean_variance(mu, sigma, feasible_target, n_assets)
        elif risk_model == "semivariance":
            semi_cov = estimates["semi_cov"]
            return solve_mean_semivariance(mu, semi_cov, feasible_target, n_assets)
        elif risk_model == "cvar":
            scenarios = estimates["scenario_returns"]
            return solve_mean_cvar(mu, scenarios, feasible_target, n_assets, CVaR_BETA)
        else:
            raise ValueError(f"Unknown risk model: {risk_model}")

    return optimizer_fn


def make_estimator_fn(rolling_est: RollingEstimator, dates: pd.DatetimeIndex):
    """Return a callable (step_idx) -> estimates dict."""

    def estimator_fn(step_idx: int) -> Dict:
        idx = min(step_idx, len(dates) - 1)
        date = dates[idx]
        return rolling_est.get(date)

    return estimator_fn


# ── observation tensor builder ─────────────────────────────────────────────────
from src.features.standardizer import build_observation_tensor


def build_obs_array(
    std_features: Dict[str, pd.DataFrame],
    tickers: List[str],
    dates: pd.DatetimeIndex,
    lookback: int = LOOKBACK,
) -> np.ndarray:
    """Pre-compute observation tensors for all dates.

    Returns: (T, n_features * n_assets * lookback) float32 array.
    """
    tensors = []
    for date in dates:
        try:
            t = build_observation_tensor(std_features, tickers, date, lookback)
        except Exception:
            n_feat = next(iter(std_features.values())).shape[1]
            t = np.zeros((n_feat, len(tickers), lookback), dtype=np.float32)
        flat = t.flatten().astype(np.float32)
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        tensors.append(flat)
    return np.stack(tensors, axis=0)  # (T, flat_obs_dim)


# ── single-window training + evaluation ───────────────────────────────────────
from src.rl_env.portfolio_env import PortfolioEnv
from src.rl_env.agents import train_agent, evaluate_agent, run_validation_runs


def train_and_eval_window(
    algo: str,
    risk_model: str,
    train_returns: np.ndarray,
    train_obs: np.ndarray,
    train_dates: pd.DatetimeIndex,
    val_returns: np.ndarray,
    val_obs: np.ndarray,
    val_dates: pd.DatetimeIndex,
    test_returns: np.ndarray,
    test_obs: np.ndarray,
    test_dates: pd.DatetimeIndex,
    rolling_est: RollingEstimator,
    n_assets: int,
    n_features: int,
    seed: int = 42,
    n_timesteps: int = N_TIMESTEPS,
    n_val_runs: int = N_VAL_RUNS,
) -> Tuple[Any, float, List[Dict]]:
    """Train one RL agent, validate it, and evaluate on test set.

    Returns: (trained_model, median_val_sharpe, test_episode_log)
    """
    opt_fn = make_optimizer_fn(risk_model, n_assets)

    def _make_env_kwargs(returns, obs_arr, dates):
        return dict(
            price_returns=returns,
            observation_tensors=obs_arr,
            optimizer_fn=opt_fn,
            estimator_fn=make_estimator_fn(rolling_est, dates),
            n_assets=n_assets,
            n_features=n_features,
            lookback=RL_OBS_LOOKBACK,   # RL observation window (decoupled from optimizer)
            initial_value=INITIAL_VALUE,
        )

    train_kwargs = _make_env_kwargs(train_returns, train_obs, train_dates)
    val_kwargs   = _make_env_kwargs(val_returns,   val_obs,   val_dates)
    test_kwargs  = _make_env_kwargs(test_returns,  test_obs,  test_dates)

    logger.info(f"  Training {algo}-{risk_model} for {n_timesteps} steps …")
    model = train_agent(algo, train_kwargs, total_timesteps=n_timesteps, seed=seed)

    logger.info(f"  Validating with {n_val_runs} runs …")
    median_sharpe, _ = run_validation_runs(model, val_kwargs, n_runs=n_val_runs,
                                           deterministic=True)

    logger.info(f"  Evaluating on test set …")
    _, test_log = evaluate_agent(model, test_kwargs, deterministic=True, seed=seed)

    return model, float(median_sharpe), test_log


# ── episode log → time-series ─────────────────────────────────────────────────
def episode_log_to_series(
    test_log: List[Dict],
    test_returns: np.ndarray,
    test_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Convert per-step episode log to a date-indexed daily time-series.

    Fills each hold period with day-by-day portfolio values.
    """
    n = len(test_returns)
    pv = np.zeros(n)
    weights_mat = np.zeros((n, test_returns.shape[1]))
    lambda_arr = np.zeros(n)
    horizon_arr = np.zeros(n)

    current_pv = INITIAL_VALUE
    pos = 0
    for step_info in test_log:
        h_t = step_info.get("h_t", 1)
        w   = np.array(step_info.get("weights", []))
        lam = step_info.get("lambda_t", 50)

        end_pos = min(pos + h_t, n)
        for d in range(pos, end_pos):
            ret_d = np.dot(w, test_returns[d]) if len(w) > 0 else 0.0
            current_pv *= np.exp(ret_d)
            pv[d] = current_pv
            if len(w) > 0:
                weights_mat[d] = w
            lambda_arr[d] = lam
            horizon_arr[d] = h_t
        pos = end_pos
        if pos >= n:
            break

    # Forward-fill any unfilled tail
    if pos < n and pos > 0:
        for d in range(pos, n):
            pv[d] = pv[pos - 1]
            weights_mat[d] = weights_mat[pos - 1]
            lambda_arr[d] = lambda_arr[pos - 1]
            horizon_arr[d] = horizon_arr[pos - 1]

    df = pd.DataFrame({
        "portfolio_value": pv,
        "daily_return": np.diff(np.log(np.maximum(pv, 1e-10)), prepend=np.log(INITIAL_VALUE + 1e-10)),
        "lambda_t": lambda_arr,
        "h_t": horizon_arr,
    }, index=test_dates[:n])
    for j in range(weights_mat.shape[1]):
        df[f"w_{j}"] = weights_mat[:, j]

    return df
