import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.optimizer.estimators import estimate_covariance, estimate_expected_returns
from src.optimizer.portfolio_optimizer import (
    estimate_semi_covariance,
    solve_mean_cvar,
    solve_mean_semivariance,
    solve_mean_variance,
)

logger = logging.getLogger(__name__)


def compute_static_benchmark(
    log_returns_df: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    risk_model: str,
    estimation_window: int = 252,
    rebalance_every: int = 30,
    beta_cvar: float = 0.95,
    semivar_benchmark: float = 0.0,
    initial_value: float = 1000.0,
) -> pd.DataFrame:
    """
    Compute a static benchmark portfolio using tangency-style optimisation.

    Re-optimises every ``rebalance_every`` trading days using a trailing
    ``estimation_window``-day window of log returns.  When the window contains
    fewer than 20 observations (e.g., near the start of history), the portfolio
    falls back to equal weighting.

    The return target is set to ``0.9 * max(mu)`` to approximate a tangency
    portfolio on the efficient frontier without requiring an explicit risk-free
    rate.

    Parameters
    ----------
    log_returns_df : pd.DataFrame
        Full history of daily log returns; index is a DatetimeIndex sorted
        chronologically, columns are asset tickers.
    test_dates : pd.DatetimeIndex
        Ordered sequence of trading dates that define the back-test period.
    risk_model : str
        Optimisation objective.  One of ``'variance'``, ``'semivariance'``,
        or ``'cvar'``.
    estimation_window : int
        Number of trailing trading days used for parameter estimation.
        Default 252 (≈ one calendar year).
    rebalance_every : int
        Number of trading days between successive re-optimisations.
        Default 30 (≈ monthly).
    beta_cvar : float
        CVaR confidence level used when ``risk_model='cvar'``.  Default 0.95.
    semivar_benchmark : float
        Return threshold B used in the semi-covariance estimator when
        ``risk_model='semivariance'``.  Default 0.0.
    initial_value : float
        Starting portfolio value in USD.  Default 1000.

    Returns
    -------
    pd.DataFrame
        Indexed by ``test_dates`` with columns:

        * ``portfolio_value`` – cumulative portfolio value (USD).
        * ``daily_return``    – log return of the portfolio for that day.
        * ``weights``         – np.ndarray of asset weights held that day.
    """
    n_assets = log_returns_df.shape[1]
    current_weights = np.ones(n_assets) / n_assets
    portfolio_value = initial_value

    records = []
    for i, date in enumerate(test_dates):
        # ── Rebalance if on a scheduled rebalance day ──────────────────────
        if i % rebalance_every == 0:
            date_loc = log_returns_df.index.get_loc(date)
            start_loc = max(0, date_loc - estimation_window)
            window_returns = log_returns_df.iloc[start_loc:date_loc].values

            if len(window_returns) < 20:
                current_weights = np.ones(n_assets) / n_assets
            else:
                mu = estimate_expected_returns(window_returns)
                mu_target = np.max(mu) * 0.9  # Tangency-style high return target

                new_weights = None
                if risk_model == 'variance':
                    sigma = estimate_covariance(window_returns)
                    new_weights = solve_mean_variance(mu, sigma, mu_target, n_assets)
                elif risk_model == 'semivariance':
                    semi_cov = estimate_semi_covariance(window_returns, semivar_benchmark)
                    new_weights = solve_mean_semivariance(mu, semi_cov, mu_target, n_assets)
                elif risk_model == 'cvar':
                    new_weights = solve_mean_cvar(
                        mu, window_returns, mu_target, n_assets, beta_cvar
                    )

                if new_weights is not None:
                    current_weights = new_weights
                else:
                    logger.warning(
                        f'Static {risk_model} optimisation failed at {date}; '
                        f'retaining previous weights.'
                    )

        # ── Mark-to-market ────────────────────────────────────────────────
        if date in log_returns_df.index:
            asset_returns = log_returns_df.loc[date].values
            port_log_return = float(np.dot(current_weights, asset_returns))
        else:
            port_log_return = 0.0

        prev_value = portfolio_value
        portfolio_value = prev_value * np.exp(port_log_return)

        records.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'daily_return': port_log_return,
            'weights': current_weights.copy(),
        })

    df = pd.DataFrame(records).set_index('date')
    return df


def compute_buy_hold_benchmark(
    benchmark_returns: pd.Series,
    test_dates: pd.DatetimeIndex,
    initial_value: float = 1000.0,
) -> pd.DataFrame:
    """
    Compute a buy-and-hold benchmark (e.g. SPY or DIA) aligned to test dates.

    Missing dates in ``benchmark_returns`` are filled with 0 (no return), so
    the portfolio value remains unchanged on non-trading days.

    Parameters
    ----------
    benchmark_returns : pd.Series
        Daily log returns of the benchmark index, indexed by date.
    test_dates : pd.DatetimeIndex
        Sequence of dates that define the back-test period.
    initial_value : float
        Starting portfolio value in USD.  Default 1000.

    Returns
    -------
    pd.DataFrame
        Indexed by ``test_dates`` with columns:

        * ``portfolio_value`` – cumulative portfolio value (USD).
        * ``daily_return``    – log return of the benchmark for that day.
    """
    aligned = benchmark_returns.reindex(test_dates).fillna(0.0)
    pv = initial_value * np.exp(aligned.cumsum())
    return pd.DataFrame(
        {'portfolio_value': pv, 'daily_return': aligned.values},
        index=test_dates,
    )
