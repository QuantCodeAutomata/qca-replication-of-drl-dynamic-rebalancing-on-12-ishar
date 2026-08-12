import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tools.tools import add_constant


def alpha_regression_newey_west(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    n_lags: int = 20,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    Regress strategy returns on benchmark returns with HAC/Newey-West errors.

    Model: ``r_{S,t} = alpha + beta * r_{B,t} + epsilon_t``

    Heteroscedasticity- and autocorrelation-consistent (HAC) standard errors
    are estimated via the Newey-West kernel with ``n_lags`` lags, following
    the recommendation of the paper for daily financial return series.

    Parameters
    ----------
    strategy_returns : np.ndarray
        Daily log returns of the strategy; shape ``(T,)``.
    benchmark_returns : np.ndarray
        Daily log returns of the benchmark; same length as ``strategy_returns``.
    n_lags : int
        Number of lags used in the Newey-West HAC covariance estimator.
        Default 20 (≈ one trading month), as suggested in the paper.
    trading_days : int
        Number of trading days per year, used to annualise ``alpha``.
        Default 252.

    Returns
    -------
    Dict[str, float]
        alpha_daily      – daily intercept from the OLS regression.
        annualized_alpha – ``alpha_daily * trading_days``.
        beta             – slope coefficient (systematic exposure).
        t_stat           – HAC t-statistic for the intercept.
        p_value          – two-sided p-value for the null ``alpha = 0``.
        n_obs            – number of observations used in the regression.
    """
    T = len(strategy_returns)
    X = add_constant(benchmark_returns, prepend=True)
    y = strategy_returns

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ols_result = OLS(y, X).fit()
        hac_cov = cov_hac(ols_result, nlags=n_lags)
        hac_se = np.sqrt(np.diag(hac_cov))

    alpha_daily = float(ols_result.params[0])
    alpha_ann = alpha_daily * trading_days
    beta_coef = float(ols_result.params[1])
    t_stat = float(alpha_daily / hac_se[0]) if hac_se[0] > 1e-12 else 0.0

    p_value = float(2 * stats.t.sf(abs(t_stat), df=T - 2))

    return {
        'alpha_daily': alpha_daily,
        'annualized_alpha': alpha_ann,
        'beta': beta_coef,
        't_stat': t_stat,
        'p_value': p_value,
        'n_obs': T,
    }


def sharpe_difference_bootstrap(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    n_bootstrap: int = 1000,
    block_length: int = 20,
    trading_days: int = 252,
    seed: int = 42,
) -> Dict:
    """
    One-sided stationary block bootstrap test for Sharpe ratio superiority.

    Tests H₀: Sharpe(A) ≤ Sharpe(B)  vs  H₁: Sharpe(A) > Sharpe(B).

    Pairs of returns ``(r_{A,t}, r_{B,t})`` are resampled jointly using the
    stationary block bootstrap with expected block length ``block_length``,
    preserving the serial dependence structure of both series simultaneously.
    The p-value is the proportion of bootstrap replications in which
    ``Sharpe(A*) - Sharpe(B*) ≤ 0``.

    Parameters
    ----------
    returns_a : np.ndarray
        Daily log returns of strategy A (the candidate); shape ``(T,)``.
    returns_b : np.ndarray
        Daily log returns of strategy B (the baseline); same length as A.
    n_bootstrap : int
        Number of bootstrap replications.  Default 1000.
    block_length : int
        Expected block length for the stationary block bootstrap.  The
        geometric distribution parameter is set to ``p = 1 / block_length``.
        Default 20 trading days.
    trading_days : int
        Number of trading days per year for Sharpe annualisation.  Default 252.
    seed : int
        Random seed for reproducibility.  Default 42.

    Returns
    -------
    Dict
        observed_sharpe_diff – Sharpe(A) - Sharpe(B) on the observed data.
        bootstrap_p_value    – fraction of bootstraps where diff ≤ 0 (H₀ holds).
        sharpe_a             – annualised Sharpe ratio of strategy A.
        sharpe_b             – annualised Sharpe ratio of strategy B.
    """
    np.random.seed(seed)
    n = len(returns_a)
    assert len(returns_b) == n, (
        f'returns_a and returns_b must have the same length; '
        f'got {n} vs {len(returns_b)}'
    )

    def _sharpe(r: np.ndarray) -> float:
        std = float(r.std())
        mean = float(r.mean())
        if std > 1e-10:
            return float(mean / std * np.sqrt(trading_days))
        # Near-zero volatility: return sign of mean × large value (≈ ±∞ Sharpe)
        if mean > 1e-12:
            return 1e6
        if mean < -1e-12:
            return -1e6
        return 0.0

    observed_diff = _sharpe(returns_a) - _sharpe(returns_b)

    # Stationary block bootstrap: geometric block lengths with p = 1/block_length
    p_block = 1.0 / block_length
    bootstrap_diffs: List[float] = []

    for _ in range(n_bootstrap):
        indices: List[int] = []
        while len(indices) < n:
            start = np.random.randint(0, n)
            while len(indices) < n:
                indices.append(start % n)
                start += 1
                if np.random.random() < p_block:
                    break

        idx = np.array(indices[:n])
        boot_a = returns_a[idx]
        boot_b = returns_b[idx]
        bootstrap_diffs.append(_sharpe(boot_a) - _sharpe(boot_b))

    boot_arr = np.array(bootstrap_diffs)
    # One-sided p-value: proportion of bootstraps where H₀ is not rejected
    p_value = float(np.mean(boot_arr <= 0))

    return {
        'observed_sharpe_diff': observed_diff,
        'bootstrap_p_value': p_value,
        'sharpe_a': _sharpe(returns_a),
        'sharpe_b': _sharpe(returns_b),
    }
