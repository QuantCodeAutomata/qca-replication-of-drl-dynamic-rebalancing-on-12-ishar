import logging
from typing import Dict

import numpy as np
import pandas as pd

from .portfolio_optimizer import estimate_semi_covariance

logger = logging.getLogger(__name__)


def estimate_expected_returns(
    log_returns: np.ndarray,
    annualize: bool = False,
) -> np.ndarray:
    """
    Estimate expected returns as the historical mean of log returns.

    The estimator is the sample mean computed over all T observations in the
    trailing window:

        mu_i = (1 / T) * sum_{t=1}^{T} r_{it}

    Annualisation (optional) multiplies daily means by 252 trading days.

    Parameters
    ----------
    log_returns : np.ndarray, shape (T, n_assets)
        Matrix of historical daily log returns; rows are time steps.
    annualize : bool
        If True, multiply the daily mean by 252 to obtain an annualised figure.
        Default False (daily scale, matching typical DRL reward scaling).

    Returns
    -------
    np.ndarray, shape (n_assets,)
        Mean daily (or annualised) log return for each asset.
    """
    mu = np.mean(log_returns, axis=0)
    if annualize:
        mu = mu * 252
    return mu


def estimate_covariance(
    log_returns: np.ndarray,
    regularize: float = 1e-8,
) -> np.ndarray:
    """
    Estimate the sample covariance matrix with Tikhonov regularisation.

    The raw sample covariance is:

        Sigma = (1 / (T-1)) * (R - mu 1')' (R - mu 1')

    A small diagonal perturbation ``regularize * I`` is added to guarantee
    strict positive-definiteness for the downstream SOCP / QP solvers even
    when the window is shorter than the number of assets.

    Parameters
    ----------
    log_returns : np.ndarray, shape (T, n_assets)
        Matrix of historical daily log returns.
    regularize : float
        Magnitude of the diagonal regularisation term.  Default 1e-8.

    Returns
    -------
    np.ndarray, shape (n_assets, n_assets)
        Regularised sample covariance matrix (positive definite).
    """
    T = log_returns.shape[0]
    if T <= 1:
        # np.cov requires at least 2 observations (divides by T-1).
        # Return a regularised identity so downstream solvers stay feasible.
        n = log_returns.shape[1]
        return regularize * np.eye(n)

    sigma = np.cov(log_returns.T)
    if sigma.ndim == 0:
        # Single-asset edge case: np.cov returns a scalar
        sigma = np.array([[float(sigma)]])
    # Replace any residual NaNs/Infs (degenerate columns) with zeros before symmetrising
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    # Enforce exact symmetry to satisfy CVXPY's Hermitian requirement
    sigma = (sigma + sigma.T) / 2.0
    sigma += regularize * np.eye(sigma.shape[0])
    return sigma


class RollingEstimator:
    """
    Precomputes and caches rolling optimiser inputs for all requested dates.

    For each query date the estimator slices the trailing ``window`` rows from
    the log-return DataFrame and computes:
        - mu         : expected returns  (sample mean)
        - sigma      : covariance matrix (regularised sample covariance)
        - semi_cov   : semi-covariance matrix (downside risk, benchmark B=0)
        - scenario_returns : raw return matrix for CVaR scenario generation

    Results are stored in an internal dictionary keyed by ``pd.Timestamp`` so
    that repeated queries for the same date return immediately from cache
    without re-computation.

    Parameters
    ----------
    log_returns_df : pd.DataFrame
        DataFrame of daily log returns; index is a DatetimeIndex, columns are
        asset tickers.  Rows must be sorted chronologically.
    window : int
        Length of the trailing estimation window in trading days (default 252,
        i.e., approximately one calendar year).
    semivar_benchmark : float
        Return threshold B used in the semi-covariance computation.
        Default 0.0 (paper's documented base case).
    """

    def __init__(
        self,
        log_returns_df: pd.DataFrame,
        window: int = 252,
        semivar_benchmark: float = 0.0,
    ) -> None:
        self.returns_df = log_returns_df
        self.window = window
        self.semivar_benchmark = semivar_benchmark
        self._cache: Dict[pd.Timestamp, Dict] = {}

    def get(self, date: pd.Timestamp) -> Dict:
        """
        Return cached optimiser inputs for ``date``, computing them if needed.

        Parameters
        ----------
        date : pd.Timestamp
            Query date; must exist in ``self.returns_df.index``.

        Returns
        -------
        dict with keys:
            ``mu``               – np.ndarray (n_assets,)
            ``sigma``            – np.ndarray (n_assets, n_assets)
            ``semi_cov``         – np.ndarray (n_assets, n_assets)
            ``scenario_returns`` – np.ndarray (window, n_assets)
            ``n_obs``            – int, actual number of observations used
        """
        if date not in self._cache:
            self._cache[date] = self._compute(date)
        return self._cache[date]

    def _compute(self, date: pd.Timestamp) -> Dict:
        """
        Compute optimiser inputs using the trailing window ending at ``date``.

        The window is ``[date - window + 1, date]`` in index-position space;
        if fewer than ``window`` observations are available (e.g., near the
        start of the series) all available rows are used.

        Parameters
        ----------
        date : pd.Timestamp
            End date of the estimation window.

        Returns
        -------
        dict
            See :meth:`get` for key descriptions.
        """
        loc = self.returns_df.index.get_loc(date)
        start_loc = max(0, loc - self.window + 1)
        window_returns = self.returns_df.iloc[start_loc : loc + 1].values  # (T, n)

        mu = estimate_expected_returns(window_returns)
        sigma = estimate_covariance(window_returns)
        semi_cov = estimate_semi_covariance(window_returns, self.semivar_benchmark)

        return {
            "mu": mu,
            "sigma": sigma,
            "semi_cov": semi_cov,
            "scenario_returns": window_returns,
            "n_obs": len(window_returns),
        }
