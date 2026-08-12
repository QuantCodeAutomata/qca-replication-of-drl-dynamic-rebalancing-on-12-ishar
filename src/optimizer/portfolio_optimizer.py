import logging

import cvxpy as cp
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def solve_mean_variance(
    mu: np.ndarray,
    sigma: np.ndarray,
    mu_target: float,
    n_assets: int,
    max_iter: int = 5000,
) -> Optional[np.ndarray]:
    """
    Solve the classical Markowitz mean-variance optimization problem.

    Formulation:
        min   w' Sigma w
        s.t.  sum(w) = 1        (fully invested)
              w >= 0             (long-only)
              mu' w >= mu_target (minimum return requirement)

    Parameters
    ----------
    mu : np.ndarray, shape (n_assets,)
        Vector of expected (mean) log returns.
    sigma : np.ndarray, shape (n_assets, n_assets)
        Covariance matrix of log returns.
    mu_target : float
        Minimum required portfolio expected return.
    n_assets : int
        Number of assets in the universe.
    max_iter : int
        Maximum solver iterations (passed to CLARABEL).

    Returns
    -------
    np.ndarray or None
        Optimal weight vector of shape (n_assets,), normalised to sum to 1,
        or None if the problem is infeasible or the solver fails.
    """
    try:
        # Sanitise sigma: NaN/Inf → 0, then enforce exact symmetry and PD-ness.
        # NaNs arise when np.cov is called with T<=1 (ddof=0 edge case).
        sigma_clean = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
        sigma_sym = (sigma_clean + sigma_clean.T) / 2.0
        # Add a small ridge so the matrix is strictly PD even if sigma was degenerate
        sigma_sym += 1e-8 * np.eye(n_assets)
        w = cp.Variable(n_assets)
        objective = cp.Minimize(cp.quad_form(w, sigma_sym))
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            mu @ w >= mu_target,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, max_iter=max_iter)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
            weights = np.maximum(w.value, 0)
            weights /= weights.sum()
            return weights
        else:
            logger.warning(f"MV solver status: {prob.status} for mu_target={mu_target:.6f}")
            return None
    except Exception as e:
        logger.warning(f"MV solver exception: {e}")
        return None


def estimate_semi_covariance(
    returns: np.ndarray,
    benchmark: float = 0.0,
) -> np.ndarray:
    """
    Estimate the semi-covariance matrix from historical returns.

    Only return observations that fall *below* the benchmark threshold B
    contribute to the dispersion measure:

        SCOV[i, j] = (1 / T) * sum_t  min(r_{it} - B, 0) * min(r_{jt} - B, 0)

    This captures downside co-movement exclusively.  The base-case benchmark
    is B = 0 (daily log-return of zero), matching the paper's documented choice.

    Parameters
    ----------
    returns : np.ndarray, shape (T, n_assets)
        Historical log return matrix (rows = time, columns = assets).
    benchmark : float
        Return threshold B below which returns are considered downside.
        Defaults to 0.0.

    Returns
    -------
    np.ndarray, shape (n_assets, n_assets)
        Semi-covariance matrix (positive semi-definite by construction).
    """
    # Sanitise before computing downside exceedances so that NaN/Inf in the
    # return matrix (e.g., from the first row of log-returns or missing prices)
    # do not propagate into the semi-covariance matrix.
    returns_clean = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    downside = np.minimum(returns_clean - benchmark, 0)  # (T, n_assets)
    T = returns_clean.shape[0]
    semi_cov = (downside.T @ downside) / max(T, 1)       # (n_assets, n_assets)
    return semi_cov


def solve_mean_semivariance(
    mu: np.ndarray,
    semi_cov: np.ndarray,
    mu_target: float,
    n_assets: int,
    max_iter: int = 5000,
) -> Optional[np.ndarray]:
    """
    Solve the mean–semi-variance optimisation problem.

    Replaces the full covariance matrix with the semi-covariance matrix SCOV,
    penalising only downside risk:

        min   w' SCOV w
        s.t.  sum(w) = 1
              w >= 0
              mu' w >= mu_target

    A small Tikhonov regularisation term (1e-8 * I) is added to SCOV before
    solving to guarantee strict positive-definiteness for the SOCP solver.

    Parameters
    ----------
    mu : np.ndarray, shape (n_assets,)
        Vector of expected log returns.
    semi_cov : np.ndarray, shape (n_assets, n_assets)
        Semi-covariance matrix (from :func:`estimate_semi_covariance`).
    mu_target : float
        Minimum required portfolio expected return.
    n_assets : int
        Number of assets in the universe.
    max_iter : int
        Maximum solver iterations.

    Returns
    -------
    np.ndarray or None
        Optimal weight vector normalised to sum to 1, or None on failure.
    """
    try:
        # Sanitise semi_cov: NaN/Inf → 0, then enforce exact symmetry and PD-ness.
        # semi_cov is symmetric by construction (X'X / T), but NaN values from
        # sparse downside periods or degenerate windows break np.allclose(P, P.T)
        # inside cp.quad_form.__init__, causing a ValueError that must be caught here.
        semi_cov_clean = np.nan_to_num(semi_cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov_reg = (semi_cov_clean + semi_cov_clean.T) / 2.0 + 1e-8 * np.eye(n_assets)
        w = cp.Variable(n_assets)
        objective = cp.Minimize(cp.quad_form(w, cov_reg))
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            mu @ w >= mu_target,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, max_iter=max_iter)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
            weights = np.maximum(w.value, 0)
            weights /= weights.sum()
            return weights
        else:
            logger.warning(f"SemiVar solver status: {prob.status}")
            return None
    except Exception as e:
        logger.warning(f"SemiVar solver exception: {e}")
        return None


def solve_mean_cvar(
    mu: np.ndarray,
    scenario_returns: np.ndarray,
    mu_target: float,
    n_assets: int,
    beta: float = 0.95,
    max_iter: int = 5000,
) -> Optional[np.ndarray]:
    """
    Minimise Conditional Value-at-Risk (CVaR) via the Rockafellar–Uryasev LP.

    The CVaR at confidence level beta is estimated as:

        CVaR_beta(w) = l + [1 / (q * (1 - beta))] * sum_k S_k

    where each scenario loss is  loss_k = -y_k' w  (negative portfolio return),
    and S_k = max(loss_k - l, 0) are the auxiliary excess-loss variables.

    Full LP formulation:
        min   l + [1 / (q * (1 - beta))] * sum_{k=1}^{q} S_k
        s.t.  S_k >= -y_k' w - l    for k = 1, ..., q   (excess losses)
              S_k >= 0               for k = 1, ..., q
              sum(w) = 1             (fully invested)
              w >= 0                 (long-only)
              mu' w >= mu_target     (return requirement)

    Parameters
    ----------
    mu : np.ndarray, shape (n_assets,)
        Vector of expected log returns.
    scenario_returns : np.ndarray, shape (q, n_assets)
        Historical return scenarios used to approximate the loss distribution.
        q is the number of trailing observations (typically 252).
    mu_target : float
        Minimum required portfolio expected return.
    n_assets : int
        Number of assets in the universe; must equal scenario_returns.shape[1].
    beta : float
        CVaR confidence level (default 0.95, i.e., 95 % CVaR).
    max_iter : int
        Maximum solver iterations.

    Returns
    -------
    np.ndarray or None
        Optimal weight vector normalised to sum to 1, or None on failure.
    """
    q, n = scenario_returns.shape
    assert n == n_assets, (
        f"scenario_returns has {n} assets but n_assets={n_assets}"
    )

    w = cp.Variable(n_assets)
    l = cp.Variable()       # VaR threshold (alpha in R-U notation)
    S = cp.Variable(q)      # Excess-loss auxiliary variables

    # Portfolio loss per scenario: loss_k = -y_k' w
    portfolio_losses = -scenario_returns @ w  # shape (q,)

    objective = cp.Minimize(l + (1.0 / (q * (1 - beta))) * cp.sum(S))
    constraints = [
        S >= portfolio_losses - l,   # S_k >= loss_k - l
        S >= 0,
        cp.sum(w) == 1,
        w >= 0,
        mu @ w >= mu_target,
    ]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, max_iter=max_iter)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
            weights = np.maximum(w.value, 0)
            weights /= weights.sum()
            return weights
        else:
            logger.warning(f"CVaR solver status: {prob.status}")
            return None
    except Exception as e:
        logger.warning(f"CVaR solver exception: {e}")
        return None
