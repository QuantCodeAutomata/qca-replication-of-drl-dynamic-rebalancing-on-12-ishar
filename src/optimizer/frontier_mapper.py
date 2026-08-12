import logging

import numpy as np

logger = logging.getLogger(__name__)


def build_target_return_grid(
    mu: np.ndarray,
    n_levels: int = 100,
) -> np.ndarray:
    """
    Build an ordered grid of target-return levels for the discrete action space.

    The DRL agent selects a risk-appetite action lambda_t in {1, ..., 100}.
    This function creates the corresponding 100-point target-return grid by
    linearly interpolating between the minimum and maximum expected asset returns:

        grid[k] = mu_min + (k / (n_levels - 1)) * (mu_max - mu_min),  k = 0, ..., n_levels-1

    Endpoint convention:
        level 1  (index 0) -> mu_min  (most conservative / minimum-return portfolio)
        level 100 (index 99) -> mu_max (most aggressive / maximum-return portfolio)

    Parameters
    ----------
    mu : np.ndarray, shape (n_assets,)
        Vector of expected log returns for each asset.
    n_levels : int
        Number of discrete risk levels (default 100, matching the paper).

    Returns
    -------
    np.ndarray, shape (n_levels,)
        Monotonically non-decreasing array of target return values.
    """
    mu_min = np.min(mu)
    mu_max = np.max(mu)

    if mu_min >= mu_max:
        # Edge case: all assets share the same expected return
        logger.warning(
            "All expected returns are equal (mu_min == mu_max). "
            "Returning a constant target-return grid."
        )
        return np.full(n_levels, mu_min)

    return np.linspace(mu_min, mu_max, n_levels)


def lambda_to_target_return(
    lambda_t: int,
    target_grid: np.ndarray,
) -> float:
    """
    Map a discrete agent action lambda_t to a scalar target return.

    The action space is {1, ..., 100} and maps directly onto the pre-built
    target-return grid via 1-based indexing:

        mu_target = target_grid[lambda_t - 1]

    Parameters
    ----------
    lambda_t : int
        Discrete risk-level action chosen by the DRL agent; must be in [1, 100].
    target_grid : np.ndarray, shape (n_levels,)
        Pre-computed target-return grid from :func:`build_target_return_grid`.

    Returns
    -------
    float
        Scalar target return corresponding to the selected action.

    Raises
    ------
    AssertionError
        If lambda_t is outside the valid range [1, 100].
    """
    assert 1 <= lambda_t <= 100, f"lambda_t must be in [1, 100], got {lambda_t}"
    return float(target_grid[lambda_t - 1])


def find_feasible_target(
    mu_target: float,
    mu: np.ndarray,
    sigma: np.ndarray,
    fallback_fraction: float = 0.99,
) -> float:
    """
    Project an infeasible target return to the nearest feasible value.

    For a long-only portfolio the maximum achievable expected return equals
    max(mu), since allocating 100 % to the highest-return asset is the extreme
    point of the feasible set.  If the requested target exceeds this bound it
    is projected to ``fallback_fraction * max(mu)`` to ensure the downstream
    optimiser always receives a feasible target.

    Parameters
    ----------
    mu_target : float
        Requested portfolio expected return from the action grid.
    mu : np.ndarray, shape (n_assets,)
        Vector of expected log returns (used to determine the feasibility bound).
    sigma : np.ndarray, shape (n_assets, n_assets)
        Covariance matrix (accepted for API consistency; not used in the
        long-only feasibility check, which depends only on mu).
    fallback_fraction : float
        Fraction of max(mu) used as the projected target when infeasible.
        Must be in (0, 1].  Default 0.99.

    Returns
    -------
    float
        Feasible target return: mu_target if already feasible, otherwise
        fallback_fraction * max(mu).
    """
    max_achievable = np.max(mu)
    if mu_target > max_achievable:
        projected = fallback_fraction * max_achievable
        logger.warning(
            f"mu_target={mu_target:.6f} exceeds max achievable return "
            f"{max_achievable:.6f}. Projecting to {projected:.6f}."
        )
        return projected
    return mu_target
