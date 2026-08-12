"""
Gymnasium environment for DRL dynamic portfolio rebalancing (Yu & Chang 2025).

At each rebalance date the RL agent selects:
  - lambda_t ∈ {1, ..., 100}: risk-appetite level that indexes the efficient frontier.
  - h_t ∈ {5, ..., 60}: hold horizon in trading days.

The environment queries a portfolio optimiser to obtain allocation weights and
computes the reward as the realised log portfolio return over the hold period.
"""

import logging
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class PortfolioEnv(gym.Env):
    """
    Gymnasium environment for DRL portfolio rebalancing (Yu & Chang 2025).

    Action space: MultiDiscrete([100, 56])
        action[0]: lambda_t in {0,...,99} mapped to {1,...,100}
        action[1]: h_t in {0,...,55} mapped to horizon {5,...,60}

    Observation space: Box of shape (n_features * n_assets * lookback,)
        Flattened from (n_features, n_assets, lookback) tensor

    Parameters
    ----------
    price_returns : np.ndarray, shape (T, n_assets)
        Daily log returns matrix.
    observation_tensors : np.ndarray, shape (T, n_features * n_assets * lookback)
        Pre-computed, per-step observation feature tensors (may be multi-dim;
        they are flattened automatically in :meth:`_get_obs`).
    optimizer_fn : Callable[[dict, float], np.ndarray]
        Portfolio optimiser: ``optimizer_fn(estimates, mu_target) -> weights``.
    estimator_fn : Callable[[int], dict]
        Return/risk estimator: ``estimator_fn(date_idx) -> {'mu', 'sigma', ...}``.
    n_assets : int
        Number of investable assets.
    n_features : int
        Number of observation features per asset per time step (default 34).
    lookback : int
        Look-back window used to construct the observation tensor (default 252).
    initial_weights : np.ndarray or None
        Starting portfolio weights.  Defaults to equal-weight.
    initial_value : float
        Starting portfolio value (default 1 000).
    min_horizon : int
        Minimum hold horizon in trading days (default 5).
    max_horizon : int
        Maximum hold horizon in trading days (default 60).
    n_lambda : int
        Number of discrete risk-appetite levels (default 100).
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        price_returns: np.ndarray,
        observation_tensors: np.ndarray,
        optimizer_fn: Callable,
        estimator_fn: Callable,
        n_assets: int,
        n_features: int = 34,
        lookback: int = 252,
        initial_weights: Optional[np.ndarray] = None,
        initial_value: float = 1000.0,
        min_horizon: int = 5,
        max_horizon: int = 60,
        n_lambda: int = 100,
    ):
        super().__init__()
        self.price_returns = price_returns          # (T, n_assets)
        self.obs_tensors = observation_tensors      # (T, ...) pre-computed
        self.optimizer_fn = optimizer_fn
        self.estimator_fn = estimator_fn
        self.n_assets = n_assets
        self.n_features = n_features
        self.lookback = lookback
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.n_lambda = n_lambda
        self.initial_value = initial_value
        self.n_horizon_choices = max_horizon - min_horizon + 1  # 56

        if initial_weights is None:
            self.initial_weights = np.ones(n_assets) / n_assets
        else:
            self.initial_weights = initial_weights

        # Action space: [lambda_idx (0-99), horizon_idx (0-55)]
        self.action_space = spaces.MultiDiscrete([n_lambda, self.n_horizon_choices])

        # Observation space: flattened state tensor
        obs_dim = n_features * n_assets * lookback
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        self.T = len(price_returns)
        # Cache optimizer results keyed by (step_idx, lambda_t) to avoid
        # redundant CVXPY solves across repeated training episodes.
        self._opt_cache: Dict[tuple, np.ndarray] = {}
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Initialise (or reinitialise) all mutable episode state."""
        self.current_step = 0
        self.portfolio_value = self.initial_value
        self.current_weights = self.initial_weights.copy()
        self.done = False

    def _get_obs(self, step_idx: int) -> np.ndarray:
        """Return the flattened observation tensor for *step_idx*.

        Parameters
        ----------
        step_idx : int
            Time index into ``obs_tensors``.  Clamped to the last available
            index when out-of-bounds.

        Returns
        -------
        np.ndarray, shape (n_features * n_assets * lookback,), dtype float32
        """
        step_idx = min(int(step_idx), len(self.obs_tensors) - 1)
        obs = self.obs_tensors[step_idx]
        if obs.ndim > 1:
            obs = obs.flatten()
        # Replace NaN/inf produced by z-score warm-up with 0 so the policy
        # network never receives invalid floating-point values.
        obs = np.nan_to_num(obs.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Reset the environment to the start of a new episode.

        Parameters
        ----------
        seed : int or None
            RNG seed forwarded to the base class.
        options : dict or None
            Unused; accepted for API compliance.

        Returns
        -------
        observation : np.ndarray
            Initial observation at time step 0.
        info : dict
            Empty info dict.
        """
        super().reset(seed=seed)
        self._reset_state()
        obs = self._get_obs(0)
        return obs, {}

    def step(self, action: np.ndarray):
        """Execute one rebalancing step.

        Decodes the agent's action, calls the portfolio optimiser for the
        current date, holds the resulting portfolio for *h_t* days, and
        returns the log-return reward.

        Parameters
        ----------
        action : array-like of length 2
            ``action[0]``: lambda index in {0,...,99}.
            ``action[1]``: horizon index in {0,...,55}.

        Returns
        -------
        observation : np.ndarray
            Observation at the new current step.
        reward : float
            Log portfolio return over the hold period, clipped to [-10, 10].
        terminated : bool
            True when the episode reaches the end of the price series.
        truncated : bool
            Always False (no external time limit).
        info : dict
            Diagnostic information including decoded action values and weights.
        """
        if self.done:
            obs = self._get_obs(self.current_step)
            return obs, 0.0, True, False, {}

        lambda_idx = int(action[0])   # 0-99
        horizon_idx = int(action[1])  # 0-55
        lambda_t = lambda_idx + 1     # 1-100
        h_t = horizon_idx + self.min_horizon  # 5-60

        # Obtain distributional estimates for the current date
        estimates = self.estimator_fn(self.current_step)
        mu = estimates['mu']

        # Map discrete action to a scalar target return on the efficient frontier
        from src.optimizer.frontier_mapper import (
            build_target_return_grid,
            find_feasible_target,
            lambda_to_target_return,
        )
        target_grid = build_target_return_grid(mu, self.n_lambda)
        mu_target = lambda_to_target_return(lambda_t, target_grid)
        mu_target = find_feasible_target(mu_target, mu, estimates.get('sigma'))

        # Cache optimizer results by (step, lambda) to avoid redundant CVXPY solves
        cache_key = (self.current_step, lambda_t)
        if cache_key in self._opt_cache:
            new_weights = self._opt_cache[cache_key]
        else:
            new_weights = self.optimizer_fn(estimates, mu_target)
            if new_weights is None:
                new_weights = np.ones(self.n_assets) / self.n_assets
                logger.warning('Optimizer failed, using equal weight fallback')
            self._opt_cache[cache_key] = new_weights

        self.current_weights = new_weights

        # Accumulate portfolio return over the hold period
        start_pv = self.portfolio_value
        end_step = min(self.current_step + h_t, self.T - 1)

        if end_step > self.current_step:
            hold_returns = self.price_returns[self.current_step:end_step]  # (h_t, n_assets)
            portfolio_log_ret = float(np.sum(hold_returns @ self.current_weights))
            self.portfolio_value = start_pv * np.exp(portfolio_log_ret)

        # Reward: log(PV_end / PV_start)
        reward = np.log(self.portfolio_value / start_pv) if start_pv > 0 else 0.0
        reward = float(np.clip(reward, -10.0, 10.0))

        # Advance the time pointer
        self.current_step = end_step
        terminated = self.current_step >= self.T - 1
        self.done = terminated

        obs = self._get_obs(min(self.current_step, len(self.obs_tensors) - 1))

        info = {
            'lambda_t': lambda_t,
            'h_t': h_t,
            'mu_target': mu_target,
            'weights': new_weights.tolist(),
            'portfolio_value': self.portfolio_value,
        }
        return obs, reward, terminated, False, info
