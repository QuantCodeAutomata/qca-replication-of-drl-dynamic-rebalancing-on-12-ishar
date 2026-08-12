"""
Tests for src.rl_env.portfolio_env.PortfolioEnv

All tests use purely synthetic data; no real market data is required.
A mock estimator and mock optimizer are provided so the environment can
be exercised without a real covariance estimator or convex solver.
"""

import numpy as np
import pytest
from gymnasium import spaces

from src.rl_env.portfolio_env import PortfolioEnv

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

N_ASSETS = 5
T = 100
LOOKBACK = 20
N_FEATURES = 5
OBS_DIM = N_FEATURES * N_ASSETS * LOOKBACK  # 500


def make_synthetic_env(n_assets: int = N_ASSETS, T: int = T, lookback: int = LOOKBACK) -> PortfolioEnv:
    """Build a PortfolioEnv backed by fully synthetic data.

    Parameters
    ----------
    n_assets : int
        Number of investable assets.
    T : int
        Length of the price-return series (trading days).
    lookback : int
        Feature look-back window (determines obs_dim).

    Returns
    -------
    PortfolioEnv
    """
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((T, n_assets)) * 0.01

    obs_dim = N_FEATURES * n_assets * lookback
    obs_tensors = rng.standard_normal((T, obs_dim)).astype(np.float32)

    def mock_estimator(step_idx: int) -> dict:
        """Return synthetic distributional estimates for any date index."""
        local_rng = np.random.default_rng(step_idx)
        mu = local_rng.standard_normal(n_assets) * 0.001
        return {
            'mu': mu,
            'sigma': np.eye(n_assets) * 0.0001,
            'semi_cov': np.eye(n_assets) * 0.00005,
            'scenario_returns': returns[:min(step_idx + 1, T)],
        }

    def mock_optimizer(estimates: dict, mu_target: float) -> np.ndarray:
        """Return an equal-weight portfolio regardless of estimates."""
        return np.ones(n_assets) / n_assets

    env = PortfolioEnv(
        price_returns=returns,
        observation_tensors=obs_tensors,
        optimizer_fn=mock_optimizer,
        estimator_fn=mock_estimator,
        n_assets=n_assets,
        n_features=N_FEATURES,
        lookback=lookback,
        initial_value=1000.0,
    )
    return env


@pytest.fixture
def env() -> PortfolioEnv:
    return make_synthetic_env()


# ---------------------------------------------------------------------------
# Action / observation space shape tests
# ---------------------------------------------------------------------------


def test_action_space_shape(env: PortfolioEnv) -> None:
    """Action space must be MultiDiscrete([100, 56])."""
    assert isinstance(env.action_space, spaces.MultiDiscrete)
    np.testing.assert_array_equal(env.action_space.nvec, [100, 56])


def test_observation_space(env: PortfolioEnv) -> None:
    """Observation space must be a Box of shape (n_features*n_assets*lookback,)."""
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.observation_space.dtype == np.float32


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_env_reset(env: PortfolioEnv) -> None:
    """reset() must return a valid (obs, info) 2-tuple."""
    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_reset_restores_initial_state(env: PortfolioEnv) -> None:
    """A second reset after stepping must restore the initial portfolio value."""
    env.reset(seed=0)
    env.step(env.action_space.sample())

    obs, _ = env.reset(seed=0)
    assert env.portfolio_value == env.initial_value
    assert env.current_step == 0
    assert obs.shape == (OBS_DIM,)


# ---------------------------------------------------------------------------
# Step output structure and types
# ---------------------------------------------------------------------------


def test_env_step(env: PortfolioEnv) -> None:
    """step() must return a valid 5-tuple with correct types."""
    env.reset(seed=0)
    action = env.action_space.sample()
    result = env.step(action)

    assert len(result) == 5, "step() must return a 5-tuple (obs, reward, terminated, truncated, info)"
    obs, reward, terminated, truncated, info = result

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32

    assert isinstance(reward, float)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)


def test_truncated_always_false(env: PortfolioEnv) -> None:
    """truncated must always be False (no external time limit in this env)."""
    env.reset(seed=0)
    for _ in range(5):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        assert truncated is False or truncated == False  # noqa: E712
        if terminated:
            break


# ---------------------------------------------------------------------------
# Portfolio weights validity
# ---------------------------------------------------------------------------


def test_weights_valid(env: PortfolioEnv) -> None:
    """Weights from the optimizer must be non-negative and sum to 1."""
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())

    weights = np.array(info['weights'])
    assert weights.ndim == 1
    assert len(weights) == N_ASSETS
    assert np.all(weights >= -1e-8), "All weights must be non-negative"
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Reward validity
# ---------------------------------------------------------------------------


def test_reward_finite(env: PortfolioEnv) -> None:
    """Reward must be a finite float (no NaN or Inf)."""
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(env.action_space.sample())

    assert np.isfinite(reward), f"Reward is not finite: {reward}"


def test_reward_bounded(env: PortfolioEnv) -> None:
    """Reward must be within the clipping bounds [-10, 10]."""
    env.reset(seed=0)
    for _ in range(10):
        action = env.action_space.sample()
        _, reward, terminated, _, _ = env.step(action)
        assert -10.0 <= reward <= 10.0, f"Reward {reward} outside [-10, 10]"
        if terminated:
            break


# ---------------------------------------------------------------------------
# Action decoding bounds
# ---------------------------------------------------------------------------


def test_horizon_bounds(env: PortfolioEnv) -> None:
    """h_t decoded from action[1] must lie in [5, 60] for all valid actions."""
    env.reset(seed=0)
    for horizon_idx in [0, 27, 55]:  # boundary and mid values
        env.reset(seed=0)
        action = np.array([0, horizon_idx])
        _, _, _, _, info = env.step(action)
        h_t = info['h_t']
        assert 5 <= h_t <= 60, f"h_t={h_t} outside [5, 60] for horizon_idx={horizon_idx}"


def test_lambda_bounds(env: PortfolioEnv) -> None:
    """lambda_t decoded from action[0] must lie in [1, 100] for all valid actions."""
    env.reset(seed=0)
    for lambda_idx in [0, 50, 99]:  # boundary and mid values
        env.reset(seed=0)
        action = np.array([lambda_idx, 0])
        _, _, _, _, info = env.step(action)
        lambda_t = info['lambda_t']
        assert 1 <= lambda_t <= 100, f"lambda_t={lambda_t} outside [1, 100] for lambda_idx={lambda_idx}"


def test_horizon_decoding_correctness(env: PortfolioEnv) -> None:
    """h_t must equal action[1] + min_horizon exactly."""
    env.reset(seed=0)
    for horizon_idx in range(0, 56, 5):
        env.reset(seed=0)
        action = np.array([0, horizon_idx])
        _, _, _, _, info = env.step(action)
        expected_h = horizon_idx + env.min_horizon
        assert info['h_t'] == expected_h


def test_lambda_decoding_correctness(env: PortfolioEnv) -> None:
    """lambda_t must equal action[0] + 1 exactly."""
    env.reset(seed=0)
    for lambda_idx in range(0, 100, 10):
        env.reset(seed=0)
        action = np.array([lambda_idx, 0])
        _, _, _, _, info = env.step(action)
        assert info['lambda_t'] == lambda_idx + 1


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------


def test_episode_terminates(env: PortfolioEnv) -> None:
    """An episode must terminate within a finite number of steps."""
    env.reset(seed=0)
    max_steps = 1000
    terminated = False

    for step_count in range(max_steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    assert terminated or truncated, (
        f"Episode did not terminate after {max_steps} steps "
        f"(T={env.T}, current_step={env.current_step})"
    )


def test_step_after_done_returns_done(env: PortfolioEnv) -> None:
    """Stepping after the episode is done must keep returning terminated=True."""
    env.reset(seed=0)
    env.done = True  # force the env into a terminal state

    obs, reward, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert reward == 0.0


# ---------------------------------------------------------------------------
# Portfolio value monotonicity sanity check
# ---------------------------------------------------------------------------


def test_portfolio_value_positive(env: PortfolioEnv) -> None:
    """Portfolio value must remain strictly positive throughout an episode."""
    env.reset(seed=0)
    for _ in range(20):
        action = env.action_space.sample()
        _, _, terminated, _, info = env.step(action)
        assert info['portfolio_value'] > 0, "Portfolio value must be positive"
        if terminated:
            break


# ---------------------------------------------------------------------------
# Info dict completeness
# ---------------------------------------------------------------------------


def test_info_keys_present(env: PortfolioEnv) -> None:
    """Info dict must contain all required keys after each step."""
    required_keys = {'lambda_t', 'h_t', 'mu_target', 'weights', 'portfolio_value'}
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert required_keys.issubset(info.keys()), (
        f"Missing keys: {required_keys - info.keys()}"
    )
