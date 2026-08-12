"""
Agent wrapper utilities for DRL portfolio rebalancing (Yu & Chang 2025).

Provides thin wrappers around Stable-Baselines3 (SB3) PPO and A2C algorithms
for training, evaluation, and ensemble model selection on the
:class:`~src.rl_env.portfolio_env.PortfolioEnv` Gymnasium environment.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .portfolio_env import PortfolioEnv

logger = logging.getLogger(__name__)

SUPPORTED_ALGOS: Dict[str, Any] = {'PPO': PPO, 'A2C': A2C}
SUPPORTED_RISK_MODELS: List[str] = ['variance', 'semivariance', 'cvar']


def make_env_factory(env_kwargs: dict):
    """Create a zero-argument factory function that instantiates a PortfolioEnv.

    Suitable for use with :class:`~stable_baselines3.common.vec_env.DummyVecEnv`.

    Parameters
    ----------
    env_kwargs : dict
        Keyword arguments forwarded verbatim to :class:`PortfolioEnv`.

    Returns
    -------
    Callable[[], PortfolioEnv]
        A callable that returns a fresh :class:`PortfolioEnv` instance each time
        it is invoked.
    """
    def _init() -> PortfolioEnv:
        return PortfolioEnv(**env_kwargs)
    return _init


def train_agent(
    algo: str,
    env_kwargs: dict,
    total_timesteps: int = 50000,
    seed: int = 42,
    policy_kwargs: Optional[dict] = None,
    hyperparams: Optional[dict] = None,
) -> Any:
    """Train a PPO or A2C agent on the portfolio environment.

    The environment is wrapped in a single-process
    :class:`~stable_baselines3.common.vec_env.DummyVecEnv` as required by SB3.
    Default network architecture is a two-layer MLP with [128, 64] hidden units,
    matching a compact policy suitable for moderate observation dimensions.

    Parameters
    ----------
    algo : str
        Algorithm identifier: ``'PPO'`` or ``'A2C'``.
    env_kwargs : dict
        Keyword arguments forwarded to :class:`PortfolioEnv`.
    total_timesteps : int
        Total environment steps to train for.  The paper trains for ~10 000
        episodes; the default here provides a reasonable baseline.
    seed : int
        Random seed for reproducibility across the model and environment.
    policy_kwargs : dict or None
        SB3 policy kwargs (e.g. ``{'net_arch': [128, 64]}``).  Defaults to
        ``{'net_arch': [128, 64]}`` when ``None``.
    hyperparams : dict or None
        Additional SB3 constructor kwargs that override the defaults (e.g.
        ``{'learning_rate': 1e-3}``).

    Returns
    -------
    trained model
        A fitted SB3 ``PPO`` or ``A2C`` model instance.

    Raises
    ------
    AssertionError
        If *algo* is not in :data:`SUPPORTED_ALGOS`.
    """
    assert algo in SUPPORTED_ALGOS, f'Unsupported algo: {algo}. Choose from {list(SUPPORTED_ALGOS)}'
    AlgoClass = SUPPORTED_ALGOS[algo]

    env = DummyVecEnv([make_env_factory(env_kwargs)])

    if policy_kwargs is None:
        policy_kwargs = {'net_arch': [128, 64]}

    # PPO n_steps must be < total_timesteps so that at least one gradient update
    # is performed.  Cap at 256 so the rollout buffer always fills quickly enough
    # for meaningful policy updates regardless of the timestep budget.
    ppo_n_steps = min(256, max(16, total_timesteps // 4))
    default_hp: Dict[str, Any] = {
        'learning_rate': 3e-4,
        'n_steps': ppo_n_steps if algo == 'PPO' else 5,
        'batch_size': min(64, ppo_n_steps) if algo == 'PPO' else None,
        'verbose': 0,
        'seed': seed,
        'policy_kwargs': policy_kwargs,
    }
    if algo == 'A2C':
        # A2C does not use batch_size; remove both to let SB3 use its defaults
        default_hp.pop('batch_size', None)
        default_hp.pop('n_steps', None)
    if hyperparams:
        default_hp.update(hyperparams)

    model = AlgoClass('MlpPolicy', env, **default_hp)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_agent(
    model: Any,
    env_kwargs: dict,
    deterministic: bool = True,
    seed: int = 0,
) -> Tuple[float, List[Dict]]:
    """Evaluate a trained agent on the portfolio environment for one episode.

    Rolls out the policy deterministically (or stochastically) from the
    beginning of the price series to termination and computes the annualised
    Sharpe ratio from the sequence of portfolio values.

    Parameters
    ----------
    model : SB3 model
        A trained ``PPO`` or ``A2C`` model returned by :func:`train_agent`.
    env_kwargs : dict
        Keyword arguments forwarded to :class:`PortfolioEnv`.
    deterministic : bool
        Whether to use deterministic (greedy) action selection.
    seed : int
        Seed passed to :meth:`PortfolioEnv.reset` for reproducible evaluation.

    Returns
    -------
    sharpe_ratio : float
        Annualised Sharpe ratio computed from per-step log portfolio returns.
        Returns 0.0 when fewer than two portfolio values are recorded or when
        the return series has near-zero variance.
    episode_log : list of dict
        Per-step ``info`` dicts augmented with the step reward under key
        ``'reward'``.
    """
    env = PortfolioEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    done = False
    episode_log: List[Dict] = []
    portfolio_values: List[float] = [env_kwargs.get('initial_value', 1000.0)]

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        info['reward'] = reward
        episode_log.append(info)
        portfolio_values.append(info['portfolio_value'])

    # Annualised Sharpe from per-step log returns (steps ≠ single days, but
    # this is the standard proxy used for model selection in the paper)
    pv = np.array(portfolio_values)
    if len(pv) > 1:
        daily_ret = np.diff(np.log(pv + 1e-10))
        if daily_ret.std() > 1e-10:
            sharpe = float(np.mean(daily_ret) / daily_ret.std() * np.sqrt(252))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    return sharpe, episode_log


def run_validation_runs(
    model: Any,
    env_kwargs: dict,
    n_runs: int = 100,
    deterministic: bool = True,
) -> Tuple[float, List[float]]:
    """Run *n_runs* evaluation episodes and return the median Sharpe ratio.

    Each run uses a distinct integer seed (0, 1, …, n_runs-1) so that any
    stochastic components of the environment are sampled reproducibly across
    different calls.

    Parameters
    ----------
    model : SB3 model
        Trained agent to evaluate.
    env_kwargs : dict
        Keyword arguments forwarded to :class:`PortfolioEnv`.
    n_runs : int
        Number of independent evaluation episodes (default 100).
    deterministic : bool
        Whether to use deterministic action selection in each run.

    Returns
    -------
    median_sharpe : float
        Median Sharpe ratio across all *n_runs* episodes.
    sharpes : list of float
        Individual Sharpe ratio for each episode.
    """
    sharpes: List[float] = []
    for run_idx in range(n_runs):
        sharpe, _ = evaluate_agent(model, env_kwargs, deterministic=deterministic, seed=run_idx)
        sharpes.append(sharpe)
    median_sharpe = float(np.median(sharpes))
    return median_sharpe, sharpes


def select_ensemble_model(
    models_with_sharpes: Dict[str, Tuple[Any, float]],
) -> Tuple[str, Any]:
    """Select the best model by highest median validation Sharpe ratio.

    When multiple models share the same Sharpe ratio, ties are broken
    alphabetically by model name (deterministic, ordering-independent).

    Parameters
    ----------
    models_with_sharpes : dict
        Mapping ``{model_name: (trained_model, median_sharpe)}``.

    Returns
    -------
    best_name : str
        Name of the selected model.
    best_model : SB3 model
        The corresponding trained model object.
    """
    best_name = max(
        sorted(models_with_sharpes.keys()),  # sort first for deterministic tie-breaking
        key=lambda k: models_with_sharpes[k][1],
    )
    return best_name, models_with_sharpes[best_name][0]
