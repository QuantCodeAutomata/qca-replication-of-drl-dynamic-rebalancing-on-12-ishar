import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def compute_performance_metrics(
    daily_returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0,
    mar: float = 0.0,
    trading_days: int = 252,
    cvar_beta: float = 0.95,
) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics as per Yu & Chang (2025).

    All return inputs are expected to be daily log returns.  The function
    always computes the core set of stand-alone metrics; if ``benchmark_returns``
    is supplied and has the same length as ``daily_returns``, a second set of
    relative metrics is also computed.

    Core metrics
    ------------
    annualized_return : float
        Geometric annualised return: ``exp(mean_log_ret * 252) - 1``.
    annualized_volatility : float
        Annualised standard deviation: ``std(r) * sqrt(252)``.
    var_95 : float
        Historical 95 % Value-at-Risk (positive number, i.e. the loss).
    cvar_95 : float
        Historical 95 % Conditional VaR: mean of returns at or below the VaR
        threshold (expressed as a positive loss figure).
    max_drawdown : float
        Peak-to-trough drawdown on the cumulative NAV curve; always ≤ 0.
    sharpe_ratio : float
        Annualised Sharpe ratio with ``risk_free_rate = 0``.
    sortino_ratio : float
        Annualised Sortino ratio using ``mar`` as the minimum acceptable return.

    Relative metrics (requires ``benchmark_returns``)
    --------------------------------------------------
    beta : float
        Systematic exposure to the benchmark.
    tracking_error : float
        Annualised standard deviation of active returns.
    information_ratio : float
        Annualised active return divided by tracking error.
    upside_capture : float
        Portfolio mean return on benchmark-up days divided by benchmark mean.
    downside_capture : float
        Portfolio mean return on benchmark-down days divided by benchmark mean.

    Parameters
    ----------
    daily_returns : np.ndarray
        Array of daily log portfolio returns.
    benchmark_returns : np.ndarray, optional
        Array of daily log benchmark returns of the same length.
    risk_free_rate : float
        Daily risk-free rate used in the Sharpe ratio (default 0).
    mar : float
        Minimum acceptable return used in the Sortino denominator (default 0).
    trading_days : int
        Number of trading days per year for annualisation (default 252).
    cvar_beta : float
        CVaR confidence level; not yet used (VaR/CVaR are fixed at 95 %).

    Returns
    -------
    Dict[str, float]
        Dictionary of metric names to values.  Empty dict if fewer than 2
        observations are provided.
    """
    if len(daily_returns) < 2:
        return {}

    r = np.array(daily_returns, dtype=float)
    n = len(r)

    # ── Annualised return ────────────────────────────────────────────────────
    total_log_return = np.sum(r)
    ann_return = float(np.exp(total_log_return * trading_days / n) - 1) if n > 0 else 0.0

    # ── Volatility ───────────────────────────────────────────────────────────
    ann_vol = float(r.std() * np.sqrt(trading_days))

    # ── VaR and CVaR at 95 % ────────────────────────────────────────────────
    var_95 = float(-np.percentile(r, 5))
    cvar_mask = r <= -var_95
    cvar_95 = float(-r[cvar_mask].mean()) if cvar_mask.sum() > 0 else var_95

    # ── Maximum drawdown ─────────────────────────────────────────────────────
    cum_returns = np.exp(np.cumsum(r))
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_dd = float(np.min(drawdowns))

    # ── Sharpe ratio (rf = 0) ────────────────────────────────────────────────
    _vol = float(r.std())
    _mean = float(r.mean())
    if _vol > 1e-10:
        sharpe = float(_mean * trading_days / (_vol * np.sqrt(trading_days)))
    elif _mean > 1e-12:
        sharpe = 1e6   # effectively infinite positive Sharpe for constant positive returns
    elif _mean < -1e-12:
        sharpe = -1e6  # effectively infinite negative Sharpe for constant negative returns
    else:
        sharpe = 0.0

    # ── Sortino ratio (mar = 0) ──────────────────────────────────────────────
    downside_returns = r[r < mar]
    if len(downside_returns) > 1:
        downside_dev = float(downside_returns.std() * np.sqrt(trading_days))
    else:
        downside_dev = 1e-10
    sortino = float(r.mean() * trading_days / downside_dev) if downside_dev > 1e-10 else 0.0

    metrics: Dict[str, float] = {
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
    }

    # ── Relative metrics ─────────────────────────────────────────────────────
    if benchmark_returns is not None and len(benchmark_returns) == len(r):
        b = np.array(benchmark_returns, dtype=float)

        # Beta
        cov_matrix = np.cov(r, b)
        beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] > 1e-10 else 1.0

        # Tracking error and information ratio
        active_returns = r - b
        te = float(active_returns.std() * np.sqrt(trading_days))
        ir = float((active_returns.mean() * trading_days) / te) if te > 1e-10 else 0.0

        # Upside / downside capture ratios
        up_mask = b > 0
        down_mask = b < 0
        if up_mask.sum() > 0 and b[up_mask].mean() != 0:
            up_capture = float(r[up_mask].mean() / b[up_mask].mean())
        else:
            up_capture = 1.0
        if down_mask.sum() > 0 and b[down_mask].mean() != 0:
            down_capture = float(r[down_mask].mean() / b[down_mask].mean())
        else:
            down_capture = 1.0

        metrics.update({
            'beta': beta,
            'tracking_error': te,
            'information_ratio': ir,
            'upside_capture': up_capture,
            'downside_capture': down_capture,
        })

    return metrics


def compute_turnover(
    weights_series: List[np.ndarray],
    rebalance_dates: list,
    n_rebalances: int,
    holding_periods: list,
    transaction_cost_bps: float = 5.0,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    Compute portfolio turnover and implied transaction-cost drag.

    Turnover at each rebalance is defined as half the sum of absolute weight
    changes (one-way turnover), consistent with the standard convention that
    buying one asset and selling another each count once.

    Parameters
    ----------
    weights_series : List[np.ndarray]
        Sequence of weight vectors at each rebalance point; each element has
        shape ``(n_assets,)`` and sums to 1.
    rebalance_dates : list
        Dates corresponding to each entry in ``weights_series``.
    n_rebalances : int
        Total number of rebalance events (used for annualisation).
    holding_periods : list
        Number of trading days between consecutive rebalances; used to compute
        the fraction of a year covered by the back-test.
    transaction_cost_bps : float
        Round-trip transaction cost in basis points applied to the traded
        notional.  Default 5 bps (0.05 %).
    trading_days : int
        Number of trading days per year for annualisation.  Default 252.

    Returns
    -------
    Dict[str, float]
        ``n_rebalances``  – number of rebalance events.
        ``avg_turnover``  – mean one-way turnover per rebalance.
        ``ann_turnover``  – turnover annualised over the back-test horizon.
        ``cost_drag_ann`` – annualised cost drag from transaction costs.
    """
    if len(weights_series) < 2:
        return {
            'n_rebalances': 0,
            'avg_turnover': 0.0,
            'ann_turnover': 0.0,
            'cost_drag_ann': 0.0,
        }

    turnovers = []
    for i in range(1, len(weights_series)):
        diff = np.abs(weights_series[i] - weights_series[i - 1])
        turnovers.append(float(diff.sum() / 2))  # One-way turnover

    avg_turnover = float(np.mean(turnovers))
    total_days = sum(holding_periods) if holding_periods else trading_days
    years = total_days / trading_days
    ann_turnover = float(avg_turnover * n_rebalances / years) if years > 0 else 0.0

    # Round-trip cost: buy + sell → multiply one-way turnover by 2
    cost_per_rebalance = avg_turnover * 2 * transaction_cost_bps / 10_000
    cost_drag_ann = float(cost_per_rebalance * n_rebalances / years) if years > 0 else 0.0

    return {
        'n_rebalances': n_rebalances,
        'avg_turnover': avg_turnover,
        'ann_turnover': ann_turnover,
        'cost_drag_ann': cost_drag_ann,
    }
