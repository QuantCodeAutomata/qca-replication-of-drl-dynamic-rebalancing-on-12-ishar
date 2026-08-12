"""Preprocessing utilities: log returns, OHLC adjustment, and calendar alignment."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_log_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from adjusted close prices.

    Uses the formula :math:`r_t = \\ln(P_t / P_{t-1})`.

    Args:
        adj_close: DataFrame of adjusted close prices (rows = dates,
            columns = tickers).

    Returns:
        DataFrame of daily log returns with the first row dropped (NaN
        from the initial shift).
    """
    return np.log(adj_close / adj_close.shift(1)).dropna()


def compute_adjusted_ohlc(
    raw_open: pd.DataFrame,
    raw_high: pd.DataFrame,
    raw_low: pd.DataFrame,
    raw_close: pd.DataFrame,
    adj_close: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adjust raw OHLC prices using the split/dividend adjustment factor.

    The adjustment factor is computed as :math:`\\text{factor} = P^{adj} / P^{close}`.
    Open, High, and Low are multiplied by this factor; Adjusted Close is
    returned as-is.

    Args:
        raw_open: DataFrame of unadjusted open prices.
        raw_high: DataFrame of unadjusted high prices.
        raw_low: DataFrame of unadjusted low prices.
        raw_close: DataFrame of unadjusted close prices.
        adj_close: DataFrame of adjusted close prices (numerator of factor).

    Returns:
        Tuple of ``(adj_open, adj_high, adj_low, adj_close_copy)`` DataFrames,
        all aligned to the same index as the inputs.
    """
    factor = adj_close / raw_close
    adj_open = raw_open * factor
    adj_high = raw_high * factor
    adj_low = raw_low * factor
    adj_close_aligned = adj_close.copy()
    return adj_open, adj_high, adj_low, adj_close_aligned


def align_trading_calendar(
    data_dict: Dict[str, pd.DataFrame],
    min_history: int = 252 + 100,
) -> Dict[str, pd.DataFrame]:
    """Align all price series to the common valid trading calendar.

    Drops any dates where at least one ticker has a NaN in ``adj_close``, then
    discards the first ``min_history`` rows so downstream feature computation
    has enough warm-up data.

    Args:
        data_dict: Dict with keys ``'open'``, ``'high'``, ``'low'``,
            ``'close'``, ``'adj_close'``, ``'volume'`` (and optionally
            ``'log_returns'``), each a :class:`pd.DataFrame`.
        min_history: Number of leading rows to discard after NaN removal.
            Defaults to 352 (252-day feature window + 100-day safety margin).

    Returns:
        New dict with the same keys, each DataFrame trimmed to the common
        valid dates starting after ``min_history`` rows have been discarded.
    """
    adj = data_dict['adj_close']
    valid_dates = adj.dropna().index
    result: Dict[str, pd.DataFrame] = {}
    for key, df in data_dict.items():
        aligned = df.loc[valid_dates]
        result[key] = aligned.iloc[min_history:]
    return result
