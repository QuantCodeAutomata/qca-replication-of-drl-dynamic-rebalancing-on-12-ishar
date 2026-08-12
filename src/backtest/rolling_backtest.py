import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    """Configuration for a single rolling train/validate/test window.

    Attributes
    ----------
    train_start : pd.Timestamp
        First date of the training segment.
    train_end : pd.Timestamp
        Last date of the training segment.
    val_start : pd.Timestamp
        First date of the validation segment.
    val_end : pd.Timestamp
        Last date of the validation segment.
    test_start : pd.Timestamp
        First date of the test (out-of-sample) segment.
    test_end : pd.Timestamp
        Last date of the test segment (clipped to the available history).
    window_idx : int
        Zero-based index of this window within the full rolling scheme.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    window_idx: int


def build_rolling_windows(
    all_dates: pd.DatetimeIndex,
    universe: str = 'etf',
) -> List[WindowConfig]:
    """
    Build rolling 2-year train / 2-year validate / 2-year test windows.

    Windows advance by 2 years per step so that consecutive test periods are
    non-overlapping.  Each window's test end date is clipped to the last
    available date in ``all_dates``; windows whose resulting test segment
    contains fewer than 20 trading days are silently skipped.

    ETF universe (8 windows starting 2003)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    +--------+------------------+------------------+------------------+
    | Window | Train            | Validate         | Test             |
    +========+==================+==================+==================+
    | 0      | 2003–2004        | 2005–2006        | 2007–2008        |
    | 1      | 2005–2006        | 2007–2008        | 2009–2010        |
    | …      | …                | …                | …                |
    | 7      | 2015–2016        | 2017–2018        | 2019–2020 *      |
    +--------+------------------+------------------+------------------+
    ``*`` truncated to Jan 2023 if data ends there.

    DJIA universe (6 windows starting 2007)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    +--------+------------------+------------------+------------------+
    | Window | Train            | Validate         | Test             |
    +========+==================+==================+==================+
    | 0      | 2007–2008        | 2009–2010        | 2011–2012        |
    | 1      | 2009–2010        | 2011–2012        | 2013–2014        |
    | …      | …                | …                | …                |
    | 5      | 2017–2018        | 2019–2020        | 2021–2022        |
    +--------+------------------+------------------+------------------+

    Parameters
    ----------
    all_dates : pd.DatetimeIndex
        Full sorted sequence of trading dates available in the dataset.
    universe : str
        Asset universe selector.  ``'etf'`` (default) or ``'djia'``.

    Returns
    -------
    List[WindowConfig]
        Ordered list of window configurations with valid test segments.
    """
    if universe == 'etf':
        base_year = 2003
        n_windows = 8
    else:  # djia
        base_year = 2007
        n_windows = 6

    windows: List[WindowConfig] = []
    max_date = all_dates[-1]

    for i in range(n_windows):
        train_s = pd.Timestamp(f'{base_year + 2 * i}-01-01')
        train_e = pd.Timestamp(f'{base_year + 2 * i + 1}-12-31')
        val_s = pd.Timestamp(f'{base_year + 2 * i + 2}-01-01')
        val_e = pd.Timestamp(f'{base_year + 2 * i + 3}-12-31')
        test_s = pd.Timestamp(f'{base_year + 2 * i + 4}-01-01')
        test_e = pd.Timestamp(f'{base_year + 2 * i + 5}-12-31')

        # Clip test end to the last available date
        test_e = min(test_e, max_date)

        # Skip windows with an insufficient test period
        test_mask = (all_dates >= test_s) & (all_dates <= test_e)
        if test_mask.sum() < 20:
            logger.info(f'Skipping window {i}: insufficient test dates '
                        f'({test_mask.sum()} < 20)')
            continue

        windows.append(WindowConfig(
            train_start=train_s,
            train_end=train_e,
            val_start=val_s,
            val_end=val_e,
            test_start=test_s,
            test_end=test_e,
            window_idx=i,
        ))

    return windows


def get_window_dates(
    all_dates: pd.DatetimeIndex,
    window: WindowConfig,
    segment: str,
) -> pd.DatetimeIndex:
    """
    Extract the trading dates belonging to a given window segment.

    Parameters
    ----------
    all_dates : pd.DatetimeIndex
        Full sorted sequence of trading dates available in the dataset.
    window : WindowConfig
        Window configuration produced by :func:`build_rolling_windows`.
    segment : str
        One of ``'train'``, ``'val'``, or ``'test'``.

    Returns
    -------
    pd.DatetimeIndex
        Subset of ``all_dates`` that falls within the requested segment.
    """
    if segment == 'train':
        mask = (all_dates >= window.train_start) & (all_dates <= window.train_end)
    elif segment == 'val':
        mask = (all_dates >= window.val_start) & (all_dates <= window.val_end)
    else:  # test
        mask = (all_dates >= window.test_start) & (all_dates <= window.test_end)
    return all_dates[mask]
