"""Download daily OHLCV + Adjusted Close data from Yahoo Finance using yfinance."""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

ETF_TICKERS: List[str] = [
    'IYW', 'IYF', 'IYZ', 'IYM', 'IYK', 'IYC',
    'IYE', 'IYG', 'IYH', 'IYJ', 'IDU', 'IYR',
]

DJIA_TICKERS: List[str] = [
    'BA', 'AMGN', 'DIS', 'NKE', 'HON', 'MMM', 'CAT', 'KO', 'PG', 'AXP',
    'GS', 'JPM', 'MCD', 'HD', 'AAPL', 'CRM', 'CSCO', 'IBM', 'MSFT', 'TRV',
    'UNH', 'CVX', 'JNJ', 'MRK', 'AMZN', 'WMT', 'INTC', 'VZ',
]


def _parse_yf_data(
    raw: pd.DataFrame, tickers: List[str]
) -> Dict[str, pd.DataFrame]:
    """Parse a raw yfinance multi-ticker DataFrame into per-price-type DataFrames.

    Args:
        raw: Multi-level column DataFrame returned by yf.download (auto_adjust=False).
        tickers: Ordered list of tickers used in the download.

    Returns:
        Dict mapping price type keys to DataFrames indexed by date with ticker columns.
    """
    # yfinance 0.2+ returns columns as (price_type, ticker) MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        price_map = {
            'open':      'Open',
            'high':      'High',
            'low':       'Low',
            'close':     'Close',
            'adj_close': 'Adj Close',
            'volume':    'Volume',
        }
        result: Dict[str, pd.DataFrame] = {}
        for key, label in price_map.items():
            if label in raw.columns.get_level_values(0):
                df = raw[label].copy()
                # Ensure all requested tickers appear as columns
                df = df.reindex(columns=tickers)
                result[key] = df
            else:
                result[key] = pd.DataFrame(index=raw.index, columns=tickers, dtype=float)
    else:
        # Single ticker fallback: promote to DataFrame with one column
        ticker = tickers[0]
        result = {
            'open':      raw[['Open']].rename(columns={'Open': ticker}),
            'high':      raw[['High']].rename(columns={'High': ticker}),
            'low':       raw[['Low']].rename(columns={'Low': ticker}),
            'close':     raw[['Close']].rename(columns={'Close': ticker}),
            'adj_close': raw[['Adj Close']].rename(columns={'Adj Close': ticker}),
            'volume':    raw[['Volume']].rename(columns={'Volume': ticker}),
        }
    return result


def download_universe(
    tickers: List[str],
    benchmark: str,
    start: str,
    end: str,
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """Download daily OHLCV + Adjusted Close data for a universe of tickers.

    Downloads data from Yahoo Finance, optionally caches the results as parquet
    files, and returns cleaned DataFrames aligned to a common trading calendar.

    Args:
        tickers: List of asset ticker symbols to download.
        benchmark: A single benchmark ticker (e.g. ``'^GSPC'``).
        start: Start date string in ``'YYYY-MM-DD'`` format (inclusive).
        end: End date string in ``'YYYY-MM-DD'`` format (exclusive).
        cache_dir: If provided, raw data are saved to / loaded from this
            directory as parquet files to avoid repeated downloads.

    Returns:
        Tuple of:
            - ``data_dict``: dict with keys ``'open'``, ``'high'``, ``'low'``,
              ``'close'``, ``'adj_close'``, ``'volume'``, each a
              :class:`pd.DataFrame` (rows = dates, columns = tickers).
            - ``benchmark_series``: :class:`pd.Series` of adjusted close prices
              for the benchmark ticker, aligned to the same dates.

    Notes:
        * Dates with **any** NaN in the ``adj_close`` slice (universe *or*
          benchmark) are dropped so that the returned data are fully
          populated.
        * ``auto_adjust=False`` is used so that both raw Close and Adj Close
          are available for downstream OHLC adjustment.
    """
    cache_path: Optional[Path] = Path(cache_dir) if cache_dir else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

    all_tickers = tickers + [benchmark]
    cache_file = cache_path / 'raw.parquet' if cache_path else None

    if cache_file is not None and cache_file.exists():
        raw = pd.read_parquet(cache_file)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = yf.download(
                all_tickers,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        if cache_file is not None:
            raw.to_parquet(cache_file)

    # Parse universe tickers
    data_dict = _parse_yf_data(raw, tickers)

    # Parse benchmark
    bench_dict = _parse_yf_data(raw, [benchmark])
    benchmark_series: pd.Series = bench_dict['adj_close'][benchmark]

    # Build combined adj_close (universe + benchmark) to find valid dates
    combined_adj = pd.concat([data_dict['adj_close'], benchmark_series.rename(benchmark)], axis=1)
    valid_dates = combined_adj.dropna().index

    # Align all DataFrames to valid dates
    for key in data_dict:
        data_dict[key] = data_dict[key].loc[valid_dates]
    benchmark_series = benchmark_series.loc[valid_dates]

    return data_dict, benchmark_series
