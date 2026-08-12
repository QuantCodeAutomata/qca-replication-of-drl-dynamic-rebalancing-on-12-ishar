"""Download ETF universe data and persist to disk.

Usage
-----
    python data/download_etf.py

Downloads daily OHLCV + Adjusted Close for the 12-ETF iShares universe used
in Yu & Chang (2025) and saves the adj_close DataFrame to
``data/etf_data.parquet``.
"""

import sys

sys.path.insert(0, '/workspace/project')

import pandas as pd
from src.data_module.downloader import ETF_TICKERS, download_universe

BENCHMARK = 'SPY'
START = '2000-01-01'
END = '2024-01-01'
CACHE_DIR = '/workspace/project/data/cache/etf'
OUTPUT = '/workspace/project/data/etf_data.parquet'


def main() -> None:
    print(f"Downloading ETF universe ({len(ETF_TICKERS)} tickers) …")
    data_dict, benchmark = download_universe(
        tickers=ETF_TICKERS,
        benchmark=BENCHMARK,
        start=START,
        end=END,
        cache_dir=CACHE_DIR,
    )
    adj_close = data_dict['adj_close']
    adj_close.to_parquet(OUTPUT)
    print(f"Saved adj_close ({adj_close.shape}) to {OUTPUT}")
    print(f"Date range: {adj_close.index[0].date()} → {adj_close.index[-1].date()}")


if __name__ == '__main__':
    main()
