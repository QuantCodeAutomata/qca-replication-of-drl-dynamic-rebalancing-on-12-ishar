"""Download DJIA constituent data and persist to disk.

Usage
-----
    python data/download_djia.py

Downloads daily OHLCV + Adjusted Close for the 28-stock DJIA universe used
in Yu & Chang (2025) and saves the adj_close DataFrame to
``data/djia_data.parquet``.
"""

import sys

sys.path.insert(0, '/workspace/project')

import pandas as pd
from src.data_module.downloader import DJIA_TICKERS, download_universe

BENCHMARK = 'DIA'
START = '2000-01-01'
END = '2024-01-01'
CACHE_DIR = '/workspace/project/data/cache/djia'
OUTPUT = '/workspace/project/data/djia_data.parquet'


def main() -> None:
    print(f"Downloading DJIA universe ({len(DJIA_TICKERS)} tickers) …")
    data_dict, benchmark = download_universe(
        tickers=DJIA_TICKERS,
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
