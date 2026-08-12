"""Tests for the data_module: downloader, preprocessor, and alignment utilities."""

import sys
sys.path.insert(0, '/workspace/project')

from typing import Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_module.preprocessor import (
    align_trading_calendar,
    compute_adjusted_ohlc,
    compute_log_returns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 50, n_cols: int = 3, seed: int = 0) -> pd.DataFrame:
    """Return a DataFrame of synthetic prices with a business-day DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2020-01-01', periods=n)
    data = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, n_cols)), axis=0))
    return pd.DataFrame(data, index=dates, columns=[f'T{i}' for i in range(n_cols)])


def _make_data_dict(n: int = 60) -> Dict[str, pd.DataFrame]:
    """Return a minimal data_dict compatible with align_trading_calendar."""
    close = _make_price_df(n=n)
    # Inject a few NaN rows to test removal
    close.iloc[5] = np.nan
    close.iloc[10] = np.nan
    factor = pd.DataFrame(
        np.random.default_rng(1).uniform(0.98, 1.02, close.shape),
        index=close.index,
        columns=close.columns,
    )
    adj_close = close * factor
    adj_close.iloc[5] = np.nan
    adj_close.iloc[10] = np.nan
    return {
        'open':      close * 0.99,
        'high':      close * 1.01,
        'low':       close * 0.98,
        'close':     close,
        'adj_close': adj_close,
        'volume':    pd.DataFrame(
            np.random.default_rng(2).integers(1000, 5000, close.shape).astype(float),
            index=close.index,
            columns=close.columns,
        ),
    }


# ---------------------------------------------------------------------------
# test_compute_log_returns
# ---------------------------------------------------------------------------

class TestComputeLogReturns:
    def test_output_shape(self):
        prices = _make_price_df(n=30)
        lr = compute_log_returns(prices)
        # First row is dropped (NaN from shift)
        assert lr.shape == (29, 3)

    def test_values_correct(self):
        prices = _make_price_df(n=10)
        lr = compute_log_returns(prices)
        expected = np.log(prices / prices.shift(1)).dropna()
        pd.testing.assert_frame_equal(lr, expected)

    def test_no_nan_in_output(self):
        prices = _make_price_df(n=20)
        lr = compute_log_returns(prices)
        assert lr.isna().sum().sum() == 0

    def test_index_alignment(self):
        prices = _make_price_df(n=10)
        lr = compute_log_returns(prices)
        # Index should start at the second date of the original
        assert lr.index[0] == prices.index[1]

    def test_single_column(self):
        prices = _make_price_df(n=10, n_cols=1)
        lr = compute_log_returns(prices)
        assert lr.shape[1] == 1


# ---------------------------------------------------------------------------
# test_adjusted_ohlc
# ---------------------------------------------------------------------------

class TestComputeAdjustedOHLC:
    def _make_ohlc(self, n: int = 20):
        close = _make_price_df(n=n)
        high = close * 1.005
        low = close * 0.995
        open_ = close.shift(1).fillna(close.iloc[0])
        adj_close = close * 1.02  # 2% adjustment factor
        return open_, high, low, close, adj_close

    def test_factor_applied_correctly(self):
        open_, high, low, close, adj_close = self._make_ohlc()
        ao, ah, al, ac = compute_adjusted_ohlc(open_, high, low, close, adj_close)
        factor = adj_close / close
        pd.testing.assert_frame_equal(ao, open_ * factor)
        pd.testing.assert_frame_equal(ah, high * factor)
        pd.testing.assert_frame_equal(al, low * factor)

    def test_adj_close_unchanged(self):
        open_, high, low, close, adj_close = self._make_ohlc()
        _, _, _, ac = compute_adjusted_ohlc(open_, high, low, close, adj_close)
        pd.testing.assert_frame_equal(ac, adj_close)

    def test_output_shapes_match_input(self):
        open_, high, low, close, adj_close = self._make_ohlc(n=15)
        ao, ah, al, ac = compute_adjusted_ohlc(open_, high, low, close, adj_close)
        for df in (ao, ah, al, ac):
            assert df.shape == close.shape

    def test_identity_when_factor_is_one(self):
        close = _make_price_df(n=10)
        adj_close = close.copy()  # factor = 1 everywhere
        open_ = close * 0.99
        high = close * 1.01
        low = close * 0.99
        ao, ah, al, ac = compute_adjusted_ohlc(open_, high, low, close, adj_close)
        pd.testing.assert_frame_equal(ao, open_)
        pd.testing.assert_frame_equal(ah, high)
        pd.testing.assert_frame_equal(al, low)


# ---------------------------------------------------------------------------
# test_align_calendar
# ---------------------------------------------------------------------------

class TestAlignTradingCalendar:
    def test_no_nan_in_adj_close_after_align(self):
        data_dict = _make_data_dict(n=80)
        aligned = align_trading_calendar(data_dict, min_history=0)
        assert aligned['adj_close'].isna().sum().sum() == 0

    def test_nan_rows_removed(self):
        data_dict = _make_data_dict(n=80)
        raw_valid = data_dict['adj_close'].dropna().shape[0]
        aligned = align_trading_calendar(data_dict, min_history=0)
        assert aligned['adj_close'].shape[0] == raw_valid

    def test_min_history_trimmed(self):
        data_dict = _make_data_dict(n=80)
        raw_valid = data_dict['adj_close'].dropna().shape[0]
        min_h = 5
        aligned = align_trading_calendar(data_dict, min_history=min_h)
        assert aligned['adj_close'].shape[0] == raw_valid - min_h

    def test_all_keys_preserved(self):
        data_dict = _make_data_dict(n=80)
        aligned = align_trading_calendar(data_dict, min_history=0)
        assert set(aligned.keys()) == set(data_dict.keys())

    def test_consistent_index_across_keys(self):
        data_dict = _make_data_dict(n=80)
        aligned = align_trading_calendar(data_dict, min_history=0)
        ref_idx = aligned['adj_close'].index
        for key in aligned:
            pd.testing.assert_index_equal(aligned[key].index, ref_idx)


# ---------------------------------------------------------------------------
# test_download_universe (mock-based — no network required)
# ---------------------------------------------------------------------------

class TestDownloadUniverse:
    """Smoke-tests for download_universe using a synthetic mock of yf.download."""

    def _make_mock_raw(self, tickers, benchmark, n=50):
        """Build a fake yfinance multi-level DataFrame."""
        all_t = tickers + [benchmark]
        dates = pd.bdate_range('2020-01-01', periods=n)
        rng = np.random.default_rng(99)
        price_types = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        arrays = [
            np.repeat(price_types, len(all_t)),
            np.tile(all_t, len(price_types)),
        ]
        cols = pd.MultiIndex.from_arrays(arrays, names=[None, None])
        data = rng.uniform(10, 200, size=(n, len(price_types) * len(all_t)))
        return pd.DataFrame(data, index=dates, columns=cols)

    def test_returns_correct_keys(self):
        from src.data_module.downloader import download_universe

        tickers = ['AAPL', 'MSFT']
        benchmark = 'SPY'
        mock_raw = self._make_mock_raw(tickers, benchmark)

        with patch('src.data_module.downloader.yf.download', return_value=mock_raw):
            data_dict, bench = download_universe(tickers, benchmark, '2020-01-01', '2020-04-01')

        expected_keys = {'open', 'high', 'low', 'close', 'adj_close', 'volume'}
        assert set(data_dict.keys()) == expected_keys

    def test_no_nan_after_download(self):
        from src.data_module.downloader import download_universe

        tickers = ['AAPL', 'MSFT']
        benchmark = 'SPY'
        mock_raw = self._make_mock_raw(tickers, benchmark)

        with patch('src.data_module.downloader.yf.download', return_value=mock_raw):
            data_dict, bench = download_universe(tickers, benchmark, '2020-01-01', '2020-04-01')

        assert data_dict['adj_close'].isna().sum().sum() == 0
        assert bench.isna().sum() == 0

    def test_benchmark_series_is_1d(self):
        from src.data_module.downloader import download_universe

        tickers = ['AAPL', 'MSFT']
        benchmark = 'SPY'
        mock_raw = self._make_mock_raw(tickers, benchmark)

        with patch('src.data_module.downloader.yf.download', return_value=mock_raw):
            _, bench = download_universe(tickers, benchmark, '2020-01-01', '2020-04-01')

        assert isinstance(bench, pd.Series)

    def test_ticker_columns_match(self):
        from src.data_module.downloader import download_universe

        tickers = ['AAPL', 'MSFT', 'GOOG']
        benchmark = 'SPY'
        mock_raw = self._make_mock_raw(tickers, benchmark)

        with patch('src.data_module.downloader.yf.download', return_value=mock_raw):
            data_dict, _ = download_universe(tickers, benchmark, '2020-01-01', '2020-04-01')

        assert list(data_dict['adj_close'].columns) == tickers

    def test_caching_skips_download(self, tmp_path):
        from src.data_module.downloader import download_universe

        tickers = ['AAPL', 'MSFT']
        benchmark = 'SPY'
        mock_raw = self._make_mock_raw(tickers, benchmark)

        call_count = {'n': 0}

        def _mock_download(*args, **kwargs):
            call_count['n'] += 1
            return mock_raw

        with patch('src.data_module.downloader.yf.download', side_effect=_mock_download):
            download_universe(tickers, benchmark, '2020-01-01', '2020-04-01',
                              cache_dir=str(tmp_path))
            download_universe(tickers, benchmark, '2020-01-01', '2020-04-01',
                              cache_dir=str(tmp_path))

        # Second call should use cache; yf.download called only once
        assert call_count['n'] == 1
