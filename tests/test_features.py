"""Tests for the feature builder and standardizer modules."""

import sys
sys.path.insert(0, '/workspace/project')

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.features.feature_builder import (
    FEATURE_NAMES,
    build_feature_matrix,
    compute_features_for_asset,
)
from src.features.standardizer import build_observation_tensor, rolling_zscore_standardize


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

N_ROWS = 300  # enough for all rolling windows


def _make_ohlcv(n: int = N_ROWS, seed: int = 42) -> Dict[str, pd.Series]:
    """Return a dict of synthetic OHLCV pd.Series with a business-day index."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2018-01-01', periods=n)
    log_ret = rng.normal(0, 0.01, n)
    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=dates, name='close')
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(rng.integers(100_000, 5_000_000, n).astype(float), index=dates)
    return {
        'adj_open': open_,
        'adj_high': high,
        'adj_low': low,
        'adj_close': close,
        'volume': volume,
    }


def _make_data_dict(tickers, n: int = N_ROWS) -> Dict[str, pd.DataFrame]:
    """Return a data_dict with multiple tickers for build_feature_matrix."""
    base = _make_ohlcv(n=n)
    rng = np.random.default_rng(7)
    dates = base['adj_close'].index
    result = {key: pd.DataFrame(index=dates) for key in ('open', 'high', 'low', 'adj_close', 'volume')}
    for i, ticker in enumerate(tickers):
        o = _make_ohlcv(n=n, seed=i)
        result['open'][ticker] = o['adj_open']
        result['high'][ticker] = o['adj_high']
        result['low'][ticker] = o['adj_low']
        result['adj_close'][ticker] = o['adj_close']
        result['volume'][ticker] = o['volume']
    return result


# ---------------------------------------------------------------------------
# test_compute_features_shape
# ---------------------------------------------------------------------------

class TestComputeFeaturesShape:
    def test_returns_dataframe(self):
        ohlcv = _make_ohlcv()
        df = compute_features_for_asset(**ohlcv)
        assert isinstance(df, pd.DataFrame)

    def test_column_count(self):
        ohlcv = _make_ohlcv()
        df = compute_features_for_asset(**ohlcv)
        assert df.shape[1] == 34

    def test_column_names_match(self):
        ohlcv = _make_ohlcv()
        df = compute_features_for_asset(**ohlcv)
        assert list(df.columns) == FEATURE_NAMES

    def test_row_count_matches_input(self):
        ohlcv = _make_ohlcv(n=200)
        df = compute_features_for_asset(**ohlcv)
        assert df.shape[0] == 200

    def test_index_matches_input(self):
        ohlcv = _make_ohlcv()
        df = compute_features_for_asset(**ohlcv)
        pd.testing.assert_index_equal(df.index, ohlcv['adj_close'].index)

    def test_adj_close_column_matches(self):
        ohlcv = _make_ohlcv()
        df = compute_features_for_asset(**ohlcv)
        pd.testing.assert_series_equal(
            df['adj_close'],
            ohlcv['adj_close'].astype(float).rename('adj_close'),
        )

    def test_no_all_nan_columns_with_enough_data(self):
        """With 300 rows every indicator should produce at least some valid values."""
        ohlcv = _make_ohlcv(n=300)
        df = compute_features_for_asset(**ohlcv)
        all_nan_cols = [c for c in df.columns if df[c].isna().all()]
        assert all_nan_cols == [], f"Columns entirely NaN: {all_nan_cols}"


# ---------------------------------------------------------------------------
# test_build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_all_tickers_present(self):
        tickers = ['AAPL', 'MSFT', 'GOOG']
        data_dict = _make_data_dict(tickers)
        matrix = build_feature_matrix(data_dict, tickers)
        assert set(matrix.keys()) == set(tickers)

    def test_each_ticker_has_34_features(self):
        tickers = ['AAPL', 'MSFT']
        data_dict = _make_data_dict(tickers)
        matrix = build_feature_matrix(data_dict, tickers)
        for ticker in tickers:
            assert matrix[ticker].shape[1] == 34, ticker

    def test_single_ticker(self):
        tickers = ['AAPL']
        data_dict = _make_data_dict(tickers)
        matrix = build_feature_matrix(data_dict, tickers)
        assert list(matrix.keys()) == ['AAPL']
        assert matrix['AAPL'].shape[1] == 34


# ---------------------------------------------------------------------------
# test_rolling_zscore
# ---------------------------------------------------------------------------

class TestRollingZscore:
    def _make_feature_df(self, n: int = 400, n_features: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        dates = pd.bdate_range('2018-01-01', periods=n)
        data = rng.standard_normal((n, n_features)) * 10 + 50
        return pd.DataFrame(data, index=dates, columns=[f'f{i}' for i in range(n_features)])

    def test_output_shape_unchanged(self):
        df = self._make_feature_df()
        z = rolling_zscore_standardize(df, window=252)
        assert z.shape == df.shape

    def test_first_window_rows_are_nan(self):
        df = self._make_feature_df(n=400)
        window = 252
        z = rolling_zscore_standardize(df, window=window)
        # Rows 0..window-2 should be NaN (insufficient history)
        assert z.iloc[:window - 1].isna().all().all()

    def test_values_after_warmup_are_finite(self):
        df = self._make_feature_df(n=500)
        z = rolling_zscore_standardize(df, window=252)
        tail = z.iloc[252:]
        assert np.isfinite(tail.values).all()

    def test_zero_std_column_gives_zero_not_nan(self):
        """A constant column should produce 0 (not NaN or inf) after warm-up."""
        dates = pd.bdate_range('2018-01-01', periods=400)
        df = pd.DataFrame({'const': np.ones(400), 'normal': np.random.randn(400)}, index=dates)
        z = rolling_zscore_standardize(df, window=252)
        assert (z['const'].iloc[252:] == 0.0).all()

    def test_standardized_mean_near_zero(self):
        """Mean of z-scored values after warm-up should be near 0."""
        df = self._make_feature_df(n=700)
        z = rolling_zscore_standardize(df, window=252)
        mean_abs = z.iloc[252:].mean().abs()
        assert (mean_abs < 0.5).all()

    def test_output_is_dataframe(self):
        df = self._make_feature_df()
        z = rolling_zscore_standardize(df)
        assert isinstance(z, pd.DataFrame)

    def test_column_names_preserved(self):
        df = self._make_feature_df()
        z = rolling_zscore_standardize(df)
        assert list(z.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# test_observation_tensor_shape
# ---------------------------------------------------------------------------

class TestObservationTensorShape:
    def _make_std_features(
        self, tickers, n: int = 400, n_features: int = 34
    ) -> Dict[str, pd.DataFrame]:
        rng = np.random.default_rng(5)
        dates = pd.bdate_range('2018-01-01', periods=n)
        result = {}
        for ticker in tickers:
            data = rng.standard_normal((n, n_features))
            result[ticker] = pd.DataFrame(data, index=dates, columns=FEATURE_NAMES)
        return result

    def test_shape_single_asset(self):
        tickers = ['AAPL']
        features = self._make_std_features(tickers)
        date = features['AAPL'].index[299]
        tensor = build_observation_tensor(features, tickers, date, lookback=252)
        assert tensor.shape == (34, 1, 252)

    def test_shape_multiple_assets(self):
        tickers = ['AAPL', 'MSFT', 'GOOG']
        features = self._make_std_features(tickers)
        date = features['AAPL'].index[299]
        tensor = build_observation_tensor(features, tickers, date, lookback=252)
        assert tensor.shape == (34, 3, 252)

    def test_dtype_is_float32(self):
        tickers = ['AAPL', 'MSFT']
        features = self._make_std_features(tickers)
        date = features['AAPL'].index[299]
        tensor = build_observation_tensor(features, tickers, date, lookback=252)
        assert tensor.dtype == np.float32

    def test_lookback_dimension(self):
        tickers = ['AAPL']
        features = self._make_std_features(tickers, n=400)
        date = features['AAPL'].index[350]
        for lb in (10, 50, 252):
            tensor = build_observation_tensor(features, tickers, date, lookback=lb)
            assert tensor.shape[2] == lb

    def test_last_column_matches_date_row(self):
        """The last time step in the tensor should equal the feature row at `date`."""
        tickers = ['AAPL']
        features = self._make_std_features(tickers, n=400)
        date = features['AAPL'].index[300]
        tensor = build_observation_tensor(features, tickers, date, lookback=50)
        # tensor[:, 0, -1] should equal features['AAPL'].loc[date].values
        expected = features['AAPL'].loc[date].values.astype(np.float32)
        np.testing.assert_array_almost_equal(tensor[:, 0, -1], expected)

    def test_raises_on_missing_date(self):
        tickers = ['AAPL']
        features = self._make_std_features(tickers, n=100)
        bad_date = pd.Timestamp('1900-01-01')
        with pytest.raises(ValueError, match="not found"):
            build_observation_tensor(features, tickers, bad_date, lookback=10)

    def test_raises_on_insufficient_history(self):
        tickers = ['AAPL']
        features = self._make_std_features(tickers, n=100)
        early_date = features['AAPL'].index[5]  # only 6 rows available
        with pytest.raises(ValueError, match="Not enough history"):
            build_observation_tensor(features, tickers, early_date, lookback=252)


# ---------------------------------------------------------------------------
# test_feature_edge_cases
# ---------------------------------------------------------------------------

class TestFeatureEdgeCases:
    def test_minimal_data_does_not_raise(self):
        """compute_features_for_asset must not raise even with very few rows."""
        ohlcv = _make_ohlcv(n=20)
        df = compute_features_for_asset(**ohlcv)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (20, 34)

    def test_constant_price_does_not_raise(self):
        """Constant prices cause zero std; function must handle gracefully."""
        n = 200
        dates = pd.bdate_range('2020-01-01', periods=n)
        close = pd.Series(np.ones(n) * 100.0, index=dates)
        high = close + 0.5
        low = close - 0.5
        open_ = close.copy()
        volume = pd.Series(np.ones(n) * 1e6, index=dates)
        df = compute_features_for_asset(open_, high, low, close, volume)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (n, 34)

    def test_feature_names_constant(self):
        assert len(FEATURE_NAMES) == 34
        assert len(set(FEATURE_NAMES)) == 34  # all unique

    def test_pct_change_log_return_finite(self):
        ohlcv = _make_ohlcv(n=200)
        df = compute_features_for_asset(**ohlcv)
        # Skip first row (NaN from shift)
        assert np.isfinite(df['pct_change'].iloc[1:].values).all()
        assert np.isfinite(df['log_return'].iloc[1:].values).all()

    def test_sma_ordering(self):
        """SMA100 should be smoother (smaller day-over-day change) than SMA10.

        We compare the standard deviation of *first differences*, which measures
        how much each moving average changes per day.  A longer window always
        dampens daily moves more than a shorter window.
        """
        ohlcv = _make_ohlcv(n=300)
        df = compute_features_for_asset(**ohlcv)
        tail = df.iloc[120:]
        sma10_diff_std = tail['sma10'].diff().std()
        sma100_diff_std = tail['sma100'].diff().std()
        assert sma100_diff_std < sma10_diff_std

    def test_rsi_bounded(self):
        """RSI should lie in [0, 100]."""
        ohlcv = _make_ohlcv(n=300)
        df = compute_features_for_asset(**ohlcv)
        rsi = df['rsi'].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_bb_upper_above_lower(self):
        """Bollinger upper band should always be >= lower band."""
        ohlcv = _make_ohlcv(n=200)
        df = compute_features_for_asset(**ohlcv)
        valid = df[['bb_upper', 'bb_lower']].dropna()
        assert (valid['bb_upper'] >= valid['bb_lower']).all()
