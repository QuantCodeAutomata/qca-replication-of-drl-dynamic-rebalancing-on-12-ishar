"""Compute 34 per-asset technical features for the DRL rebalancing model."""

from typing import Dict, List

import numpy as np
import pandas as pd
import pandas_ta as ta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(fn, fallback_index: pd.Index) -> pd.Series:
    """Execute *fn* and return its result; on any exception return NaN Series."""
    try:
        result = fn()
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        return result.astype(float)
    except Exception:
        return pd.Series(np.nan, index=fallback_index, dtype=float)


def _linreg_values(series: pd.Series, window: int) -> pd.Series:
    """Return the OLS fitted value at each point using a rolling window."""
    x = np.arange(window, dtype=float)
    x -= x.mean()
    x_sq = (x * x).sum()

    def _ols_endpoint(y: np.ndarray) -> float:
        slope = (x * y).sum() / x_sq
        intercept = y.mean()
        return intercept + slope * x[-1]

    return series.rolling(window).apply(_ols_endpoint, raw=True)


def _supertrend_bands(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    multiplier: float = 3.0,
) -> tuple:
    """Compute the classic Supertrend upper and lower bands.

    Returns:
        Tuple of ``(upper_band, lower_band)`` as :class:`pd.Series`.
    """
    try:
        st = ta.supertrend(high, low, close, length=length, multiplier=multiplier)
        upper_col = [c for c in st.columns if c.startswith('SUPERTs')][0]
        lower_col = [c for c in st.columns if c.startswith('SUPERTl')][0]
        return st[upper_col].astype(float), st[lower_col].astype(float)
    except Exception:
        nan = pd.Series(np.nan, index=close.index, dtype=float)
        return nan, nan.copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    'adj_close', 'pct_change', 'log_return',
    'aroon_osc', 'awesome_osc',
    'bb_upper', 'bb_lower',
    'cmo', 'cci', 'cti', 'adx',
    'er_bull', 'er_bear',
    'inertia', 'ker', 'kst',
    'linreg', 'macd', 'pvo', 'pgo', 'psl',
    'qqe', 'rsi', 'rvi_signal',
    'sma10', 'sma20', 'sma100',
    'stochrsi_k',
    'supertrend_upper', 'supertrend_lower',
    'fisher', 'tema', 'vvi', 'zscore_close',
]


def compute_features_for_asset(
    adj_open: pd.Series,
    adj_high: pd.Series,
    adj_low: pd.Series,
    adj_close: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """Compute all 34 technical features for a single asset.

    All features are computed in a strictly look-back fashion (no future data
    leakage).  When an indicator cannot be computed (e.g. insufficient history
    or a library error), the corresponding column is filled with ``NaN``.

    Args:
        adj_open: Adjusted open price series.
        adj_high: Adjusted high price series.
        adj_low: Adjusted low price series.
        adj_close: Adjusted close price series.
        volume: Volume series.

    Returns:
        :class:`pd.DataFrame` with ``len(FEATURE_NAMES)`` columns (34 total)
        and the same index as the input series.
    """
    idx = adj_close.index
    out: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # 1. adj_close – price level
    # ------------------------------------------------------------------
    out['adj_close'] = adj_close.astype(float)

    # ------------------------------------------------------------------
    # 2. pct_change – 1-day percentage change
    # ------------------------------------------------------------------
    out['pct_change'] = _safe(lambda: adj_close.pct_change(), idx)

    # ------------------------------------------------------------------
    # 3. log_return – 1-day log return
    # ------------------------------------------------------------------
    out['log_return'] = _safe(
        lambda: np.log(adj_close / adj_close.shift(1)), idx
    )

    # ------------------------------------------------------------------
    # 4. aroon_osc – Aroon Oscillator n=25
    # ------------------------------------------------------------------
    def _aroon():
        df = ta.aroon(adj_high, adj_low, length=25)
        col = [c for c in df.columns if 'OSC' in c][0]
        return df[col]

    out['aroon_osc'] = _safe(_aroon, idx)

    # ------------------------------------------------------------------
    # 5. awesome_osc – AO = SMA(5) - SMA(34) of median price
    # ------------------------------------------------------------------
    out['awesome_osc'] = _safe(lambda: ta.ao(adj_high, adj_low), idx)

    # ------------------------------------------------------------------
    # 6-7. bb_upper / bb_lower – Bollinger Bands n=20 k=2
    # ------------------------------------------------------------------
    def _bbands():
        df = ta.bbands(adj_close, length=20, std=2)
        upper_col = [c for c in df.columns if c.startswith('BBU')][0]
        lower_col = [c for c in df.columns if c.startswith('BBL')][0]
        return df[upper_col], df[lower_col]

    try:
        _bb = ta.bbands(adj_close, length=20, std=2)
        _bbu = [c for c in _bb.columns if c.startswith('BBU')][0]
        _bbl = [c for c in _bb.columns if c.startswith('BBL')][0]
        out['bb_upper'] = _bb[_bbu].astype(float)
        out['bb_lower'] = _bb[_bbl].astype(float)
    except Exception:
        out['bb_upper'] = pd.Series(np.nan, index=idx, dtype=float)
        out['bb_lower'] = pd.Series(np.nan, index=idx, dtype=float)

    # ------------------------------------------------------------------
    # 8. cmo – Chande Momentum Oscillator n=14
    # ------------------------------------------------------------------
    out['cmo'] = _safe(lambda: ta.cmo(adj_close, length=14), idx)

    # ------------------------------------------------------------------
    # 9. cci – Commodity Channel Index n=30
    # ------------------------------------------------------------------
    out['cci'] = _safe(
        lambda: ta.cci(adj_high, adj_low, adj_close, length=30), idx
    )

    # ------------------------------------------------------------------
    # 10. cti – Correlation Trend Indicator n=30
    # ------------------------------------------------------------------
    out['cti'] = _safe(lambda: ta.cti(adj_close, length=30), idx)

    # ------------------------------------------------------------------
    # 11. adx – Average Directional Index n=30
    # ------------------------------------------------------------------
    def _adx():
        df = ta.adx(adj_high, adj_low, adj_close, length=30)
        col = [c for c in df.columns if c.startswith('ADX_')][0]
        return df[col]

    out['adx'] = _safe(_adx, idx)

    # ------------------------------------------------------------------
    # 12-13. er_bull / er_bear – Elder-Ray Bull/Bear Power
    # ------------------------------------------------------------------
    def _elder_ray():
        ema13 = ta.ema(adj_close, length=13)
        bull = adj_high - ema13
        bear = adj_low - ema13
        return bull.astype(float), bear.astype(float)

    try:
        out['er_bull'], out['er_bear'] = _elder_ray()
    except Exception:
        out['er_bull'] = pd.Series(np.nan, index=idx, dtype=float)
        out['er_bear'] = pd.Series(np.nan, index=idx, dtype=float)

    # ------------------------------------------------------------------
    # 14. inertia – Linear regression of RVI n=20
    # ------------------------------------------------------------------
    out['inertia'] = _safe(
        lambda: ta.inertia(adj_close, adj_high, adj_low, length=20), idx
    )

    # ------------------------------------------------------------------
    # 15. ker – Kaufman Efficiency Ratio n=10
    # ------------------------------------------------------------------
    out['ker'] = _safe(lambda: ta.er(adj_close, length=10), idx)

    # ------------------------------------------------------------------
    # 16. kst – Know Sure Thing (pandas_ta defaults)
    # ------------------------------------------------------------------
    def _kst():
        df = ta.kst(adj_close)
        col = [c for c in df.columns if c.startswith('KST_')][0]
        return df[col]

    out['kst'] = _safe(_kst, idx)

    # ------------------------------------------------------------------
    # 17. linreg – Rolling OLS fitted value at each point (window=10)
    # ------------------------------------------------------------------
    out['linreg'] = _safe(
        lambda: ta.linreg(adj_close, length=10), idx
    )

    # ------------------------------------------------------------------
    # 18. macd – MACD line 12/26/9
    # ------------------------------------------------------------------
    def _macd():
        df = ta.macd(adj_close, fast=12, slow=26, signal=9)
        col = [c for c in df.columns if c.startswith('MACD_')][0]
        return df[col]

    out['macd'] = _safe(_macd, idx)

    # ------------------------------------------------------------------
    # 19. pvo – Percentage Volume Oscillator 12/26
    # ------------------------------------------------------------------
    def _pvo():
        df = ta.pvo(volume)
        col = [c for c in df.columns if c.startswith('PVO_')][0]
        return df[col]

    out['pvo'] = _safe(_pvo, idx)

    # ------------------------------------------------------------------
    # 20. pgo – Pretty Good Oscillator n=14
    # ------------------------------------------------------------------
    out['pgo'] = _safe(
        lambda: ta.pgo(adj_high, adj_low, adj_close, length=14), idx
    )

    # ------------------------------------------------------------------
    # 21. psl – Psychological Line n=12
    # ------------------------------------------------------------------
    out['psl'] = _safe(lambda: ta.psl(adj_close, length=12), idx)

    # ------------------------------------------------------------------
    # 22. qqe – QQE smoothed RSI (RSI 14, EMA 5)
    # ------------------------------------------------------------------
    def _qqe():
        df = ta.qqe(adj_close)
        col = [c for c in df.columns if c.startswith('QQE_')][0]
        return df[col]

    out['qqe'] = _safe(_qqe, idx)

    # ------------------------------------------------------------------
    # 23. rsi – RSI n=30
    # ------------------------------------------------------------------
    out['rsi'] = _safe(lambda: ta.rsi(adj_close, length=30), idx)

    # ------------------------------------------------------------------
    # 24. rvi_signal – Relative Vigor Index signal line n=10
    # ------------------------------------------------------------------
    def _rvgi_signal():
        df = ta.rvgi(adj_open, adj_high, adj_low, adj_close, length=10)
        col = [c for c in df.columns if c.startswith('RVGIs')][0]
        return df[col]

    out['rvi_signal'] = _safe(_rvgi_signal, idx)

    # ------------------------------------------------------------------
    # 25-27. sma10 / sma20 / sma100 – Simple Moving Averages
    # ------------------------------------------------------------------
    out['sma10'] = _safe(lambda: adj_close.rolling(10).mean(), idx)
    out['sma20'] = _safe(lambda: adj_close.rolling(20).mean(), idx)
    out['sma100'] = _safe(lambda: adj_close.rolling(100).mean(), idx)

    # ------------------------------------------------------------------
    # 28. stochrsi_k – Stochastic RSI %K n=14
    # ------------------------------------------------------------------
    def _stochrsi():
        df = ta.stochrsi(adj_close, length=14)
        col = [c for c in df.columns if c.startswith('STOCHRSIk')][0]
        return df[col]

    out['stochrsi_k'] = _safe(_stochrsi, idx)

    # ------------------------------------------------------------------
    # 29-30. supertrend_upper / supertrend_lower – n=10 mult=3
    # ------------------------------------------------------------------
    out['supertrend_upper'], out['supertrend_lower'] = _supertrend_bands(
        adj_high, adj_low, adj_close, length=10, multiplier=3.0
    )

    # ------------------------------------------------------------------
    # 31. fisher – Gaussian Fisher Transform n=9
    # ------------------------------------------------------------------
    def _fisher():
        df = ta.fisher(adj_high, adj_low, length=9)
        col = [c for c in df.columns if c.startswith('FISHERT_')][0]
        return df[col]

    out['fisher'] = _safe(_fisher, idx)

    # ------------------------------------------------------------------
    # 32. tema – Triple Exponential Moving Average n=30
    # ------------------------------------------------------------------
    out['tema'] = _safe(lambda: ta.tema(adj_close, length=30), idx)

    # ------------------------------------------------------------------
    # 33. vvi – Volume Variation Index n=20
    # ------------------------------------------------------------------
    def _vvi(n: int = 20) -> pd.Series:
        vol_mean = volume.rolling(n).mean()
        ratio = volume / vol_mean
        return ratio.ewm(span=n, adjust=False).mean()

    out['vvi'] = _safe(lambda: _vvi(20), idx)

    # ------------------------------------------------------------------
    # 34. zscore_close – Z-score of close n=75
    # ------------------------------------------------------------------
    def _zscore(n: int = 75) -> pd.Series:
        mu = adj_close.rolling(n).mean()
        sigma = adj_close.rolling(n).std()
        return (adj_close - mu) / sigma.replace(0, np.nan)

    out['zscore_close'] = _safe(lambda: _zscore(75), idx)

    # ------------------------------------------------------------------
    # Assemble and return
    # ------------------------------------------------------------------
    return pd.DataFrame(out, index=idx)[FEATURE_NAMES]


def build_feature_matrix(
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
) -> Dict[str, pd.DataFrame]:
    """Compute 34 features for every ticker in the universe.

    Args:
        data_dict: Dict with at minimum the keys ``'open'``, ``'high'``,
            ``'low'``, ``'adj_close'``, and ``'volume'`` — each a
            :class:`pd.DataFrame` indexed by date with tickers as columns.
        tickers: Ordered list of ticker symbols to process.

    Returns:
        Dict mapping each ticker symbol to its 34-column feature
        :class:`pd.DataFrame`.
    """
    adj_open = data_dict['open']
    adj_high = data_dict['high']
    adj_low = data_dict['low']
    adj_close = data_dict['adj_close']
    volume = data_dict['volume']

    result: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        result[ticker] = compute_features_for_asset(
            adj_open=adj_open[ticker],
            adj_high=adj_high[ticker],
            adj_low=adj_low[ticker],
            adj_close=adj_close[ticker],
            volume=volume[ticker],
        )
    return result
