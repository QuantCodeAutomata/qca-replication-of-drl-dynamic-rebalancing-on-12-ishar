"""Rolling z-score standardization and observation tensor construction."""

from typing import Dict

import numpy as np
import pandas as pd


def rolling_zscore_standardize(
    feature_df: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Apply a trailing rolling z-score normalization to each feature column.

    Uses only past observations (no look-ahead): each row is normalized
    using the mean and standard deviation of the preceding ``window`` rows.
    Columns with zero rolling standard deviation are set to 0 rather than
    producing ``inf`` or ``NaN``.

    Args:
        feature_df: :class:`pd.DataFrame` of raw feature values (rows = dates,
            columns = feature names).
        window: Look-back window in trading days. Default: ``252`` (1 year).

    Returns:
        :class:`pd.DataFrame` of the same shape as *feature_df* containing
        z-scored values.  The first ``window - 1`` rows will be ``NaN`` for
        features whose rolling statistics require more observations.
    """
    roll_mean = feature_df.rolling(window=window, min_periods=window).mean()
    roll_std = feature_df.rolling(window=window, min_periods=window).std()
    # Divide safely: replace 0 std with NaN so we get NaN (not inf) in those cells
    safe_std = roll_std.where(roll_std != 0.0, other=np.nan)
    z = (feature_df - roll_mean) / safe_std
    # Where roll_std was exactly zero (window is full but series is constant),
    # the standardized value is defined as 0 rather than NaN.
    zero_std_mask = (roll_std == 0.0) & roll_std.notna()
    z = z.where(~zero_std_mask, other=0.0)
    return z


def build_observation_tensor(
    standardized_features: Dict[str, pd.DataFrame],
    tickers: list,
    date: pd.Timestamp,
    lookback: int = 252,
) -> np.ndarray:
    """Build a 3-D observation tensor for a single time step.

    Stacks the last ``lookback`` rows of each asset's feature DataFrame into a
    tensor of shape ``(n_features, n_assets, lookback)``.

    Args:
        standardized_features: Dict mapping ticker symbol to its standardized
            feature :class:`pd.DataFrame` (rows = dates, columns = feature names).
        tickers: Ordered list of ticker symbols; determines the asset axis
            ordering in the returned tensor.
        date: The current date (inclusive upper bound of the look-back window).
        lookback: Number of trading days to include. Default: ``252``.

    Returns:
        :class:`np.ndarray` of shape ``(n_features, n_assets, lookback)``
        with ``dtype=float32``.

    Raises:
        ValueError: If ``date`` is not found in a ticker's feature DataFrame or
            if fewer than ``lookback`` rows are available before ``date``.
    """
    n_assets = len(tickers)
    # Determine n_features from the first available ticker
    ref_df = standardized_features[tickers[0]]
    n_features = ref_df.shape[1]

    tensor = np.empty((n_features, n_assets, lookback), dtype=np.float32)

    for a_idx, ticker in enumerate(tickers):
        df = standardized_features[ticker]
        # Locate the row corresponding to `date`
        if date not in df.index:
            raise ValueError(f"Date {date} not found in feature DataFrame for {ticker}.")
        loc = df.index.get_loc(date)
        if loc + 1 < lookback:
            raise ValueError(
                f"Not enough history for {ticker} at {date}: "
                f"need {lookback} rows, have {loc + 1}."
            )
        window_df = df.iloc[loc + 1 - lookback: loc + 1]
        # Shape: (lookback, n_features) → (n_features, lookback)
        tensor[:, a_idx, :] = window_df.to_numpy(dtype=np.float32).T

    return tensor
