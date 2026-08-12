# DRL Dynamic Rebalancing – Repository Notes (Yu & Chang 2025)

## Project Layout

```
src/
  data_module/         # Download & preprocessing pipeline
    downloader.py      # yfinance download → dict of OHLCV DataFrames
    preprocessor.py    # log returns, OHLC adjustment, calendar alignment
  features/
    feature_builder.py # 34 per-asset technical indicators
    standardizer.py    # rolling z-score + observation tensor builder
  rl_env/              # Gymnasium trading environment
  optimizer/           # MVO / CVaR / semi-variance solvers
  stats/               # Performance metrics
  backtest/            # Back-test harness
data/
  download_etf.py      # Run to fetch 12-ETF universe (needs internet)
  download_djia.py     # Run to fetch 28-stock DJIA universe (needs internet)
tests/                 # pytest suite (all synthetic data – no network needed)
```

## Key Design Decisions

### Data Download (`downloader.py`)
- `yfinance.download(auto_adjust=False)` – keeps both raw Close **and** Adj Close so
  OHLC prices can be adjusted by the factor `AdjClose / Close`.
- Returns `(data_dict, benchmark_series)` tuple.
- Cache stored as `raw.parquet` under `cache_dir/` — second call reads from disk.
- Parquet requires **pyarrow** (`pip install pyarrow`).

### Feature Builder (`feature_builder.py`)
- 34 features per asset, ordered exactly as `FEATURE_NAMES`.
- Uses **pandas_ta** for most indicators; manual numpy/pandas for VVI, z-score,
  Elder-Ray, and a few others.
- Every indicator is wrapped in `_safe(fn, idx)` which returns a NaN Series on
  any exception, so one broken indicator never kills the whole pipeline.
- Column name extraction from pandas_ta results uses `startswith()` prefix
  matching rather than hardcoded strings to be robust to version differences.

### Standardizer (`standardizer.py`)
- `rolling_zscore_standardize`: trailing window only; first `window-1` rows are NaN
  (insufficient history).  Cells where rolling std == 0 (constant subseries) are
  set to **0** not NaN.
- `build_observation_tensor` → shape `(n_features, n_assets, lookback)`, `float32`.

## pandas_ta Column Name Conventions (installed version)
| Indicator     | Column used                       |
|---------------|-----------------------------------|
| aroon(25)     | `AROONOSC_25`                     |
| bbands(20,2)  | `BBU_20_*` / `BBL_20_*` (prefix) |
| supertrend    | `SUPERTs_*` (upper) / `SUPERTl_*` (lower) |
| macd(12,26,9) | `MACD_12_26_9`                    |
| kst()         | `KST_*` (first col)               |
| stochrsi(14)  | `STOCHRSIk_14_14_3_3`             |
| rvgi(10)      | `RVGIs_10_4` (signal)             |
| fisher(9)     | `FISHERT_9_1`                     |
| qqe()         | `QQE_14_5_4.236`                  |
| adx(30)       | `ADX_30`                          |
| pvo()         | `PVO_12_26_9`                     |

## Running Tests
```bash
python -m pytest tests/ -v        # 50 tests, no network required
```

## Installing Dependencies
```bash
pip install yfinance pandas_ta pyarrow
```
## Backtest Module (`src/backtest/`)

### `benchmarks.py`
| Function | Description |
|---|---|
| `compute_static_benchmark(log_returns_df, test_dates, risk_model, ...)` | Re-optimises every `rebalance_every` (default 30) days using trailing 252-day window; returns DataFrame with `portfolio_value`, `daily_return`, `weights` |
| `compute_buy_hold_benchmark(benchmark_returns, test_dates, ...)` | Cumulative value for a buy-and-hold index (SPY / DIA) |

Key design choices:
- Falls back to **equal weighting** when trailing window has < 20 observations (avoids calling the optimiser on tiny samples).
- `mu_target = 0.9 * max(mu)` approximates a tangency portfolio without an explicit risk-free rate.
- `risk_model` in `{'variance', 'semivariance', 'cvar'}`.

### `rolling_backtest.py`
| Symbol | Description |
|---|---|
| `WindowConfig` | Dataclass with 6 timestamps + `window_idx` |
| `build_rolling_windows(all_dates, universe)` | Returns list of `WindowConfig`; skips windows with < 20 test dates |
| `get_window_dates(all_dates, window, segment)` | Slice `all_dates` to `'train'`/`'val'`/`'test'` |

Window schedules: `'etf'` base_year=2003, 8 windows; `'djia'` base_year=2007, 6 windows; 2-year advance.

## Stats Module (`src/stats/`)

### `metrics.py`
Core metrics (always computed): `annualized_return`, `annualized_volatility`, `var_95`, `cvar_95`, `max_drawdown`, `sharpe_ratio`, `sortino_ratio`.
Relative metrics (need `benchmark_returns`): `beta`, `tracking_error`, `information_ratio`, `upside_capture`, `downside_capture`.
Edge cases: returns `{}` for < 2 observations; `sharpe_ratio = 0` when `std < 1e-10`.

### `statistical_tests.py`
| Function | Description |
|---|---|
| `alpha_regression_newey_west(strategy_returns, benchmark_returns, n_lags=20)` | OLS with HAC/Newey-West SEs via statsmodels |
| `sharpe_difference_bootstrap(returns_a, returns_b, n_bootstrap=1000, block_length=20)` | Stationary block bootstrap for one-sided Sharpe superiority test |

**CRITICAL**: Do NOT use `import statsmodels.api as sm` on Python 3.12 — it triggers a
`deprecate_kwarg` bug in `statsmodels.tsa.stattools`. Use direct imports:
```python
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac
```

## Viz Module (`src/viz/`)

### `plots.py`
| Function | Description |
|---|---|
| `plot_portfolio_value(portfolio_values, ...)` | Line chart of cumulative USD value for multiple strategies |
| `plot_dynamic_weights(weights_df, tickers, ...)` | Stacked area chart of portfolio weights |
| `plot_lambda_horizon(lambda_series, horizon_series, ...)` | Two-panel scatter: lambda_t and h_t over time |
| `plot_performance_table(metrics_df, ...)` | Matplotlib table rendering of a metrics DataFrame |
| `plot_sensitivity_tornado(sensitivity_df, metric, ...)` | Horizontal tornado bar chart |

All functions accept optional `save_path`; parent directory is auto-created with `os.makedirs`.

## Running Tests (updated total)
```bash
python -m pytest tests/ -v        # includes test_stats.py (10 tests, all synthetic)
```
`test_benchmark_computation` is designed to trigger the equal-weight fallback (30-day
window, estimation_window=252) so no optimiser calls are needed in tests.
