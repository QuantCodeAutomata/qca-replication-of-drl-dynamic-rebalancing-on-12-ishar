# DRL Dynamic Rebalancing — Replication of Yu & Chang (2025)

Replication of **"Smart Tangency Portfolio: Deep Reinforcement Learning for Dynamic Rebalancing"** (Yu & Chang, 2025) on two asset universes:

- **exp_1** — 12 iShares U.S. Sector ETFs (IYW, IYF, IYZ, IYM, IYK, IYC, IYE, IYG, IYH, IYJ, IDU, IYR)  
- **exp_2** — 28 static DJIA component stocks (fixed as of Feb 2024)  
- **exp_3** — Sensitivity analysis over unresolved paper ambiguities  

## Research Questions

1. Does DRL-guided control of frontier position (λ) and rebalancing horizon (h) outperform static mean-risk benchmarks?  
2. Does ensemble model selection by median validation Sharpe improve OOS robustness?  
3. Which paper ambiguities (semivariance benchmark, lambda mapping, estimation window, evaluation policy) have first-order effects on replication outcomes?

## Repository Structure

```
├── data/               # Data download scripts (yfinance cache)
├── src/
│   ├── data_module/    # Download, align, adjusted-OHLC
│   ├── features/       # 34 technical indicators + 252-day rolling z-score
│   ├── optimizer/      # MV, Semivariance, CVaR solvers (cvxpy); frontier mapper
│   ├── rl_env/         # Gymnasium portfolio environment; SB3 PPO/A2C wrappers
│   ├── backtest/       # Rolling windows; static benchmarks
│   ├── stats/          # Performance metrics; alpha regression; Sharpe bootstrap
│   ├── viz/            # matplotlib plots
│   └── pipeline.py     # Main integration module
├── exp/
│   ├── run_experiment.py    # exp_1 + exp_2 runner
│   └── run_sensitivity.py   # exp_3 sensitivity study
├── tests/              # pytest tests (115+ tests, all passing)
├── results/            # Output tables, plots, RESULTS.md
├── AGENTS.md           # Persistent agent memory
└── README.md
```

## Methodology Summary

### DRL Strategy
- **Action space**: `MultiDiscrete([100, 56])` — λ ∈ {1,…,100} (frontier index) × h ∈ {5,…,60} (rebalancing horizon)
- **6 models**: PPO/A2C × Mean-Variance / Mean-Semivariance / Mean-CVaR
- **Reward**: `log(PV_{t+h} / PV_t)` — cumulative log return over hold period
- **Ensemble**: Single best model by median validation Sharpe (model selection, not averaging)
- **Rolling design**: 2-year train / 2-year validation / 2-year test, advancing by 2 years

### Portfolio Optimizers
- **Mean-Variance**: `min w'Σw` s.t. `sum(w)=1, w>=0, w'μ >= μ₀`
- **Mean-Semivariance**: `min w'SCOV w` with `SCOV[i,j] = (1/T)Σ min(r_it−B,0)·min(r_jt−B,0)`, B=0
- **Mean-CVaR**: Rockafellar–Uryasev LP, β=0.95

### Documented Ambiguity Resolutions

| Ambiguity | Choice |
|-----------|--------|
| Semivariance B | B=0 (base case; sensitivity tested) |
| Lambda mapping | `linspace(min_μ, max_μ, 100)` |
| Estimation window | 252 trading days |
| Adjusted OHLC | `factor = AdjClose/Close` applied to O/H/L |
| Test evaluation | Deterministic |
| Tie-breaking in ensemble | Alphabetical by model name |
| Newey-West lags | 20 |
| Bootstrap replications | 500 |

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `stable-baselines3`, `gymnasium`, `cvxpy`, `pandas-ta`, `yfinance`, `statsmodels`, `matplotlib`, `numpy`, `pandas`.

## Running Experiments

```bash
# ETF universe (exp_1)
python exp/run_experiment.py --universe etf

# DJIA universe (exp_2)
python exp/run_experiment.py --universe djia

# Both
python exp/run_experiment.py --universe both

# Sensitivity study (exp_3)
python exp/run_sensitivity.py --universe etf

# Control training budget (default 5000 timesteps)
DRL_TIMESTEPS=20000 python exp/run_experiment.py --universe etf
```

## Running Tests

```bash
pytest tests/ -v
```

115+ tests across data, features, optimizer, environment, backtest, statistics, and pipeline integration. All pass without network access (synthetic data only).

## Expected Results

See `results/RESULTS.md` for full metrics table after running experiments.

Paper-scale targets:
- **ETF Ensemble**: Annual return ~11.3%, Vol ~19.1%, Sharpe ~0.59
- **ETF PPO MV**: Annual return ~10.7%, Vol ~18.2%, Sharpe ~0.59
- **DJIA PPO MV**: Annual return ~15.7%, Vol ~17.7%, Sharpe ~0.887

> ⚠️ Note: Paper specifies 10,000 training *episodes* (≈170,000 SB3 timesteps). Default runs use fewer timesteps for feasibility; results are directionally consistent but not fully converged. Set `DRL_TIMESTEPS=170000` for full replication.

## Survivorship Bias Note (exp_2)

The DJIA universe uses a **static** 28-stock universe fixed as of Feb 2024. This intentionally replicates the paper's setup but introduces survivorship bias. This is documented as a faithful replication choice, not an oversight.

## License

MIT
