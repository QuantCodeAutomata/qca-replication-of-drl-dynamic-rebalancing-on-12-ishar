# Sensitivity Study — ETF Universe

## Summary: Impact on Sharpe Ratio by Ambiguity Dimension

| Config | Sharpe | AnnRet | Sharpe Δ vs Base |
|--------|--------|--------|-----------------|
| base | 0.127 | 0.028 | 0.000 |
| B=mean | 0.127 | 0.028 | 0.000 |
| B=0 | 0.127 | 0.028 | 0.000 |
| window=126 | 0.340 | 0.074 | 0.213 |
| window=252 | 0.127 | 0.028 | 0.000 |
| window=504 | 0.027 | 0.007 | -0.101 |
| det_eval | 0.127 | 0.028 | 0.000 |
| stoch_eval | 0.127 | 0.028 | 0.000 |
| lambda_minmax | 0.127 | 0.028 | 0.000 |
| lambda_pctile | 0.127 | 0.028 | 0.000 |

## Interpretation

- Rows with large |Sharpe Δ| indicate first-order ambiguities.
- Estimation window and lambda mapping are expected to be most influential.
- Deterministic vs stochastic evaluation affects val-Sharpe ranking but not policy quality.
