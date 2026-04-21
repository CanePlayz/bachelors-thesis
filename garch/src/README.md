# GARCH(1,1)-X with Mean-Group aggregation

Per-stock GARCH(1,1)-X estimation with social-media variables (attention,
positive/negative sentiment, controls) entering the variance equation as
exogenous regressors. Population effects are aggregated via the Mean-Group
(MG) estimator of Pesaran & Smith (1995). Likelihood-ratio homogeneity tests
on the pooled-parameter panel-GARCH are run separately.

## Model

For stock $i$ on trading day $t$:

$$r_{i,t} = \mu_i + \varepsilon_{i,t}, \qquad
  \sigma^2_{i,t} = \omega_i + \alpha_i\,\varepsilon^2_{i,t-1}
                  + \beta_i\,\sigma^2_{i,t-1}
                  + \boldsymbol{\delta}_i' \, \mathbf{x}_{i,t-1}$$

Estimation is done one stock at a time via Gaussian quasi-maximum likelihood
(`arch` package, L-BFGS-B). The cross-sectional average of the per-stock
$\boldsymbol{\delta}_i$ estimates is the Mean-Group estimator; standard errors
follow Pesaran & Smith (1995) and a sample-weighted variant is reported
alongside.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Spec dataclass, default exogenous variables, paths, robustness lists |
| `data.py` | Load the regression panel, slice it into per-stock arrays |
| `baseline.py` | Per-stock GARCH(1,1)-X via `arch.arch_model` (Gaussian QML) |
| `model.py` | Mean-Group aggregation, LaTeX/JSON result writers |
| `model_panel.py` | Pooled-parameter panel-GARCH used by `run_lr_tests.py` |
| `run_garch.py` | CLI: primary model + robustness checks |
| `run_lr_tests.py` | CLI: LR tests for $\alpha$, $\beta$, $\delta$ homogeneity |

## Usage

```bash
# Primary model + paper robustness checks
python run_garch.py

# Primary only
python run_garch.py --skip-robustness

# Override the per-stock minimum sample size
python run_garch.py --min-obs 200

# Likelihood-ratio homogeneity tests (alpha, beta, delta)
python run_lr_tests.py
```

## Robustness checks

Exactly the two GARCH-side robustness families discussed in §6 of the thesis:

1. **Alternative sentiment models:** `bertweet`, `fintwit`.
2. **Alternative attention specifications:** absolute/relative, global /
   within-stock standardisation.

(Volatility-measure robustness is panel-only — see `regression/`.)

## Output

Per run, under `out/garch/<run-name>/`:

- `stock_garch_results.csv` / `.parquet` — per-stock parameter estimates.
- `mean_group_results.json` — MG and sample-weighted aggregates.
- `mean_group_table.tex` — LaTeX table of MG estimates.

LR-test outputs are written under `out/garch/lr_tests/`.

## References

- Pesaran, M. H., & Smith, R. (1995). Estimating long-run relationships from
  dynamic heterogeneous panels. *Journal of Econometrics*, 68(1), 79–113.
- Bollerslev, T. (1986). Generalized autoregressive conditional
  heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
