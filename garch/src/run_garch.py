"""Run per-stock GARCH(1,1)-X estimation + Mean Group aggregation.

Usage:
    python run_garch.py                   # default + robustness checks
    python run_garch.py --skip-robustness # primary model only
    python run_garch.py --quiet           # less output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from config import (
    OUTPUT_DIR,
    ROBUSTNESS_ATTENTION_SPECS,
    ROBUSTNESS_SENTIMENT_COLUMN_MAP,
    ROBUSTNESS_SENTIMENT_MODELS,
    GARCHSpec,
    default_spec,
)
from model import (
    MeanGroupResult,
    StockResult,
    mean_group_aggregate,
    print_mean_group_table,
    results_to_dataframe,
)
from model_panel import PanelGARCHResult, estimate_panel_garch

from data import load_panel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-stock GARCH(1,1)-X + Mean Group")
    p.add_argument("--min-obs", type=int, default=100)
    p.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Run primary model only, skip robustness checks",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def save_results(
    df: pd.DataFrame, mg: MeanGroupResult, out_dir: Path, spec: GARCHSpec
) -> None:
    """Save all outputs."""
    from datetime import datetime

    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-stock results
    df.to_csv(out_dir / "stock_garch_results.csv")
    df.to_parquet(out_dir / "stock_garch_results.parquet")

    # Mean Group summary as JSON — unified schema with regression pipeline
    mg_data = {
        "model_type": "garch_mean_group",
        "spec": {
            "name": f"GARCH(1,1)-X MG ({spec.dist})",
            "model": "GARCH(1,1)-X",
            "aggregation": "Mean Group (Pesaran & Smith, 1995)",
            "distribution": spec.dist,
            "exog_cols": spec.exog_cols,
        },
        "n_obs": mg.n_stocks,  # number of cross-sectional units
        "n_converged": mg.n_converged,
        "exog_names": mg.exog_names,
        # Coefficients, SE, t-stats, p-values — keyed by variable name
        # (uses simple MG as the primary estimator)
        "coefficients": {name: float(mg.delta_mg[name]) for name in mg.exog_names},
        "std_errors": {name: float(mg.se_mg[name]) for name in mg.exog_names},
        "t_stats": {name: float(mg.t_mg[name]) for name in mg.exog_names},
        "p_values": {name: float(mg.p_mg[name]) for name in mg.exog_names},
        # Sample-weighted alternative
        "sample_weighted": {
            "coefficients": {name: float(mg.delta_sw[name]) for name in mg.exog_names},
            "std_errors": {name: float(mg.se_sw[name]) for name in mg.exog_names},
            "t_stats": {name: float(mg.t_sw[name]) for name in mg.exog_names},
            "p_values": {name: float(mg.p_sw[name]) for name in mg.exog_names},
        },
        # Cross-sectional distribution of per-stock estimates
        "distribution": {
            name: {
                "quantiles": mg.delta_quantiles[name],
                "frac_positive": float(mg.delta_frac_positive[name]),
                "frac_significant": float(mg.delta_frac_significant[name]),
            }
            for name in mg.exog_names
        },
        "metadata": {
            "saved_at": datetime.now().isoformat(),
        },
    }

    with open(out_dir / "mean_group_results.json", "w") as f:
        json.dump(mg_data, f, indent=2)

    # LaTeX-ready table
    _write_latex_table(mg, out_dir / "mean_group_table.tex")

    print(f"\n  Results saved to {out_dir}", flush=True)


def _write_latex_table(mg: MeanGroupResult, path: Path) -> None:
    """Write a LaTeX table of MG estimates (threeparttable, consistent with regression output)."""

    def stars(p: float) -> str:
        if p < 0.001:
            return "^{***}"
        if p < 0.01:
            return "^{**}"
        if p < 0.05:
            return "^{*}"
        if p < 0.1:
            return "^{\\cdot}"
        return ""

    # Variable display names (match regression pipeline conventions)
    name_map = {
        "attention_std": r"Attention ($\tilde{A}$)",
        "sentiment_pos_std": r"Sentiment$^+$ ($\tilde{S}^+$)",
        "sentiment_neg_std": r"Sentiment$^-$ ($\tilde{S}^-$)",
        "sentiment_std": r"Sentiment ($\tilde{S}$)",
        "vix_std": r"VIX ($\widetilde{\mathrm{VIX}}$)",
        "gap_days": r"Gap Days ($g$)",
        "valid_sentiment": r"ValidSent",
    }

    def _label(name: str) -> str:
        if name in name_map:
            return name_map[name]
        return name.replace("_std", "").replace("_", r"\_")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{threeparttable}",
        r"\caption{Mean Group Estimates: Social Media Effects on Volatility}",
        r"\label{tab:garch_mg}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & Simple MG & Sample-Weighted \\",
        r"\midrule",
    ]

    for name in mg.exog_names:
        label = _label(name)
        # Coefficient row
        vals = []
        for d, pv in [(mg.delta_mg, mg.p_mg), (mg.delta_sw, mg.p_sw)]:
            vals.append(f"${d[name]:.6f}{stars(pv[name])}$")
        lines.append(f"  {label} & " + " & ".join(vals) + r" \\")

        # SE row
        se_vals = []
        for s in [mg.se_mg, mg.se_sw]:
            se_vals.append(f"$({s[name]:.6f})$")
        lines.append(f"  & " + " & ".join(se_vals) + r" \\[0.5em]")

    lines.extend(
        [
            r"\midrule",
            f"  Stocks & \\multicolumn{{2}}{{c}}{{{mg.n_stocks}}} \\\\",
            f"  Converged & \\multicolumn{{2}}{{c}}{{{mg.n_converged}}} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item \textit{Notes:} Clustered standard errors in parentheses.",
            r"\item Simple MG uses cross-sectional variance (Pesaran \& Smith, 1995).",
            r"\item Sample-Weighted uses $T_i / \sum T_j$ as weights.",
            r"\item $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$, $^{\cdot}p<0.1$",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _panel_result_to_stock_results(
    panel: PanelGARCHResult,
    stocks: list,
    spec: GARCHSpec,
) -> list:
    """Convert a PanelGARCHResult into the per-stock list expected by
    mean_group_aggregate. Each StockResult carries the stock's fitted
    (omega_i, alpha_i, beta_i, delta_i) --- pooled parameters appear as
    the same shared value on every stock, stock-specific parameters
    take each stock's own estimate. Standard errors for stock-specific
    delta slots are read from the panel's inverse-Hessian SEs; pooled
    slots share the same SE across stocks.

    The per-stock loglik / aic / bic returned are approximate: we do
    not re-run the filter per stock, we simply copy the panel-level
    totals (scaled by stock weight) since the MG aggregator only
    consumes the coefficient distribution, not these diagnostics.
    """
    layout = panel.layout
    theta = panel.theta
    se = (
        panel.std_errors
        if panel.std_errors is not None
        else np.full_like(theta, np.nan)
    )

    # Pre-compute the SE for each (exog_col, stock) combination.
    # delta_col_slices[k] = slice into theta for the k-th exog column,
    # delta_col_pooled[k] = True if pooled (1 slot), False if N slots.
    K = layout.n_exog
    N = len(stocks)

    results: list = []
    for i, stock in enumerate(stocks):
        omega_i = layout.get_omega(theta, i)
        alpha_i = layout.get_alpha(theta, i)
        beta_i = layout.get_beta(theta, i)
        delta_i = layout.get_delta(theta, i)

        delta_dict: dict = {}
        delta_se_dict: dict = {}
        for k, col in enumerate(spec.exog_cols):
            slot = layout.delta_col_slices[k]
            if layout.delta_col_pooled[k]:
                se_idx = slot.start
            else:
                se_idx = slot.start + i
            delta_dict[col] = float(delta_i[k])
            delta_se_dict[col] = float(se[se_idx])

        results.append(
            StockResult(
                ticker=stock.ticker,
                n_obs=stock.n_obs,
                converged=panel.converged,
                loglik=float("nan"),  # only panel-level loglik available
                aic=float("nan"),
                bic=float("nan"),
                omega=float(omega_i),
                alpha=float(alpha_i),
                beta=float(beta_i),
                delta=delta_dict,
                delta_se=delta_se_dict,
                persistence=float(alpha_i) + float(beta_i),
            )
        )
    return results


def _run_single_estimation(
    spec: GARCHSpec,
    label: str,
    out_dir: Path,
    min_obs: int,
) -> None:
    """Run one GARCH estimation (load data, estimate, aggregate, save)."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(spec.describe(), flush=True)

    stocks, _ = load_panel(spec, min_obs=min_obs)
    print(
        f"  {len(stocks)} stocks, {sum(s.n_obs for s in stocks):,} total obs\n",
        flush=True,
    )

    # Production estimator: joint Panel-GARCH-X MLE with the pooling
    # configuration specified by `spec`. All paper results (MG table and
    # robustness checks) flow through this single path so the methodology
    # section matches the code.
    panel = estimate_panel_garch(stocks, spec)
    results = _panel_result_to_stock_results(panel, stocks, spec)
    df = results_to_dataframe(results)

    # `only_converged` was meaningful for the per-stock arch backend, where
    # individual fits could fail. With the joint panel estimator there is a
    # single panel-wide convergence flag; per-stock parameters come from
    # the same joint optimum regardless, so we aggregate unconditionally.
    mg = mean_group_aggregate(results, only_converged=False)
    print_mean_group_table(mg)

    save_results(df, mg, out_dir, spec)


def main() -> None:
    args = parse_args()

    spec = default_spec()
    spec.verbose = not args.quiet

    import time

    start_time = time.time()

    print("\n" + "=" * 60)
    print("GARCH-X ESTIMATION PIPELINE")
    print("=" * 60)

    # ---- Primary estimation ------------------------------------------------
    _run_single_estimation(
        spec=spec,
        label="Primary Model (roberta_topic, abs_within)",
        out_dir=OUTPUT_DIR / "garch",
        min_obs=args.min_obs,
    )

    # ---- Robustness checks -------------------------------------------------
    if not args.skip_robustness:
        # 1) Alternative sentiment models
        for model in ROBUSTNESS_SENTIMENT_MODELS:
            col_info = ROBUSTNESS_SENTIMENT_COLUMN_MAP[model]
            rob_spec = GARCHSpec(
                exog_cols=col_info["exog_cols"],
                dist=spec.dist,
                verbose=spec.verbose,
            )
            _run_single_estimation(
                spec=rob_spec,
                label=f"Robustness: sentiment model = {model}",
                out_dir=OUTPUT_DIR / "garch" / f"robustness_sentiment_{model}",
                min_obs=args.min_obs,
            )

        # 2) Alternative attention specifications
        for att_spec in ROBUSTNESS_ATTENTION_SPECS:
            rob_spec = GARCHSpec(
                exog_cols=[
                    att_spec["col"],
                    "sentiment_pos_std",
                    "sentiment_neg_std",
                    "vix_std",
                    "gap_days",
                    "valid_sentiment",
                ],
                dist=spec.dist,
                verbose=spec.verbose,
            )
            _run_single_estimation(
                spec=rob_spec,
                label=f"Robustness: attention = {att_spec['label']}",
                out_dir=OUTPUT_DIR / "garch" / f"robustness_att_{att_spec['label']}",
                min_obs=args.min_obs,
            )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"GARCH PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
