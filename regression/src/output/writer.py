"""Output writer for regression results.

This module provides functions to save regression results in multiple formats:
- JSON: Full results with all statistics
- CSV: Summary tables for import to Excel/Stata
- LaTeX: Publication-ready, self-contained tables that can be compiled standalone

Output organization:
    out/regression/raw/{type}/{horizon}/
        - volatility_h1_roberta_topic.json
        - ...
    out/regression/tables/
        - summary_baseline.csv
        - summary_baseline.tex
        - regression_results.tex (standalone compilable document)
        - ...
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from regression.specs import RegressionResult

# Robustness key prefixes for the alternative volatility measures
# (one per robustness measure registered in VolatilityConfig.robustness_measures).
_VOL_ROBUSTNESS_PREFIXES = (
    "robustness_ja_rs_",
    "robustness_ja_pk_",
    "robustness_gk_",
    "robustness_pk_",
    "robustness_sqret_",
)


def _is_vol_robustness_key(key: str) -> bool:
    return key.startswith(_VOL_ROBUSTNESS_PREFIXES)


def save_all_results(
    results: Dict[str, List[RegressionResult]],
    output_dir: Path,
    model_name: str = "roberta_topic",
) -> None:
    """Save all regression results to organized output structure.

    Args:
        results: Dictionary mapping regression type to list of results
        output_dir: Base output directory
        model_name: Sentiment model name (for file naming)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    # Save individual results
    for reg_type, result_list in results.items():
        for result in result_list:
            save_regression_results(
                result,
                output_dir=output_dir / "raw" / reg_type,
                model_name=model_name,
            )

    # Save summary tables
    all_results = []
    for result_list in results.values():
        all_results.extend(result_list)

    if all_results:
        save_summary_csv(all_results, output_dir / "tables" / "summary_all.csv")
        save_summary_latex(all_results, output_dir / "tables" / "summary_all.tex")
        save_standalone_latex_document(
            results, output_dir / "tables" / "regression_results.tex"
        )

    # Save baseline summary
    baseline_results = results.get("baseline_volatility", [])
    if baseline_results:
        save_summary_csv(
            baseline_results, output_dir / "tables" / "summary_baseline.csv"
        )
        save_summary_latex(
            baseline_results, output_dir / "tables" / "summary_baseline.tex"
        )

    # Save robustness summaries by dimension
    robustness_sent = []
    robustness_vol = []
    robustness_att = []
    for key, res_list in results.items():
        if not key.startswith("robustness_"):
            continue
        if key.startswith("robustness_att_"):
            robustness_att.extend(res_list)
        elif _is_vol_robustness_key(key):
            robustness_vol.extend(res_list)
        else:
            robustness_sent.extend(res_list)

    if robustness_sent:
        save_summary_csv(
            robustness_sent, output_dir / "tables" / "summary_robustness_sentiment.csv"
        )
        save_summary_latex(
            robustness_sent,
            output_dir / "tables" / "summary_robustness_sentiment.tex",
            caption="Robustness: Alternative Sentiment Models",
            label="tab:robustness_sentiment",
        )
    if robustness_vol:
        save_summary_csv(
            robustness_vol, output_dir / "tables" / "summary_robustness_volatility.csv"
        )
        save_summary_latex(
            robustness_vol,
            output_dir / "tables" / "summary_robustness_volatility.tex",
            caption="Robustness: Alternative Volatility Measures",
            label="tab:robustness_volatility",
        )
    if robustness_att:
        save_summary_csv(
            robustness_att, output_dir / "tables" / "summary_robustness_attention.csv"
        )
        save_summary_latex(
            robustness_att,
            output_dir / "tables" / "summary_robustness_attention.tex",
            caption="Robustness: Alternative Attention Specifications",
            label="tab:robustness_attention",
        )

    print(f"  → Saved {len(all_results)} regression results")


def save_regression_results(
    result: RegressionResult,
    output_dir: Path,
    model_name: str = "roberta_topic",
) -> Path:
    """Save a single regression result to JSON.

    Args:
        result: Regression result to save
        output_dir: Output directory
        model_name: Sentiment model name

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filename: {name}_{model}.json
    filename = f"{result.spec.name}_{model_name}.json"
    filepath = output_dir / filename

    # Convert to dict and add metadata
    data = result.to_dict()
    data["model_type"] = "panel_ols"
    data["metadata"] = {
        "saved_at": datetime.now().isoformat(),
        "sentiment_model": model_name,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return filepath


def save_summary_csv(
    results: List[RegressionResult],
    filepath: Path,
) -> Path:
    """Save summary table as CSV.

    Args:
        results: List of regression results
        filepath: Output file path

    Returns:
        Path to saved file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    rows = [r.to_summary_row() for r in results]
    df = pd.DataFrame(rows)

    df.to_csv(filepath, index=False)

    return filepath


def save_summary_latex(
    results: List[RegressionResult],
    filepath: Path,
    caption: str = "Regression Results",
    label: str = "tab:regression_results",
) -> Path:
    """Save summary table as LaTeX.

    Args:
        results: List of regression results
        filepath: Output file path
        caption: Table caption
        label: Table label

    Returns:
        Path to saved file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append("")

    if results:
        lines.extend(_format_latex_subtable(results))
        lines.append("")

    lines.append(r"\end{table}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath


def _format_latex_subtable(results: List[RegressionResult]) -> List[str]:
    """Format a subset of results as LaTeX table rows."""
    if not results:
        return []

    # Get all unique regressors across results
    all_regressors = set()
    for r in results:
        all_regressors.update(r.spec.regressors)
    regressors = sorted(all_regressors)

    # Build table
    n_cols = 1 + len(results)  # Variable name + one column per horizon
    col_spec = "l" + "c" * len(results)

    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row
    header_parts = ["Variable"]
    for r in results:
        header_parts.append(f"h={r.spec.horizon}")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # Coefficient rows
    for var in regressors:
        # Variable name (escape underscores)
        var_name = var.replace("_", r"\_")

        # Coefficient row
        coef_parts = [var_name]
        for r in results:
            if var in r.coefficients:
                coef = r.coefficients[var]
                pval = r.p_values.get(var, 1.0)
                stars = _significance_stars(pval)
                coef_parts.append(f"{coef:.4f}{stars}")
            else:
                coef_parts.append("--")
        lines.append(" & ".join(coef_parts) + r" \\")

        # Standard error row
        se_parts = [""]
        for r in results:
            if var in r.std_errors:
                se = r.std_errors[var]
                se_parts.append(f"({se:.4f})")
            else:
                se_parts.append("")
        lines.append(" & ".join(se_parts) + r" \\")

    lines.append(r"\midrule")

    # Footer with R² and N
    obs_parts = ["Observations"]
    r2_parts = [r"$R^2$"]
    r2w_parts = [r"Within $R^2$"]
    n_ent_parts = ["Stocks"]

    for r in results:
        obs_parts.append(f"{r.n_obs:,}")
        r2_parts.append(f"{r.r_squared:.4f}")
        r2w_parts.append(f"{r.r_squared_within:.4f}")
        n_ent_parts.append(f"{r.n_entities:,}")

    lines.append(" & ".join(obs_parts) + r" \\")
    lines.append(" & ".join(r2_parts) + r" \\")
    lines.append(" & ".join(r2w_parts) + r" \\")
    lines.append(" & ".join(n_ent_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Add notes
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(r"\item Notes: Standard errors in parentheses, clustered by stock.")
    lines.append(r"\item $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$, $^{.}p<0.1$")
    lines.append(r"\end{tablenotes}")

    return lines


def _significance_stars(p: float) -> str:
    """Return significance stars based on p-value."""
    if p < 0.001:
        return "^{***}"
    elif p < 0.01:
        return "^{**}"
    elif p < 0.05:
        return "^{*}"
    elif p < 0.1:
        return "^{.}"
    else:
        return ""


def _strip_lag_suffix(var: str) -> str:
    """Strip _lagN suffix from variable name to get base name."""
    return re.sub(r"_lag\d+$", "", var)


def _resolve_var(result: "RegressionResult", base_var: str) -> str:
    """Resolve a base variable name to the actual key in a result's coefficients.

    Returns the base_var itself if present, otherwise tries base_var_lag{h}.
    """
    if base_var in result.coefficients:
        return base_var
    lag_var = f"{base_var}_lag{result.spec.horizon}"
    if lag_var in result.coefficients:
        return lag_var
    return base_var


# =============================================================================
# Standalone LaTeX Document
# =============================================================================


def save_standalone_latex_document(
    results: Dict[str, List[RegressionResult]],
    filepath: Path,
) -> Path:
    """Save a fully self-contained, compilable LaTeX document with all results.

    This document can be compiled directly with pdflatex to view all regression
    results at a glance.

    Args:
        results: Dictionary mapping regression type to list of results
        filepath: Output file path

    Returns:
        Path to saved file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    # Document preamble
    lines.append(r"\documentclass[11pt,a4paper]{article}")
    lines.append("")
    lines.append(r"\usepackage[margin=2cm]{geometry}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{amsmath}")
    lines.append(r"\usepackage{multirow}")
    lines.append(r"\usepackage{pdflscape}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\usepackage{threeparttable}")
    lines.append(r"\usepackage{caption}")
    lines.append(r"\usepackage{xcolor}")
    lines.append("")
    lines.append(r"\title{Regression Results Summary}")
    lines.append(r"\author{Auto-generated}")
    lines.append(r"\date{\today}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append(r"\tableofcontents")
    lines.append(r"\newpage")
    lines.append("")

    # Baseline Return Regressions
    if "baseline_return" in results and results["baseline_return"]:
        lines.append(r"\section{Baseline Return Regressions}")
        lines.append("")
        lines.extend(
            _format_standalone_table(
                results["baseline_return"],
                caption="Baseline Return Predictions",
                label="tab:baseline_return",
            )
        )
        lines.append(r"\newpage")
        lines.append("")

    # Baseline Volatility Regressions
    if "baseline_volatility" in results and results["baseline_volatility"]:
        lines.append(r"\section{Baseline Volatility Regressions}")
        lines.append("")
        lines.extend(
            _format_standalone_table(
                results["baseline_volatility"],
                caption="Baseline Volatility Predictions",
                label="tab:baseline_volatility",
            )
        )
        lines.append(r"\newpage")
        lines.append("")

    # Extension Volatility Regressions
    if "extension_volatility" in results and results["extension_volatility"]:
        lines.append(r"\section{Extended Volatility Regressions}")
        lines.append(r"Including sentiment dispersion and intensity.")
        lines.append("")
        lines.extend(
            _format_standalone_table(
                results["extension_volatility"],
                caption="Extended Volatility with Dispersion and Intensity",
                label="tab:extension_volatility",
            )
        )
        lines.append(r"\newpage")
        lines.append("")

    # Robustness checks — grouped by dimension
    robustness_keys = [k for k in results if k.startswith("robustness_")]

    if robustness_keys:
        lines.append(r"\section{Robustness Checks}")
        lines.append("")

        # ---- Sentiment model robustness ----
        sent_keys = [
            k
            for k in robustness_keys
            if not k.startswith("robustness_att_") and not _is_vol_robustness_key(k)
        ]
        if sent_keys:
            lines.append(r"\subsection{Alternative Sentiment Models}")
            lines.append("")
            for key in sorted(sent_keys):
                # Extract model name: robustness_{model}_volatility
                model = (
                    key.replace("robustness_", "")
                    .replace("_volatility", "")
                    .replace("_return", "")
                )
                vol_res = list(results[key])
                if vol_res:
                    lines.extend(
                        _format_standalone_table(
                            vol_res,
                            caption=f"Robustness: Baseline Volatility ({model})",
                            label=f"tab:robustness_{model}_vol",
                        )
                    )
                    lines.append("")
            lines.append(r"\newpage")
            lines.append("")

        # ---- Volatility measure robustness ----
        vol_measure_keys = [k for k in robustness_keys if _is_vol_robustness_key(k)]
        if vol_measure_keys:
            lines.append(r"\subsection{Alternative Volatility Measures}")
            lines.append("")
            for key in sorted(vol_measure_keys):
                measure = key.replace("robustness_", "").replace("_volatility", "")
                # Pretty labels for the volatility-measure keys
                _measure_label_map = {
                    "ja_rs": "Jump-Adjusted Rogers--Satchell",
                    "ja_pk": "Jump-Adjusted Parkinson",
                    "gk": "Garman--Klass",
                    "pk": "Parkinson",
                    "sqret": "Squared Returns",
                }
                measure_label = _measure_label_map.get(
                    measure, measure.replace("_", " ").title()
                )
                vol_res = results[key]
                if vol_res:
                    lines.extend(
                        _format_standalone_table(
                            vol_res,
                            caption=f"Robustness: Baseline Volatility ({measure_label})",
                            label=f"tab:robustness_{measure}_vol",
                        )
                    )
                    lines.append("")
            lines.append(r"\newpage")
            lines.append("")

        # ---- Attention specification robustness ----
        att_keys = [k for k in robustness_keys if k.startswith("robustness_att_")]
        if att_keys:
            lines.append(r"\subsection{Alternative Attention Specifications}")
            lines.append("")
            for key in sorted(att_keys):
                # robustness_att_{measure}_{std}_volatility
                spec_label = key.replace("robustness_att_", "").replace(
                    "_volatility", ""
                )
                spec_display = spec_label.replace("_", ", ")
                vol_res = results[key]
                if vol_res:
                    lines.extend(
                        _format_standalone_table(
                            vol_res,
                            caption=f"Robustness: Baseline Volatility (attention: {spec_display})",
                            label=f"tab:robustness_att_{spec_label}_vol",
                        )
                    )
                    lines.append("")
            lines.append(r"\newpage")
            lines.append("")

    # Summary statistics section
    lines.append(r"\section{Summary Statistics}")
    lines.append("")
    lines.extend(_format_summary_stats(results))
    lines.append("")

    # End document
    lines.append(r"\end{document}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath


def _format_standalone_table(
    results: List[RegressionResult],
    caption: str,
    label: str,
) -> List[str]:
    """Format results as a standalone table with threeparttable."""
    if not results:
        return []

    # Build ordered list of base variable names (strip _lagN suffixes)
    # and a per-result mapping from base_var -> actual_var
    base_regressors: List[str] = []
    seen: set = set()
    result_var_maps: List[Dict[str, str]] = []

    for r in results:
        var_map: Dict[str, str] = {}
        for reg in r.spec.regressors:
            base = _strip_lag_suffix(reg)
            var_map[base] = reg
            if base not in seen:
                base_regressors.append(base)
                seen.add(base)
        result_var_maps.append(var_map)

    # Build table
    col_spec = "l" + "c" * len(results)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row
    header_parts = [""]
    for r in results:
        header_parts.append(f"$h={r.spec.horizon}$")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # Coefficient rows (using base variable names)
    for base_var in base_regressors:
        var_display = _format_var_name(base_var)

        # Coefficient row
        coef_parts = [var_display]
        for i, r in enumerate(results):
            actual_var = result_var_maps[i].get(base_var)
            if actual_var and actual_var in r.coefficients:
                coef = r.coefficients[actual_var]
                pval = r.p_values.get(actual_var, 1.0)
                stars = _significance_stars(pval)
                coef_parts.append(f"${coef:.4f}{stars}$")
            else:
                coef_parts.append("--")
        lines.append(" & ".join(coef_parts) + r" \\")

        # Standard error row
        se_parts = [""]
        for i, r in enumerate(results):
            actual_var = result_var_maps[i].get(base_var)
            if actual_var and actual_var in r.std_errors:
                se = r.std_errors[actual_var]
                se_parts.append(f"$({se:.4f})$")
            else:
                se_parts.append("")
        lines.append(" & ".join(se_parts) + r" \\[0.5em]")

    lines.append(r"\midrule")

    # Footer with statistics
    obs_parts = ["Observations"]
    r2_parts = ["$R^2$"]
    r2w_parts = ["Within $R^2$"]
    n_ent_parts = ["Stocks"]
    n_time_parts = ["Trading Days"]

    for r in results:
        obs_parts.append(f"{r.n_obs:,}")
        r2_parts.append(f"{r.r_squared:.4f}")
        r2w_parts.append(f"{r.r_squared_within:.4f}")
        n_ent_parts.append(f"{r.n_entities:,}")
        n_time_parts.append(f"{r.n_time:,}")

    lines.append(" & ".join(obs_parts) + r" \\")
    lines.append(" & ".join(r2_parts) + r" \\")
    lines.append(" & ".join(r2w_parts) + r" \\")
    lines.append(" & ".join(n_ent_parts) + r" \\")
    lines.append(" & ".join(n_time_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Table notes
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item \textit{Notes:} Clustered standard errors in parentheses.")
    lines.append(r"\item $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$, $^{\cdot}p<0.1$")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")

    return lines


def _format_var_name(var: str) -> str:
    """Format variable name for LaTeX display."""
    # Map variable names to readable LaTeX
    name_map = {
        "attention_std": r"Attention ($\tilde{A}$)",
        "sentiment_pos_std": r"Sentiment$^+$ ($\tilde{S}^+$)",
        "sentiment_neg_std": r"Sentiment$^-$ ($\tilde{S}^-$)",
        "sentiment_std": r"Sentiment ($\tilde{S}$)",
        "sentiment_pos_x_attention": r"$\tilde{S}^+ \times \tilde{A}$",
        "sentiment_neg_x_attention": r"$\tilde{S}^- \times \tilde{A}$",
        "sentiment_x_attention": r"$\tilde{S} \times \tilde{A}$",
        "valid_sentiment": r"ValidSent",
        "lag_return": r"$r_{t-1}$",
        "lag_log_volatility": r"$\ln V_{t-1}$",
        "sentiment_disp_std": r"Dispersion ($\tilde{D}$)",
        "sentiment_intensity_std": r"Intensity ($|\tilde{S}|$)",
        "log_variance_ja_gk": r"$\ln \sigma^2_{JA\text{-}GK,t-h}$",
        "log_variance_ja_rs": r"$\ln \sigma^2_{JA\text{-}RS,t-h}$",
        "log_variance_ja_pk": r"$\ln \sigma^2_{JA\text{-}PK,t-h}$",
        "log_variance_gk": r"$\ln \sigma^2_{GK,t-h}$",
        "log_variance_pk": r"$\ln \sigma^2_{PK,t-h}$",
        "log_variance_sqret": r"$\ln \sigma^2_{SR,t-h}$",
    }

    if var in name_map:
        return name_map[var]

    # Try stripping lag suffix and looking up the base name
    base = _strip_lag_suffix(var)
    if base in name_map:
        return name_map[base]

    # Default: escape underscores
    return var.replace("_", r"\_")


def _format_summary_stats(
    results: Dict[str, List[RegressionResult]],
) -> List[str]:
    """Format summary statistics table."""
    lines = []

    # Count total regressions
    total = sum(len(v) for v in results.values())

    # Compute average statistics
    all_results = []
    for v in results.values():
        all_results.extend(v)

    if not all_results:
        lines.append("No results to summarize.")
        return lines

    avg_r2_all = sum(r.r_squared for r in all_results) / len(all_results)
    avg_r2w_all = sum(r.r_squared_within for r in all_results) / len(all_results)
    avg_n_all = sum(r.n_obs for r in all_results) / len(all_results)

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary of Regression Results}")
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(r"Statistic & Value \\")
    lines.append(r"\midrule")
    lines.append(f"Regressions & {total} \\\\")
    lines.append(f"Avg. $R^2$ & {avg_r2_all:.4f} \\\\")
    lines.append(f"Avg. Within $R^2$ & {avg_r2w_all:.4f} \\\\")
    lines.append(f"Avg. Observations & {avg_n_all:,.0f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return lines
