"""
Compute descriptive statistics for the regression panel.

Produces:
1. Summary statistics table (Mean, Median, SD, Min, Max, Skewness, Kurtosis)
   for the key variables used in regressions.
2. Correlation matrix between key variables.
3. Panel dimensions info (N stocks, T days, total obs).

Reads the merged panel from `regression/out/panel/panel_data.parquet`
(produced by `regression/src/build_panel.py`).

Outputs LaTeX table fragments under
  analysis/out/descriptive_stats/
    descriptive_stats.tex
    correlation_matrix.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure UTF-8 stdout/stderr on Windows so unicode characters in log output
# don't crash under the default cp1252 codec.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np
import pandas as pd
import polars as pl  # type: ignore[import-not-found]
from scipy import stats as spstats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
ANALYSIS_DIR = SCRIPT_DIR.parent
REPO_ROOT = ANALYSIS_DIR.parent
OUTPUT_DIR = ANALYSIS_DIR / "out"
PANEL_PATH = REPO_ROOT / "regression" / "out" / "panel" / "panel_data.parquet"

# ---------------------------------------------------------------------------
# Columns we care about
# ---------------------------------------------------------------------------
NEEDED_COLS = [
    "date",
    "ticker",
    # Financial
    "log_return",
    "volatility_ja_gk",
    "log_volatility_1d",
    "volume",
    "market_volatility_ja_gk",
    # Attention
    "mention_count",
    "attention",
    "attention_std",
    # Sentiment
    "sentiment_avg",
    "sentiment",
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_std",
    "sentiment_pos_std",
    "sentiment_neg_std",
    "sentiment_disp",
    "sentiment_disp_std",
    "valid_sentiment",
    # Correlation extras
    "sentiment_intensity_std",
]

# ---------------------------------------------------------------------------
# Load panel (only needed columns, via polars → numpy → pandas)
# ---------------------------------------------------------------------------
print(f"Loading panel from {PANEL_PATH} ...")
pldf = pl.read_parquet(PANEL_PATH)
available_cols = set(pldf.columns)
cols_to_load = [c for c in dict.fromkeys(NEEDED_COLS) if c in available_cols]
missing_cols = [c for c in dict.fromkeys(NEEDED_COLS) if c not in available_cols]
if missing_cols:
    print(f"Note: columns not in panel: {missing_cols}")

# Convert subset to pandas efficiently via numpy
data = {}
for col in cols_to_load:
    series = pldf[col]
    if series.dtype in (pl.Date, pl.Datetime):
        data[col] = series.to_list()
    elif series.dtype == pl.Utf8:
        data[col] = series.to_list()
    else:
        data[col] = series.to_numpy()
df = pd.DataFrame(data)
del pldf
print(f"Panel shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")

# ---------------------------------------------------------------------------
# Panel dimensions
# ---------------------------------------------------------------------------
n_stocks = df["ticker"].nunique()
n_days = df["date"].nunique()
date_min = df["date"].min()
date_max = df["date"].max()
print(f"N stocks: {n_stocks}")
print(f"T trading days: {n_days}")
print(f"Date range: {date_min} – {date_max}")
print(f"Total observations: {df.shape[0]:,}\n")

# ---------------------------------------------------------------------------
# Select key variables for descriptive stats
# ---------------------------------------------------------------------------
# We report statistics for the main variables that appear in regressions.
# Use raw / interpretable versions where possible, standardized where needed.

var_map = {}

# --- Financial variables ---
if "log_return" in df.columns:
    var_map["Log Return"] = "log_return"
if "volatility_ja_gk" in df.columns:
    var_map["Volatility"] = "volatility_ja_gk"
if "log_volatility_1d" in df.columns:
    var_map[r"Log Volatility (1d)"] = "log_volatility_1d"
if "volume" in df.columns:
    var_map["Trading Volume"] = "volume"

# --- Attention variables ---
if "mention_count" in df.columns:
    var_map["Mention Count"] = "mention_count"
if "attention" in df.columns:
    var_map[r"Attention $\ln(1+n)$"] = "attention"
if "attention_std" in df.columns:
    var_map["Attention (std)"] = "attention_std"

# --- Sentiment variables ---
if "sentiment_avg" in df.columns:
    var_map["Sentiment (raw avg)"] = "sentiment_avg"
if "sentiment" in df.columns:
    var_map["Sentiment (adj)"] = "sentiment"
if "sentiment_pos" in df.columns:
    var_map[r"Sentiment$^+$"] = "sentiment_pos"
if "sentiment_neg" in df.columns:
    var_map[r"Sentiment$^-$"] = "sentiment_neg"
if "sentiment_std" in df.columns:
    var_map["Sentiment (std)"] = "sentiment_std"
if "sentiment_pos_std" in df.columns:
    var_map[r"Sentiment$^+$ (std)"] = "sentiment_pos_std"
if "sentiment_neg_std" in df.columns:
    var_map[r"Sentiment$^-$ (std)"] = "sentiment_neg_std"
if "sentiment_disp" in df.columns:
    var_map["Sent. Dispersion"] = "sentiment_disp"
if "valid_sentiment" in df.columns:
    var_map["Valid Sentiment"] = "valid_sentiment"

# --- Market ---
if "market_volatility_ja_gk" in df.columns:
    var_map["Market Vol."] = "market_volatility_ja_gk"

print("Variables selected for descriptive statistics:")
for label, col in var_map.items():
    print(f"  {label:30s} -> {col}")
print()

# ---------------------------------------------------------------------------
# Compute summary statistics
# ---------------------------------------------------------------------------
records = []
for label, col in var_map.items():
    s = df[col].dropna()
    records.append(
        {
            "Variable": label,
            "Obs": len(s),
            "Mean": s.mean(),
            "Median": s.median(),
            "SD": s.std(),
            "Min": s.min(),
            "Max": s.max(),
            "Skew": spstats.skew(s, nan_policy="omit"),
            "Kurt": spstats.kurtosis(s, nan_policy="omit"),  # excess kurtosis
        }
    )

stats_df = pd.DataFrame(records)
print("=== Summary Statistics ===")
print(stats_df.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# Correlation matrix (key regression variables)
# ---------------------------------------------------------------------------
corr_vars_map = {}
# Pick the standardized versions used in regressions
for label, col in [
    ("Log Vol.", "log_volatility_1d"),
    ("Return", "log_return"),
    ("Attention", "attention_std"),
    (r"Sent.$^+$", "sentiment_pos_std"),
    (r"Sent.$^-$", "sentiment_neg_std"),
    ("Sent. Disp.", "sentiment_disp_std"),
    ("Market Vol.", "market_volatility_ja_gk"),
]:
    if col in df.columns:
        corr_vars_map[label] = col

corr_df = df[list(corr_vars_map.values())].corr()
corr_df.index = list(corr_vars_map.keys())
corr_df.columns = list(corr_vars_map.keys())

print("=== Correlation Matrix ===")
print(corr_df.round(3).to_string())
print()

# ---------------------------------------------------------------------------
# Format numbers for LaTeX
# ---------------------------------------------------------------------------


def fmt(x, decimals=4):
    """Format a number for LaTeX tables."""
    if pd.isna(x):
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if abs(x) >= 1000:
        return f"{x:,.{decimals}f}"
    return f"{x:.{decimals}f}"


# ---------------------------------------------------------------------------
# Build LaTeX summary statistics table
# ---------------------------------------------------------------------------
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"  \centering")
lines.append(r"  \caption{Descriptive Statistics of Key Panel Variables}")
lines.append(r"  \label{tab:desc_stats}")
lines.append(r"  \small")
lines.append(r"  \begin{tabular}{l r r r r r r r r}")
lines.append(r"    \toprule")
lines.append(r"    Variable & Obs & Mean & Median & SD & Min & Max & Skew. & Kurt. \\")
lines.append(r"    \midrule")

# Group: Financial
lines.append(r"    \multicolumn{9}{l}{\textit{Financial Variables}} \\[2pt]")
financial_labels = [
    "Log Return",
    "Volatility",
    r"Log Volatility (1d)",
    "Trading Volume",
    "Market Vol.",
]
for label in financial_labels:
    if label not in var_map:
        continue
    row = stats_df[stats_df["Variable"] == label].iloc[0]
    lines.append(
        f"    {label} & {fmt(row['Obs'], 0)} & {fmt(row['Mean'])} & "
        f"{fmt(row['Median'])} & {fmt(row['SD'])} & "
        f"{fmt(row['Min'])} & {fmt(row['Max'])} & "
        f"{fmt(row['Skew'], 2)} & {fmt(row['Kurt'], 2)} \\\\"
    )

lines.append(r"    \addlinespace")

# Group: Social Media – Attention
lines.append(r"    \multicolumn{9}{l}{\textit{Social Media -- Attention}} \\[2pt]")
attn_labels = [
    "Mention Count",
    r"Attention $\ln(1+n)$",
    "Attention (std)",
]
for label in attn_labels:
    if label not in var_map:
        continue
    row = stats_df[stats_df["Variable"] == label].iloc[0]
    lines.append(
        f"    {label} & {fmt(row['Obs'], 0)} & {fmt(row['Mean'])} & "
        f"{fmt(row['Median'])} & {fmt(row['SD'])} & "
        f"{fmt(row['Min'])} & {fmt(row['Max'])} & "
        f"{fmt(row['Skew'], 2)} & {fmt(row['Kurt'], 2)} \\\\"
    )

lines.append(r"    \addlinespace")

# Group: Social Media – Sentiment
lines.append(r"    \multicolumn{9}{l}{\textit{Social Media -- Sentiment}} \\[2pt]")
sent_labels = [
    "Sentiment (raw avg)",
    "Sentiment (adj)",
    r"Sentiment$^+$",
    r"Sentiment$^-$",
    "Sentiment (std)",
    r"Sentiment$^+$ (std)",
    r"Sentiment$^-$ (std)",
    "Sent. Dispersion",
    "Valid Sentiment",
]
for label in sent_labels:
    if label not in var_map:
        continue
    row = stats_df[stats_df["Variable"] == label].iloc[0]
    lines.append(
        f"    {label} & {fmt(row['Obs'], 0)} & {fmt(row['Mean'])} & "
        f"{fmt(row['Median'])} & {fmt(row['SD'])} & "
        f"{fmt(row['Min'])} & {fmt(row['Max'])} & "
        f"{fmt(row['Skew'], 2)} & {fmt(row['Kurt'], 2)} \\\\"
    )

lines.append(r"    \bottomrule")
lines.append(r"  \end{tabular}")
lines.append(r"  \begin{minipage}{0.95\textwidth}")
lines.append(r"    \vspace{4pt}")
lines.append(r"    \footnotesize")
lines.append(
    r"    \textit{Notes:} This table reports summary statistics for the main "
    r"variables in the stock--day panel. "
    r"Volatility is the daily \emph{jump-adjusted Garman--Klass} (JA-GK) estimator: "
    r"$\sigma^2_{JA\text{-}GK,t} = \sigma^2_{\text{overnight},t} + \sigma^2_{GK,t}$, "
    r"following the overnight-plus-intraday decomposition of Yang and Zhang (2000) "
    r"with the Garman--Klass (1980) intraday range estimator endorsed by Moln\'ar (2012). "
    r"Log Volatility (1d) is $\ln(\bar{V}^{(1)}_{i,t})$. "
    r"Attention is $\ln(1 + n_{i,t})$ where $n$ is the daily Reddit mention count; "
    r"Attention (std) is the within-stock z-score. "
    r"Sentiment is the daily average score from the RoBERTa-topic model; "
    r"Sentiment$^+$ = $\max(S,0)$, Sentiment$^-$ = $\max(-S,0)$. "
    r"``(std)'' denotes global z-scores. "
    r"Valid Sentiment equals 1 if $n_{i,t} \geq 5$. "
    r"Skew.\ and Kurt.\ report Fisher skewness and excess kurtosis. "
    f"The panel covers {n_stocks:,} stocks over {n_days:,} trading days "
    f"({date_min} to {date_max})."
)
lines.append(r"  \end{minipage}")
lines.append(r"\end{table}")

summary_tex = "\n".join(lines)

# ---------------------------------------------------------------------------
# Build LaTeX correlation table
# ---------------------------------------------------------------------------
corr_lines = []
corr_lines.append(r"\begin{table}[htbp]")
corr_lines.append(r"  \centering")
corr_lines.append(r"  \caption{Pairwise Correlations of Key Variables}")
corr_lines.append(r"  \label{tab:corr_matrix}")
corr_lines.append(r"  \small")

n_corr = len(corr_vars_map)
col_spec = "l" + " r" * n_corr
corr_lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
corr_lines.append(r"    \toprule")

# Header
header = "    " + " & ".join([""] + list(corr_vars_map.keys())) + r" \\"
corr_lines.append(header)
corr_lines.append(r"    \midrule")

# Rows (lower triangle only)
labels = list(corr_vars_map.keys())
for i, row_label in enumerate(labels):
    cells = [row_label]
    for j in range(n_corr):
        if j <= i:
            val = corr_df.iloc[i, j]
            cells.append(f"{val:.3f}" if i != j else "1.000")
        else:
            cells.append("")
    corr_lines.append("    " + " & ".join(cells) + r" \\")

corr_lines.append(r"    \bottomrule")
corr_lines.append(r"  \end{tabular}")
corr_lines.append(r"\end{table}")

corr_tex = "\n".join(corr_lines)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
DESC_OUTPUT_DIR = OUTPUT_DIR / "descriptive_stats"
DESC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_path_summary = DESC_OUTPUT_DIR / "descriptive_stats.tex"
out_path_corr = DESC_OUTPUT_DIR / "correlation_matrix.tex"

out_path_summary.write_text(summary_tex, encoding="utf-8")
print(f"\nSummary stats table written to: {out_path_summary}")

out_path_corr.write_text(corr_tex, encoding="utf-8")
print(f"Correlation matrix written to: {out_path_corr}")

# Also print both tables
print("\n" + "=" * 80)
print("SUMMARY STATISTICS TABLE (LaTeX)")
print("=" * 80)
print(summary_tex)
print("\n" + "=" * 80)
print("CORRELATION MATRIX TABLE (LaTeX)")
print("=" * 80)
print(corr_tex)
