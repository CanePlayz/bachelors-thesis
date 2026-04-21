"""Panel data builder for regression analysis.

This module orchestrates the construction of regression-ready panel data:
1. Load financial and sentiment data
2. Align to trading calendar
3. Apply sample filters
4. Compute features (attention, sentiment, returns, volatility)
5. Save panel with metadata

The resulting panel is indexed by (ticker, date) and contains all variables
needed for the baseline and extension regressions.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import pandas as pd
from config import OUTPUT_DIR, RegressionConfig

from .features import (compute_attention_variables, compute_interaction_terms,
                       compute_lagged_controls, compute_lagged_predictors,
                       compute_rolling_volatility, compute_sentiment_variables,
                       standardize)
from .io import load_panel_data, panel_is_stale, save_panel
from .loaders import (filter_stocks_by_criteria, get_trading_calendar,
                      load_benchmark_data, load_returns_data,
                      load_sentiment_data, map_calendar_to_trading_day)


def panel_exists(output_dir: Path | None = None) -> bool:
    """Check if panel data exists.

    Args:
        output_dir: Directory to check (defaults to OUTPUT_DIR/panel)

    Returns:
        True if panel_data.parquet exists
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "panel"

    return (output_dir / "panel_data.parquet").exists()


def load_panel(
    output_dir: Path | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load existing panel data.

    Args:
        output_dir: Directory containing panel data
        columns: Specific columns to load

    Returns:
        Panel DataFrame
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "panel"

    return load_panel_data(output_dir, columns=columns)


def build_panel(
    config: RegressionConfig,
    force: bool = False,
    output_dir: Path | None = None,
) -> Tuple[pd.DataFrame, Path]:
    """Build regression-ready panel data.

    Main orchestration function that:
    1. Checks if rebuild is needed (unless force=True)
    2. Loads and merges financial + sentiment data
    3. Applies sample filters
    4. Computes all regression variables
    5. Saves panel with metadata

    Args:
        config: Regression configuration
        force: Force rebuild even if panel exists
        output_dir: Output directory (defaults to OUTPUT_DIR/panel)

    Returns:
        Tuple of (panel DataFrame, path to saved panel)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "panel"

    # Check if rebuild needed
    if not force and panel_exists(output_dir):
        if not panel_is_stale(output_dir, config):
            print("Panel data exists and is up-to-date. Use --force to rebuild.")
            return load_panel(output_dir), output_dir / "panel_data.parquet"
        else:
            print("Panel data is stale, rebuilding...")

    print("\n" + "=" * 60)
    print("BUILDING PANEL DATA")
    print("=" * 60)
    start_time = time.time()

    # -------------------------------------------------------------------------
    # Step 1: Load source data
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading source data...")

    returns_df = load_returns_data(config)
    benchmark_df = load_benchmark_data(config)
    sentiment_df = load_sentiment_data(config, model=config.primary_sentiment_model)

    # -------------------------------------------------------------------------
    # Step 2: Create trading calendar and map sentiment dates
    # -------------------------------------------------------------------------
    print("\n[2/6] Aligning to trading calendar...")

    trading_calendar = get_trading_calendar(returns_df)
    print(f"  → Trading calendar: {len(trading_calendar):,} days")

    # Map sentiment data (which may include weekends) to trading days
    sentiment_df = map_calendar_to_trading_day(
        sentiment_df,
        trading_calendar,
        date_col="date",
    )

    # Aggregate sentiment to trading day level (sum mentions, weighted avg sentiment)
    sentiment_df = _aggregate_sentiment_to_trading_day(sentiment_df)
    print(f"  → Sentiment aggregated to trading days: {len(sentiment_df):,} rows")

    # -------------------------------------------------------------------------
    # Step 3: Apply sample filters
    # -------------------------------------------------------------------------
    print("\n[3/6] Applying sample filters...")

    included_tickers, filter_summary = filter_stocks_by_criteria(
        returns_df, sentiment_df, config
    )

    # Filter data to included tickers
    returns_df = returns_df[returns_df["ticker"].isin(included_tickers)]
    sentiment_df = sentiment_df[sentiment_df["ticker"].isin(included_tickers)]

    print(
        f"  → After filtering: {len(returns_df):,} return rows, {len(sentiment_df):,} sentiment rows"
    )

    # -------------------------------------------------------------------------
    # Step 4: Merge financial and sentiment data
    # -------------------------------------------------------------------------
    print("\n[4/6] Merging data sources...")

    # Merge on (date, ticker)
    panel = returns_df.merge(
        sentiment_df,
        left_on=["date", "ticker"],
        right_on=["trading_date", "ticker"],
        how="left",
    )

    # Fill missing sentiment data (days with no mentions)
    panel["mention_count"] = panel["mention_count"].fillna(0).astype(int)
    panel["sentiment_avg"] = panel["sentiment_avg"].fillna(0.0)
    if "sentiment_disp" in panel.columns:
        panel["sentiment_disp"] = panel["sentiment_disp"].fillna(0.0)

    # Compute gap_days from the trading calendar (property of the date, not ticker).
    # Under backward aggregation, non-trading-day posts fold into the *preceding*
    # trading day.  gap_days therefore counts how many non-trading days follow
    # this trading day before the next one opens:
    #   gap_days = (calendar days until next trading day) - 1
    # Friday → next Mon → 3-1 = 2 (absorbs Sat+Sun), normal weekday → 0.
    _tc_dates = (
        pd.to_datetime(trading_calendar["date"]).sort_values().reset_index(drop=True)
    )
    _gap = (_tc_dates.shift(-1) - _tc_dates).dt.days.fillna(1).astype(int) - 1
    _gap_df = pd.DataFrame(
        {"date": _tc_dates.dt.strftime("%Y-%m-%d"), "gap_days": _gap}
    )
    # Remove any gap_days column that came from sentiment merge (partial)
    if "gap_days" in panel.columns:
        panel = panel.drop(columns=["gap_days"])
    panel = panel.merge(_gap_df, on="date", how="left")
    panel["gap_days"] = panel["gap_days"].fillna(0).astype(int)

    # Add benchmark data
    if not benchmark_df.empty:
        benchmark_cols = ["date", "market_return", "market_log_return"]
        if "market_volatility_ja_gk" in benchmark_df.columns:
            benchmark_cols.append("market_volatility_ja_gk")
        panel = panel.merge(
            benchmark_df[benchmark_cols],
            on="date",
            how="left",
        )

    print(f"  → Merged panel: {len(panel):,} rows")

    # -------------------------------------------------------------------------
    # Step 5: Compute regression variables
    # -------------------------------------------------------------------------
    print("\n[5/6] Computing regression variables...")

    # Attention variables
    print("  → Computing attention variables...")
    panel = compute_attention_variables(panel, config)

    # Sentiment variables
    print("  → Computing sentiment variables...")
    panel = compute_sentiment_variables(panel, config)

    # Daily volatility measures are computed upstream in financial_data/returns.py
    # (baseline JA-GK plus five robustness alternatives) and merged in via the
    # returns parquet, so no per-day variance computation is needed here.

    # Rolling volatility target (backward-looking RMS) — baseline measure
    print("  → Computing rolling volatility (backward-looking, baseline: ja_gk)...")
    panel = compute_rolling_volatility(
        panel,
        config.horizons,
        aggregation=config.volatility.aggregation,
        measure="ja_gk",
    )

    # Rolling volatility for robustness measures
    for rob_measure in config.volatility.robustness_measures:
        print(f"  → Computing rolling volatility (robustness: {rob_measure})...")
        panel = compute_rolling_volatility(
            panel,
            config.horizons,
            aggregation=config.volatility.aggregation,
            measure=rob_measure,
        )

    # Lagged controls (base columns before horizon-specific shifting)
    print("  → Computing lagged controls...")
    panel = compute_lagged_controls(panel)

    # Interaction terms (computed on unlagged columns first)
    print("  → Computing interaction terms...")
    panel = compute_interaction_terms(panel, config)

    # Robustness columns for GARCH (alt sentiment models + alt attention specs)
    print("  → Computing robustness columns for GARCH...")
    panel = _compute_robustness_columns(panel, config, trading_calendar)

    # Lagged predictors (shift all regressors by h for backward-looking model)
    print("  → Computing lagged predictors (backward-looking shift)...")
    panel = compute_lagged_predictors(panel, config.horizons)

    # -------------------------------------------------------------------------
    # Step 6: Clean and save
    # -------------------------------------------------------------------------
    print("\n[6/6] Cleaning and saving panel...")

    # Sort by ticker, date
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Drop rows with missing target variables (for any horizon)
    target_cols = []
    for h in config.horizons:
        target_cols.append(f"log_volatility_{h}d")

    # Count before dropping
    n_before = len(panel)
    panel = panel.dropna(subset=target_cols, how="all")
    n_after = len(panel)
    print(f"  → Dropped {n_before - n_after:,} rows with missing targets")

    # Select and order columns
    panel = _select_output_columns(panel, config)

    # Save
    panel_path = save_panel(panel, config, output_dir)

    # Summary
    elapsed = time.time() - start_time
    print(f"\nPanel build complete in {elapsed:.1f}s")

    return panel, panel_path


def _aggregate_sentiment_to_trading_day(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment data to trading day level.

    When multiple calendar days map to the same trading day (backward
    aggregation: non-trading days → preceding trading day), aggregate by
    summing mentions and taking engagement-weighted average of sentiment.
    Uses weight_sum (sum of engagement weights from the Reddit pipeline) as the
    weighting variable, consistent with how the pipeline originally computes
    sentiment_avg within each calendar day.
    """
    # Group by (trading_date, ticker)
    agg_dict = {
        "mention_count": "sum",  # Sum mentions across days
    }

    # Weighted average for sentiment (weight by engagement weight_sum)
    if "sentiment_avg" in df.columns:
        df["_weighted_sentiment"] = df["sentiment_avg"] * df["weight_sum"]
        agg_dict["_weighted_sentiment"] = "sum"
        agg_dict["weight_sum"] = "sum"

    if "sentiment_disp" in df.columns:
        # Take max dispersion (conservative approach)
        agg_dict["sentiment_disp"] = "max"

    # Aggregate
    result = df.groupby(["trading_date", "ticker"]).agg(agg_dict).reset_index()

    # Compute weighted average sentiment
    if "_weighted_sentiment" in result.columns:
        result["sentiment_avg"] = result["_weighted_sentiment"] / result[
            "weight_sum"
        ].replace(0, 1)
        result = result.drop(columns=["_weighted_sentiment"])

    return result


def _compute_robustness_columns(
    panel: pd.DataFrame,
    config: RegressionConfig,
    trading_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Compute robustness columns for GARCH estimation.

    Adds suffixed columns for alternative sentiment models and attention specs
    so that the GARCH pipeline can run robustness checks directly from the panel
    without re-deriving columns at runtime.

    Adds:
      - sentiment_pos_std_{model}, sentiment_neg_std_{model}  for each alt model
      - attention_std_{measure}_{standardization}              for each alt spec

    These are UNLAGGED base columns — GARCH handles its own lag structure.
    """
    import numpy as np
    from config import ATTENTION_ROBUSTNESS_SPECS, ROBUSTNESS_SENTIMENT_MODELS

    # ----- Alternative sentiment models -------------------------------------
    for model in ROBUSTNESS_SENTIMENT_MODELS:
        print(f"    Alt sentiment model: {model}")

        sentiment_df = load_sentiment_data(config, model=model)
        sentiment_df = map_calendar_to_trading_day(
            sentiment_df,
            trading_calendar,
            date_col="date",
        )
        sentiment_df = _aggregate_sentiment_to_trading_day(sentiment_df)
        sentiment_df = sentiment_df.rename(columns={"trading_date": "date"})

        # Merge alt sentiment onto panel
        merge_cols = ["date", "ticker", "sentiment_avg", "mention_count"]
        if "weight_sum" in sentiment_df.columns:
            merge_cols.append("weight_sum")
        panel = panel.merge(
            sentiment_df[merge_cols],
            on=["date", "ticker"],
            how="left",
            suffixes=("", f"_{model}"),
        )

        mc_col = f"mention_count_{model}"
        sa_col = f"sentiment_avg_{model}"

        # If no suffix was added (no conflict), rename explicitly
        if mc_col not in panel.columns:
            # Conflict: mention_count already exists, so pandas added suffix
            # This shouldn't happen — mention_count exists in base panel,
            # so the suffix WILL be added.  Just guard against edge cases.
            pass

        panel[mc_col] = panel[mc_col].fillna(0).astype(int)
        panel[sa_col] = panel[sa_col].fillna(0.0)

        # Valid sentiment for this model
        min_posts = config.sentiment.min_posts_for_valid_sentiment
        valid = (panel[mc_col] >= min_posts).astype(int)
        sent = np.where(valid == 1, panel[sa_col], 0.0)

        # Positive / negative split
        sent_pos = np.maximum(sent, 0.0)
        sent_neg = np.minimum(sent, 0.0)

        # Standardize (match primary model's standardization scope)
        std_scope = config.sentiment.standardization
        if std_scope == "within_stock":
            panel[f"sentiment_pos_std_{model}"] = standardize(
                pd.Series(sent_pos, index=panel.index),
                scope="within_stock",
                group_col=panel["ticker"],
            )
            panel[f"sentiment_neg_std_{model}"] = standardize(
                pd.Series(sent_neg, index=panel.index),
                scope="within_stock",
                group_col=panel["ticker"],
            )
        else:
            panel[f"sentiment_pos_std_{model}"] = standardize(
                pd.Series(sent_pos, index=panel.index),
                scope="global",
            )
            panel[f"sentiment_neg_std_{model}"] = standardize(
                pd.Series(sent_neg, index=panel.index),
                scope="global",
            )

        # Drop intermediate columns (keep only the final _std variants)
        panel = panel.drop(
            columns=[mc_col, sa_col]
            + (
                [f"weight_sum_{model}"]
                if f"weight_sum_{model}" in panel.columns
                else []
            ),
            errors="ignore",
        )

    # ----- Alternative attention specifications -----------------------------
    for att_config in ATTENTION_ROBUSTNESS_SPECS:
        label = f"{att_config.measure[:3]}_{att_config.standardization}"
        # e.g. abs_global, rel_within, rel_global
        print(f"    Alt attention spec: {label}")

        if att_config.measure == "absolute":
            raw_attention = pd.Series(
                np.log1p(panel["mention_count"]),
                index=panel.index,
            )
        elif att_config.measure == "relative":
            daily_total = panel.groupby("date")["mention_count"].transform("sum")
            daily_total = daily_total.replace(0, 1)
            raw_attention = pd.Series(
                panel["mention_count"] / daily_total,
                index=panel.index,
            )
        else:
            raise ValueError(f"Unknown attention measure: {att_config.measure}")

        if att_config.standardization == "within_stock":
            panel[f"attention_std_{label}"] = standardize(
                raw_attention,
                scope="within_stock",
                group_col=panel["ticker"],
            )
        else:
            panel[f"attention_std_{label}"] = standardize(
                raw_attention,
                scope="global",
            )

    return panel


def _select_output_columns(df: pd.DataFrame, config: RegressionConfig) -> pd.DataFrame:
    """Select and order output columns for the panel."""
    # Core identification
    columns = ["date", "ticker"]

    # Financial data: log return + the full daily-volatility battery
    columns.extend(
        [
            "adj_close",
            "log_return",
            "volatility_ja_gk",
            "volatility_ja_rs",
            "volatility_ja_pk",
            "volatility_gk",
            "volatility_pk",
            "volatility_sqret",
        ]
    )

    # Attention
    columns.extend(["mention_count", "attention", "attention_std"])

    # Gap days (non-trading days aggregated into this row; GARCH control)
    columns.append("gap_days")

    # Sentiment - always include BOTH parameterizations
    # Baseline uses S⁺/S⁻, Extension uses aggregate S
    columns.extend(["sentiment_avg", "sentiment", "valid_sentiment"])

    # Aggregate sentiment (for extension)
    columns.append("sentiment_std")

    # Positive/negative split (for baseline)
    columns.extend(
        [
            "sentiment_pos",
            "sentiment_neg",
            "sentiment_pos_std",
            "sentiment_neg_std",
        ]
    )

    if config.sentiment.include_dispersion:
        columns.extend(["sentiment_disp", "sentiment_disp_std"])

    if config.sentiment.include_intensity:
        columns.extend(["sentiment_intensity", "sentiment_intensity_std"])

    # Interaction terms - always include BOTH types
    columns.append("sentiment_x_attention")  # for extension
    columns.extend(
        ["sentiment_pos_x_attention", "sentiment_neg_x_attention"]
    )  # for baseline

    # Volatility targets (backward-looking)
    for h in config.horizons:
        columns.extend([f"rolling_volatility_{h}d", f"log_volatility_{h}d"])
        # Robustness volatility targets
        for rob in config.volatility.robustness_measures:
            suffix = f"_{rob}"
            columns.extend(
                [
                    f"rolling_volatility_{h}d{suffix}",
                    f"log_volatility_{h}d{suffix}",
                ]
            )

    # Forward return targets
    # (none — the backward-looking volatility model does not use forward returns)

    # Base predictor columns (unlagged): one log_variance per measure plus the
    # lagged-return auxiliary control. We list them explicitly so that the
    # output schema is stable regardless of column-presence checks elsewhere.
    columns.extend(
        [
            "log_variance_ja_gk",
            "log_variance_ja_rs",
            "log_variance_ja_pk",
            "log_variance_gk",
            "log_variance_pk",
            "log_variance_sqret",
            "lag_return",
        ]
    )

    # Lagged predictor columns (horizon-specific)
    for h in config.horizons:
        lag_cols = [
            f"attention_std_lag{h}",
            f"sentiment_pos_std_lag{h}",
            f"sentiment_neg_std_lag{h}",
            f"sentiment_std_lag{h}",
            f"valid_sentiment_lag{h}",
            f"sentiment_pos_x_attention_lag{h}",
            f"sentiment_neg_x_attention_lag{h}",
            f"sentiment_x_attention_lag{h}",
            f"log_variance_ja_gk_lag{h}",
            f"log_variance_ja_rs_lag{h}",
            f"log_variance_ja_pk_lag{h}",
            f"log_variance_gk_lag{h}",
            f"log_variance_pk_lag{h}",
            f"log_variance_sqret_lag{h}",
            f"lag_return_lag{h}",
            f"sentiment_disp_std_lag{h}",
            f"sentiment_intensity_std_lag{h}",
        ]
        columns.extend(lag_cols)

    # Market benchmark
    if "market_return" in df.columns:
        columns.extend(["market_return", "market_log_return"])
    if "market_volatility_ja_gk" in df.columns:
        columns.append("market_volatility_ja_gk")

    # GARCH robustness columns (unlagged — GARCH handles its own lag structure)
    # Alt sentiment models
    from config import ATTENTION_ROBUSTNESS_SPECS, ROBUSTNESS_SENTIMENT_MODELS

    for model in ROBUSTNESS_SENTIMENT_MODELS:
        columns.extend(
            [
                f"sentiment_pos_std_{model}",
                f"sentiment_neg_std_{model}",
            ]
        )
    # Alt attention specifications
    for att_config in ATTENTION_ROBUSTNESS_SPECS:
        label = f"{att_config.measure[:3]}_{att_config.standardization}"
        columns.append(f"attention_std_{label}")

    # Select only columns that exist
    existing_columns = [c for c in columns if c in df.columns]

    return df[existing_columns]
