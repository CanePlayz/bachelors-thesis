"""Data loading utilities for regression analysis.

This module provides efficient loading of:
- Financial data (returns, volatility)
- Sentiment data (from Reddit pipeline)
- Market benchmark data

Key features:
- Memory-efficient Parquet loading with column selection
- Trading calendar derivation from financial data
- Calendar-to-trading-day mapping for weekend/holiday aggregation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Set, Tuple

import numpy as np
import pandas as pd
from config import (BENCHMARK_CSV, RETURNS_PARQUET, SENTIMENT_CSV,
                    RegressionConfig)


def load_returns_data(
    config: RegressionConfig,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load stock returns and volatility data.

    Args:
        config: Regression configuration
        columns: Specific columns to load (None = all)

    Returns:
        DataFrame with columns [date, ticker, ...financial columns...]
    """
    if not RETURNS_PARQUET.exists():
        print(f"ERROR: Returns data not found: {RETURNS_PARQUET}")
        print("Run the financial data pipeline first:")
        print("  cd financial_data && python fetch_stock_data.py")
        sys.exit(1)

    print(f"Loading returns data: {RETURNS_PARQUET.name}")

    # Default columns for regression
    if columns is None:
        columns = [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "log_return",
            "volatility_ja_gk",
            "volatility_ja_rs",
            "volatility_ja_pk",
            "volatility_gk",
            "volatility_pk",
            "volatility_sqret",
        ]

    df = pd.read_parquet(RETURNS_PARQUET, columns=columns)

    # Apply date filter if specified
    if config.sample.start_date:
        df = df[df["date"] >= config.sample.start_date]
    if config.sample.end_date:
        df = df[df["date"] <= config.sample.end_date]

    print(f"  → {len(df):,} rows, {df['ticker'].nunique():,} tickers")
    print(f"  → Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def load_benchmark_data(config: RegressionConfig) -> pd.DataFrame:
    """Load market benchmark data (S&P 500).

    Args:
        config: Regression configuration

    Returns:
        DataFrame with columns [date, market_return, market_volatility_ja_gk, ...]
    """
    if not BENCHMARK_CSV.exists():
        print(f"WARNING: Benchmark data not found: {BENCHMARK_CSV}")
        return pd.DataFrame()

    print(f"Loading benchmark data: {BENCHMARK_CSV.name}")
    df = pd.read_csv(BENCHMARK_CSV)

    # Apply date filter
    if config.sample.start_date:
        df = df[df["date"] >= config.sample.start_date]
    if config.sample.end_date:
        df = df[df["date"] <= config.sample.end_date]

    print(f"  → {len(df):,} trading days")

    return df


def load_sentiment_data(
    config: RegressionConfig,
    model: str | None = None,
) -> pd.DataFrame:
    """Load sentiment data from Reddit pipeline.

    Args:
        config: Regression configuration
        model: Sentiment model to filter by (None = all models)

    Returns:
        DataFrame with columns [date, ticker, model, sentiment_avg, sentiment_disp, mention_count, ...]
    """
    if not SENTIMENT_CSV.exists():
        print(f"ERROR: Sentiment data not found: {SENTIMENT_CSV}")
        print("Run the Reddit pipeline first to generate sentiment data.")
        sys.exit(1)

    print(f"Loading sentiment data: {SENTIMENT_CSV.name}")

    df = pd.read_csv(SENTIMENT_CSV)

    # Filter by model if specified
    if model:
        df = df[df["model"] == model]
        print(f"  → Filtered to model: {model}")

    # Apply date filter
    if config.sample.start_date:
        df = df[df["date"] >= config.sample.start_date]
    if config.sample.end_date:
        df = df[df["date"] <= config.sample.end_date]

    print(f"  → {len(df):,} rows, {df['ticker'].nunique():,} tickers")
    if "model" in df.columns:
        print(f"  → Models: {sorted(df['model'].unique())}")
    print(f"  → Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def get_trading_calendar(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Derive trading calendar from financial data.

    Trading days are defined as days when the market was open,
    derived from the dates present in the returns data.

    Args:
        returns_df: Returns DataFrame with 'date' column

    Returns:
        DataFrame with columns [date, trading_day_index]
        where trading_day_index is a sequential integer (0, 1, 2, ...)
    """
    # Get unique sorted trading dates
    trading_dates = pd.DataFrame({"date": sorted(returns_df["date"].unique())})
    trading_dates["trading_day_index"] = range(len(trading_dates))

    return trading_dates


def map_calendar_to_trading_day(
    df: pd.DataFrame,
    trading_calendar: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Map calendar dates to trading days (backward aggregation).

    Posts from non-trading days are aggregated to the *preceding* trading day.
    Weekend posts (Saturday/Sunday) are assigned to Friday; holiday posts are
    assigned to the last open trading day before the gap.  This ensures that
    information accumulated over non-trading periods enters the predictor slot
    rather than the outcome window in the lagged regression design.

    Posts published on a trading day itself are assigned to that same day.

    Args:
        df: DataFrame with calendar dates
        trading_calendar: Trading calendar from get_trading_calendar()
        date_col: Name of the date column in df

    Returns:
        DataFrame with 'trading_date' column added
    """
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    # Generate all calendar dates in range
    all_calendar_dates = pd.date_range(min_date, max_date, freq="D")

    # Convert trading calendar dates to datetime for comparison
    trading_dates = pd.to_datetime(trading_calendar["date"]).sort_values()
    trading_set = set(trading_dates)

    # Map each calendar date to the most recent trading date (backward)
    date_mapping = {}
    for cal_date in all_calendar_dates:
        if cal_date in trading_set:
            # Trading day → maps to itself
            date_mapping[cal_date.strftime("%Y-%m-%d")] = cal_date.strftime("%Y-%m-%d")
        else:
            # Non-trading day → find preceding trading day
            prev_trading = trading_dates[trading_dates <= cal_date]
            if len(prev_trading) > 0:
                date_mapping[cal_date.strftime("%Y-%m-%d")] = prev_trading.iloc[
                    -1
                ].strftime("%Y-%m-%d")
            else:
                # No prior trading day (edge case at start of data)
                date_mapping[cal_date.strftime("%Y-%m-%d")] = None

    # Apply mapping
    df = df.copy()
    df["trading_date"] = df[date_col].map(date_mapping)

    # Drop rows that couldn't be mapped (beyond data range)
    n_unmapped = df["trading_date"].isna().sum()
    if n_unmapped > 0:
        print(f"  → Dropped {n_unmapped} rows beyond trading calendar range")
        df = df.dropna(subset=["trading_date"])

    return df


def filter_stocks_by_criteria(
    returns_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    config: RegressionConfig,
) -> Tuple[Set[str], pd.DataFrame]:
    """Filter stocks based on inclusion criteria.

    Applies minimum observation requirements:
    - min_trading_days_financial: Days with price data
    - min_trading_days_mentioned: Days with any social media mention
    - min_trading_days_valid_sentiment: Days with valid sentiment (>= min_posts)

    Args:
        returns_df: Returns data
        sentiment_df: Sentiment data (already filtered to primary model)
        config: Regression configuration

    Returns:
        Tuple of (included_tickers, summary_df with filtering stats)
    """
    print("\nApplying stock inclusion filters...")

    # Count trading days per stock in financial data
    financial_counts = returns_df.groupby("ticker")["date"].nunique().reset_index()
    financial_counts.columns = ["ticker", "n_financial_days"]

    # Determine date column in sentiment data (date or trading_date after aggregation)
    date_col = "trading_date" if "trading_date" in sentiment_df.columns else "date"

    # Count days with mentions per stock
    mention_counts = sentiment_df.groupby("ticker")[date_col].nunique().reset_index()
    mention_counts.columns = ["ticker", "n_mention_days"]

    # Count days with valid sentiment (>= min_posts threshold)
    min_posts = config.sentiment.min_posts_for_valid_sentiment
    valid_sentiment = sentiment_df[sentiment_df["mention_count"] >= min_posts]
    valid_counts = valid_sentiment.groupby("ticker")[date_col].nunique().reset_index()
    valid_counts.columns = ["ticker", "n_valid_sentiment_days"]

    # Merge counts
    summary = financial_counts.merge(mention_counts, on="ticker", how="outer")
    summary = summary.merge(valid_counts, on="ticker", how="outer")
    summary = summary.fillna(0)

    # Apply filters
    min_fin = config.sample.min_trading_days_financial
    min_ment = config.sample.min_trading_days_mentioned
    min_valid = config.sample.min_trading_days_valid_sentiment

    included = summary[
        (summary["n_financial_days"] >= min_fin)
        & (summary["n_mention_days"] >= min_ment)
        & (summary["n_valid_sentiment_days"] >= min_valid)
    ]

    included_tickers = set(included["ticker"])

    # Print summary
    print(
        f"  → Stocks with >= {min_fin} financial days: {(summary['n_financial_days'] >= min_fin).sum()}"
    )
    print(
        f"  → Stocks with >= {min_ment} mention days: {(summary['n_mention_days'] >= min_ment).sum()}"
    )
    print(
        f"  → Stocks with >= {min_valid} valid sentiment days: {(summary['n_valid_sentiment_days'] >= min_valid).sum()}"
    )
    print(f"  → Stocks meeting all criteria: {len(included_tickers)}")

    return included_tickers, summary
