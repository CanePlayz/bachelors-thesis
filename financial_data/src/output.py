"""Output saving utilities for two-stage financial data pipeline.

Stage 1: Fetch raw OHLCV from yfinance → stock_prices.parquet, market_benchmark.csv
Stage 2: Calculate returns/daily volatility measures → stock_returns.parquet

The two-stage design separates:
- Data fetching (slow, depends on API) in Stage 1
- Calculations (fast, can be re-run cheaply) in Stage 2
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from config import MARKET_BENCHMARK


def save_stage1_outputs(
    prices_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    benchmark_returns_df: pd.DataFrame,
    errors: Dict[str, str],
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_dir: Path,
    tickers_source: str = "unknown",
) -> None:
    """Save Stage 1 outputs: raw prices and benchmark.

    Output files:
    - stock_prices.parquet: Raw OHLCV data for all tickers
    - market_benchmark.csv: S&P 500 data with returns and volatility
    - stage1_log.json: Metadata about the fetch operation
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save prices (Parquet+ZSTD)
    if not prices_df.empty:
        path = output_dir / "stock_prices.parquet"
        prices_df.to_parquet(path, index=False, compression="zstd")
        print(f"\n  Saved: {path}")
        print(f"    → {len(prices_df):,} rows, {prices_df['ticker'].nunique()} tickers")

    # Save benchmark (CSV - small, human-readable)
    if not benchmark_df.empty:
        path = output_dir / "market_benchmark.csv"
        if not benchmark_returns_df.empty:
            merge_cols = ["date", "market_return", "market_log_return"]
            if "market_volatility_ja_gk" in benchmark_returns_df.columns:
                merge_cols.append("market_volatility_ja_gk")
            bench_out = benchmark_df.merge(
                benchmark_returns_df[merge_cols], on="date", how="left"
            )
        else:
            bench_out = benchmark_df
        bench_out.to_csv(path, index=False)
        print(f"  Saved: {path}")
        print(f"    → {len(bench_out):,} days")

    # Save Stage 1 log
    tickers_fetched = prices_df["ticker"].nunique() if not prices_df.empty else 0
    tickers_failed = len(errors)
    attempted = len(tickers)
    ok_rate = (tickers_fetched / attempted) if attempted else 0.0

    no_data = sum(
        1
        for msg in errors.values()
        if isinstance(msg, str) and msg.strip().lower().startswith("no data")
    )
    other_error = tickers_failed - no_data

    log = {
        "stage": 1,
        "fetch_timestamp": datetime.now().isoformat(),
        "tickers_source": tickers_source,
        "date_range": {"start": start_date, "end": end_date},
        "tickers_requested": attempted,
        "tickers_fetched": tickers_fetched,
        "tickers_failed": tickers_failed,
        "ok_rate": round(ok_rate, 6),
        "failures": {"no_data": no_data, "other_error": other_error},
        "total_price_rows": len(prices_df),
        "benchmark_ticker": MARKET_BENCHMARK,
        "benchmark_days": len(benchmark_df),
        "errors": errors if errors else {},
    }
    log_path = output_dir / "stage1_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: {log_path}")

    if errors:
        print(f"\n  Warning: {len(errors)} tickers failed (details in {log_path.name})")


def save_stage2_outputs(
    returns_df: pd.DataFrame,
    output_dir: Path,
    prices_rows: int,
    prices_tickers: int,
) -> None:
    """Save Stage 2 outputs: calculated returns and daily volatility measures.

    Output files:
    - stock_returns.parquet: Returns and volatility for all tickers
    - stage2_log.json: Metadata about the calculation
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save returns (Parquet+ZSTD)
    if not returns_df.empty:
        path = output_dir / "stock_returns.parquet"
        returns_df.to_parquet(path, index=False, compression="zstd")
        print(f"\n  Saved: {path}")
        print(f"    → {len(returns_df):,} rows")

        # Print column info for verification
        cols = returns_df.columns.tolist()
        print(f"    → Columns: {', '.join(cols)}")

    # Save Stage 2 log
    log = {
        "stage": 2,
        "process_timestamp": datetime.now().isoformat(),
        "volatility_method": "overnight_plus_garman_klass",
        "input_price_rows": prices_rows,
        "input_tickers": prices_tickers,
        "output_return_rows": len(returns_df),
        "output_tickers": returns_df["ticker"].nunique() if not returns_df.empty else 0,
        "output_columns": returns_df.columns.tolist() if not returns_df.empty else [],
    }
    log_path = output_dir / "stage2_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: {log_path}")
