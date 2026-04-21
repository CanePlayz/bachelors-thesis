"""Fetch and process stock market data for empirical sentiment analysis.

Two-stage pipeline:
    Stage 1: Fetch raw OHLCV from Yahoo Finance
             Outputs: stock_prices.parquet, market_benchmark.csv, stage1_log.json
    Stage 2: Calculate returns and the daily volatility battery
             Outputs: stock_returns.parquet, stage2_log.json

The two-stage design separates data fetching (slow, API-dependent) from
calculations (fast, can be re-run cheaply to experiment with different methods).

Volatility measures:
    Stage 2 computes six daily volatility series. The baseline is the
    jump-adjusted Garman--Klass (JA-GK) estimator: the squared overnight log
    return plus the Garman--Klass (1980) intraday range estimator, which
    Molnár (2012) identifies as the most efficient range-based intraday
    estimator. Five additional series (jump-adjusted Rogers--Satchell,
    jump-adjusted Parkinson, pure Garman--Klass, pure Parkinson, squared
    returns) are computed for downstream robustness checks.

Ticker universe and date range:
    Both are auto-discovered from the Reddit pipeline stats output. The
    ticker list comes from `ticker_stats.csv` (every ticker mentioned at
    least once); the date range comes from `daily_totals.csv`. The Reddit
    stats pipeline must therefore have been run first. Use --tickers to
    override the ticker list manually (mainly for debugging).

Re-run Logic:
    - No flags: auto-discover which stages need running
    - --force: re-run all stages
    - --stage N: start discovery from stage N
    - --stage N --force: force re-run from stage N onwards

Usage:
    python fetch_stock_data.py                    # Auto-discover and run needed stages
    python fetch_stock_data.py --stage 2          # Re-calculate returns only (uses existing prices)
    python fetch_stock_data.py --force            # Re-fetch everything from yfinance
    python fetch_stock_data.py --stage 2 --force  # Force re-run Stage 2 only
    python fetch_stock_data.py --tickers AAPL MSFT  # Debugging: fetch a specific subset

Output:
    out/stock_prices.parquet   - Daily OHLCV + adjusted close (Stage 1)
    out/stock_returns.parquet  - Returns + daily volatility battery (Stage 2)
    out/market_benchmark.csv   - S&P 500 benchmark (Stage 1)
    out/stage1_log.json        - Stage 1 metadata
    out/stage2_log.json        - Stage 2 metadata
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from config import OUTPUT_DIR
from fetcher import fetch_stock_data
from output import save_stage1_outputs, save_stage2_outputs
from returns import calculate_benchmark_returns, calculate_returns
from tickers import (
    filter_non_etf_tickers,
    load_mentioned_tickers,
    load_reddit_date_range,
    normalize_tickers,
)

# -----------------------------------------------------------------------------
# Stage Discovery Functions
# -----------------------------------------------------------------------------


def stage1_has_outputs(output_dir: Path) -> bool:
    """Check if Stage 1 outputs exist."""
    prices_path = output_dir / "stock_prices.parquet"
    return prices_path.exists()


def stage2_has_outputs(output_dir: Path) -> bool:
    """Check if Stage 2 outputs exist and are not stale.

    Stage 2 is stale if stock_prices.parquet is newer than stock_returns.parquet.
    """
    prices_path = output_dir / "stock_prices.parquet"
    returns_path = output_dir / "stock_returns.parquet"

    if not returns_path.exists():
        return False

    if not prices_path.exists():
        # No Stage 1 output - Stage 2 can't be valid
        return False

    # Check staleness: Stage 2 output must be newer than Stage 1 input
    prices_mtime = os.path.getmtime(prices_path)
    returns_mtime = os.path.getmtime(returns_path)

    if prices_mtime > returns_mtime:
        return False  # Stage 1 is newer -> Stage 2 is stale

    return True


def discover_first_missing_stage(output_dir: Path) -> int:
    """Discover which stage needs to run first.

    Returns:
        Stage number (1 or 2), or 3 if all stages complete.
    """
    if not stage1_has_outputs(output_dir):
        return 1
    if not stage2_has_outputs(output_dir):
        return 2
    return 3  # All stages complete


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    if seconds < 0:
        return "--:--"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# -----------------------------------------------------------------------------
# Stage Execution Functions
# -----------------------------------------------------------------------------


def run_stage1(
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_dir: Path,
    tickers_source: str,
    include_benchmark: bool,
    progress_interval: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Stage 1: Fetch raw data from yfinance."""
    print("\n" + "=" * 60)
    print("STAGE 1: FETCHING DATA FROM YAHOO FINANCE")
    print("=" * 60)

    prices_df, benchmark_df, errors = fetch_stock_data(
        tickers,
        start_date,
        end_date,
        include_benchmark=include_benchmark,
        progress_interval=progress_interval,
    )

    # Calculate benchmark returns (included in Stage 1 output)
    benchmark_returns_df = calculate_benchmark_returns(benchmark_df)

    # Save Stage 1 outputs
    print("\nSaving Stage 1 outputs...")
    save_stage1_outputs(
        prices_df=prices_df,
        benchmark_df=benchmark_df,
        benchmark_returns_df=benchmark_returns_df,
        errors=errors,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        tickers_source=tickers_source,
    )

    return prices_df, benchmark_df


def run_stage2(output_dir: Path) -> pd.DataFrame:
    """Run Stage 2: Calculate returns and the daily volatility battery from Stage 1 prices."""
    print("\n" + "=" * 60)
    print("STAGE 2: CALCULATING RETURNS AND DAILY VOLATILITY MEASURES")
    print("=" * 60)

    # Load Stage 1 prices
    prices_path = output_dir / "stock_prices.parquet"
    if not prices_path.exists():
        print(f"ERROR: Stage 1 output not found: {prices_path}")
        sys.exit(1)

    print(f"Loading prices from: {prices_path}")
    prices_df = pd.read_parquet(prices_path)
    print(f"  -> {len(prices_df):,} rows, {prices_df['ticker'].nunique()} tickers")

    # Calculate returns and the daily volatility battery
    print("\nCalculating returns and daily volatility measures...")
    returns_df = calculate_returns(prices_df)

    # Save Stage 2 outputs
    print("\nSaving Stage 2 outputs...")
    save_stage2_outputs(
        returns_df=returns_df,
        output_dir=output_dir,
        prices_rows=len(prices_df),
        prices_tickers=prices_df["ticker"].nunique(),
    )

    return returns_df


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch and process stock market data (two-stage pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fetch_stock_data.py                    # Auto-discover and run needed stages
    python fetch_stock_data.py --stage 2          # Re-calculate returns only
    python fetch_stock_data.py --force            # Re-fetch everything
    python fetch_stock_data.py --stage 2 --force  # Force re-run Stage 2 only
    python fetch_stock_data.py --tickers AAPL MSFT  # Debugging: fetch a specific subset

Ticker universe and date range are auto-discovered from the Reddit pipeline
stats output (run that first).

Stages:
    1. Fetch raw OHLCV from Yahoo Finance -> stock_prices.parquet
    2. Calculate returns and volatility -> stock_returns.parquet

Re-run Logic:
    - No flags: auto-detect first missing/stale stage, run from there
    - --force: re-run everything from Stage 1
    - --stage N: start checking from Stage N
    - --stage N --force: force re-run from Stage N (requires Stage N-1 outputs)
        """,
    )

    # Stage control
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        metavar="N",
        help="Start discovery from stage N (with --force: re-run from N)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even if outputs exist",
    )

    # Ticker selection (only relevant for Stage 1).
    # By default the ticker list is auto-discovered from the Reddit mention
    # stats; --tickers overrides this for debugging.
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Override the auto-discovered ticker list (debugging only)",
    )

    # Other options
    parser.add_argument(
        "--no-benchmark", action="store_true", help="Skip market benchmark (Stage 1)"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, help="Limit tickers (for testing)")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress table every N tickers (0 to disable)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("STOCK DATA PIPELINE")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Determine which stages to run
    # -------------------------------------------------------------------------

    output_dir: Path = args.output_dir

    if args.force and args.stage is None:
        # --force without --stage: run everything from Stage 1
        run_stages = [1, 2]
        print("Force mode: will reprocess all stages")

    elif args.force and args.stage is not None:
        # --stage N --force: require prior stage outputs, run from N onwards
        start_stage = args.stage

        if start_stage == 2 and not stage1_has_outputs(output_dir):
            print("ERROR: --stage 2 --force requires Stage 1 outputs to exist")
            print(f"  Expected: {output_dir / 'stock_prices.parquet'}")
            return 1

        run_stages = list(range(start_stage, 3))  # e.g., [2] or [1, 2]
        print(f"Force mode: reprocessing from Stage {start_stage} onwards")

    elif args.stage is not None:
        # --stage N without --force: start discovery from Stage N
        run_stages = []
        for stage in range(args.stage, 3):
            if stage == 1:
                has_output = stage1_has_outputs(output_dir)
            else:
                has_output = stage2_has_outputs(output_dir)

            if has_output:
                print(f"Stage {stage}: outputs exist and up-to-date, skipping")
            else:
                run_stages = list(range(stage, 3))
                reason = (
                    "stale"
                    if stage == 2 and stage1_has_outputs(output_dir)
                    else "missing"
                )
                print(f"Stage {stage}: outputs {reason}, will run stages {run_stages}")
                break

        if not run_stages:
            print(
                "\nAll stages already have up-to-date outputs! Use --force to reprocess."
            )
            return 0

    else:
        # No flags: auto-discover first missing stage
        first_missing = discover_first_missing_stage(output_dir)

        if first_missing > 2:
            print(
                "\nAll stages already have up-to-date outputs! Use --force to reprocess."
            )
            return 0

        run_stages = list(range(first_missing, 3))
        reason = "stale" if first_missing == 2 else "missing"
        print(f"Auto-discovered: Stage {first_missing} is first with {reason} outputs")
        print(f"Will run stages: {run_stages}")

    pipeline_start = time.time()

    # -------------------------------------------------------------------------
    # Stage 1: Fetch from yfinance
    # -------------------------------------------------------------------------

    if 1 in run_stages:
        # Date range is always auto-discovered from the Reddit daily totals.
        try:
            start_date, end_date = load_reddit_date_range()
        except (FileNotFoundError, ValueError) as e:
            print(f"\nERROR: Could not auto-discover date range: {e}")
            return 1
        print(
            f"\nAuto-discovered date range from Reddit output: {start_date} -> {end_date}"
        )

        # Ticker list: --tickers override (debugging) or auto-discover.
        if args.tickers:
            tickers = filter_non_etf_tickers(normalize_tickers(args.tickers))
            tickers_source = "manual"
            print(f"Using {len(tickers)} manually specified tickers")
        else:
            try:
                tickers = load_mentioned_tickers()
            except (FileNotFoundError, ValueError) as e:
                print(f"\nERROR: Could not auto-discover tickers: {e}")
                return 1
            tickers_source = "mentions_stats"
            print(f"Loaded {len(tickers)} mentioned tickers from Reddit stats")

        if not tickers:
            print("ERROR: No tickers to fetch!")
            return 1

        if args.limit:
            tickers = tickers[: args.limit]
            print(f"Limited to {len(tickers)} tickers")

        run_stage1(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            tickers_source=tickers_source,
            include_benchmark=not args.no_benchmark,
            progress_interval=args.progress_every,
        )

    # -------------------------------------------------------------------------
    # Stage 2: Calculate returns
    # -------------------------------------------------------------------------

    if 2 in run_stages:
        run_stage2(output_dir)

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Stages run: {run_stages}")
    print(f"Total time: {format_duration(total_time)}")

    # Show final output summary
    prices_path = output_dir / "stock_prices.parquet"
    returns_path = output_dir / "stock_returns.parquet"
    if prices_path.exists():
        prices_df = pd.read_parquet(prices_path)
        print(f"\nOutputs:")
        print(
            f"  stock_prices.parquet: {len(prices_df):,} rows, {prices_df['ticker'].nunique()} tickers"
        )
    if returns_path.exists():
        returns_df = pd.read_parquet(returns_path)
        print(f"  stock_returns.parquet: {len(returns_df):,} rows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
