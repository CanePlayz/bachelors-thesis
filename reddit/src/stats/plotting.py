"""Plotting utilities for v2 pipeline (stats module).

This module provides visualization functions for analyzing ticker mention patterns
from processed Reddit data. It generates several plot types to help understand
the distribution and temporal patterns of stock ticker mentions.

Reads from: out/stats/global/global_daily_ticker_counts.csv
Writes to: out/stats/global/

Entry point:
- generate_plots_from_global_csv(): Reads pre-aggregated global CSV
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict


def generate_plots_from_global_csv(csv_path: str, output_dir: str) -> None:
    """Generate plots from pre-aggregated global_daily_ticker_counts.csv.

    This is the preferred entry point for the unified stats runner.
    It reads the already-aggregated CSV instead of re-reading Stage 1 outputs.

    Args:
        csv_path: Path to global_daily_ticker_counts.csv
        output_dir: Directory to write plot PNGs
    """
    # Attempt to import plotting dependencies
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed, skipping plots")
        return

    from datetime import datetime

    if not os.path.exists(csv_path):
        print(f"No global daily counts CSV found at: {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Parse the global CSV
    global_daily: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        tickers = header[1:]  # All columns after 'date'

        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            date = parts[0]
            counts = parts[1:]
            for i, c in enumerate(counts):
                if i < len(tickers) and c:
                    global_daily[date][tickers[i]] = int(c)

    if not global_daily:
        print("No data in global daily counts CSV")
        return

    # Sort dates chronologically
    sorted_dates = sorted(global_daily.keys())
    dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in sorted_dates]
    dates = np.array(dates_dt)

    # Calculate total mentions per ticker
    total_counts: Dict[str, int] = defaultdict(int)
    for day_counts in global_daily.values():
        for t, c in day_counts.items():
            total_counts[t] += c

    print(
        f"  Generating plots from {len(sorted_dates)} days, {len(total_counts)} tickers..."
    )

    # --- Plot 1: Total mentions per day ---
    fig, ax = plt.subplots(figsize=(14, 6))
    daily_totals = [sum(global_daily[d].values()) for d in sorted_dates]
    ax.bar(dates, daily_totals, width=1.0, alpha=0.7, color="steelblue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Mentions")
    ax.set_title("Daily Total Ticker Mentions (All Subreddits)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "daily_total_mentions.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    # --- Plot 2: Top 15 tickers over time ---
    fig, ax = plt.subplots(figsize=(14, 8))
    top_tickers = sorted(total_counts.items(), key=lambda x: -x[1])[:15]
    top_symbols = [t for t, _ in top_tickers]
    for ticker in top_symbols:
        counts = [global_daily[d].get(ticker, 0) for d in sorted_dates]
        ax.plot(dates, counts, label=ticker, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Mentions per day")
    ax.set_title("Top 15 Tickers: Daily Mentions Over Time")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "top_tickers_over_time.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    # --- Plot 3: Sample size histogram ---
    fig, ax = plt.subplots(figsize=(12, 6))
    counts_arr = np.array(list(total_counts.values()))
    if len(counts_arr) > 0 and max(counts_arr) > 0:
        ax.hist(
            counts_arr,
            bins=np.logspace(0, np.log10(max(counts_arr) + 1), 40),
            color="steelblue",
            alpha=0.8,
        )
        ax.set_xscale("log")
    ax.set_xlabel("Total mentions per ticker (log scale)")
    ax.set_ylabel("# of tickers")
    ax.set_title("Sample Sizes: Mentions per Ticker")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "ticker_sample_sizes_hist.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    # --- Plot 4: Coverage curve ---
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_counts = sorted(total_counts.values(), reverse=True)
    if sorted_counts:
        cumulative = np.cumsum(sorted_counts)
        total = cumulative[-1] if cumulative.size else 1
        ranks = np.arange(1, len(sorted_counts) + 1)
        ax.plot(ranks, cumulative / total, color="darkorange", linewidth=2)
        ax.set_xscale("log")
    ax.set_xlabel("Ticker rank (by total mentions)")
    ax.set_ylabel("Cumulative share of mentions")
    ax.set_title("Coverage Curve: How Many Tickers Capture Most Mentions?")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "ticker_coverage_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    # --- Plot 5: Top 30 bar chart ---
    fig, ax = plt.subplots(figsize=(12, 10))
    top30 = sorted(total_counts.items(), key=lambda x: -x[1])[:30]
    tickers_list = [t for t, _ in top30]
    counts_list = [c for _, c in top30]
    ax.barh(tickers_list[::-1], counts_list[::-1], color="steelblue")
    ax.set_xlabel("Total Mentions")
    ax.set_title("Top 30 Most Mentioned Tickers (All Datasets)")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "top30_tickers.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    print("  Plotting complete!")
