"""Stats I/O helpers for reading Stage 1 outputs and writing stats files.

This module provides utilities for:
- Sampling records from compressed JSONL files
- Aggregating daily ticker counts
- Computing per-ticker statistics
- Writing CSV outputs

Used by stats/runner.py for the unified stats pass.
"""

from __future__ import annotations

import os
import random
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import orjson
from common.io_utils import iter_zst_ndjson
from helpers.logging_utils import timestamp_to_date

from .utils import percentile

# =============================================================================
# Sampling
# =============================================================================


def sample_file(path: str, sample_size: int = 20) -> Tuple[List[Dict[str, Any]], int]:
    """Reservoir-sample records from a .zst JSONL file.

    Returns (samples, total_count). Samples contain only text/tickers/matched_on.
    """
    reservoir: List[Dict[str, Any]] = []
    count = 0
    for i, item in enumerate(iter_zst_ndjson(path), start=1):
        count = i
        if i <= sample_size:
            reservoir.append(item)
        else:
            j = random.randrange(i)
            if j < sample_size:
                reservoir[j] = item
    # Extract minimal fields for inspection
    samples = [
        {
            "text": r.get("text"),
            "tickers": r.get("tickers"),
            "matched_on": r.get("matched_on"),
        }
        for r in reservoir
    ]
    return samples, count


def sample_file_by_match_type(
    path: str, match_type: str, sample_size: int = 20
) -> Tuple[List[Dict[str, Any]], int]:
    """Reservoir-sample records that have at least one ticker of the given match type.

    Args:
        path: Path to the .zst JSONL file
        match_type: Match type to filter for (e.g., 'ticker_dollar', 'ticker', 'name', 'alias')
        sample_size: Number of samples to collect

    Returns:
        (samples, match_count). Samples contain text/tickers/matched_on,
        filtered to only include tickers matched via the specified type.
    """
    reservoir: List[Dict[str, Any]] = []
    count = 0  # Count of records with this match type

    for item in iter_zst_ndjson(path):
        match_types_dict = item.get("match_types", {})

        # Filter to only matches of the target type
        matched_tickers = [t for t, mt in match_types_dict.items() if mt == match_type]
        if not matched_tickers:
            continue

        count += 1

        # Reservoir sampling
        if count <= sample_size:
            reservoir.append(item)
        else:
            j = random.randrange(count)
            if j < sample_size:
                reservoir[j] = item

    # Extract minimal fields, filtered to only tickers of the target match type
    samples = []
    for r in reservoir:
        match_types_dict = r.get("match_types", {})
        matched_on = r.get("matched_on", {})

        # Filter to only the target match type
        filtered_tickers = [t for t, mt in match_types_dict.items() if mt == match_type]
        filtered_matched_on = {t: matched_on.get(t, "") for t in filtered_tickers}

        samples.append(
            {
                "text": r.get("text"),
                "tickers": filtered_tickers,
                "matched_on": filtered_matched_on,
            }
        )

    return samples, count


# =============================================================================
# Aggregation from Stage 1 mention files
# =============================================================================


# Match type categories for statistics
MATCH_TYPE_CATEGORIES = ("ticker_dollar", "ticker", "name", "alias")




def aggregate_mention_file(
    path: str,
) -> Tuple[
    Dict[str, Dict[str, int]],
    Dict[str, int],
    int,
    int,
    Dict[str, int],
]:
    """Stream a mention .jsonl.zst file and aggregate daily ticker counts.

    Returns:
        daily_counts: Dict[date, Dict[ticker, count]]
        ticker_totals: Dict[ticker, total_count]
        record_count: total records processed
        target_pairs: count of (text, target) pairs for targeted sentiment models
        match_type_counts: Dict[match_type, count] - how many tickers matched by each method
    """
    daily: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: Dict[str, int] = defaultdict(int)
    count = 0
    target_pairs = 0

    # Track match type statistics
    match_type_counts: Dict[str, int] = {mt: 0 for mt in MATCH_TYPE_CATEGORIES}

    for record in iter_zst_ndjson(path):
        count += 1
        date_str = timestamp_to_date(record["created_utc"])
        tickers = record.get("tickers", [])

        # Track total targets (sentiment instances)
        target_pairs += len(tickers)

        # Track match type distribution
        match_types = record.get("match_types", {})
        for mt in match_types.values():
            if mt in match_type_counts:
                match_type_counts[mt] += 1

        for ticker in tickers:
            daily[date_str][ticker] += 1
            totals[ticker] += 1

    return (
        {d: dict(tc) for d, tc in daily.items()},
        dict(totals),
        count,
        target_pairs,
        match_type_counts,
    )


# =============================================================================
# Ticker stats computation
# =============================================================================


def compute_ticker_stats(
    daily_counts: Dict[str, Dict[str, int]],
    ticker_totals_subs: Optional[Dict[str, int]] = None,
    ticker_totals_coms: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Compute per-ticker stats from daily counts.

    Args:
        daily_counts: Dict[date, Dict[ticker, count]]
        ticker_totals_subs: Optional ticker → submission count
        ticker_totals_coms: Optional ticker → comment count

    Returns:
        List of dicts with ticker stats (for CSV output).
    """
    if not daily_counts:
        return []

    sorted_dates = sorted(daily_counts.keys())
    total_days = len(sorted_dates)
    all_tickers = sorted(set(t for day in daily_counts.values() for t in day))

    ticker_totals_subs = ticker_totals_subs or {}
    ticker_totals_coms = ticker_totals_coms or {}

    rows = []
    for ticker in all_tickers:
        counts = [daily_counts.get(d, {}).get(ticker, 0) for d in sorted_dates]
        total_mentions = sum(counts)
        active_days = sum(1 for c in counts if c > 0)

        rows.append(
            {
                "ticker": ticker,
                "total_mentions": total_mentions,
                "submissions": ticker_totals_subs.get(ticker, 0),
                "comments": ticker_totals_coms.get(ticker, 0),
                "active_days": active_days,
                "total_days": total_days,
                "mean_per_day": (
                    round(total_mentions / total_days, 3) if total_days else 0
                ),
                "median_per_day": round(statistics.median(counts), 3) if counts else 0,
                "p90_per_day": round(percentile(counts, 90), 3),
                "p95_per_day": round(percentile(counts, 95), 3),
                "max_per_day": max(counts) if counts else 0,
            }
        )

    return rows


# =============================================================================
# CSV writing
# =============================================================================


def write_daily_totals_csv(path: str, daily_counts: Dict[str, Dict[str, int]]) -> None:
    """Write simple daily totals CSV (date, total_mentions).

    Useful for investigating which timeframe to use for regression analysis.
    """
    if not daily_counts:
        return

    sorted_dates = sorted(daily_counts.keys())

    with open(path, "w", encoding="utf-8") as f:
        f.write("date,total_mentions\n")
        for d in sorted_dates:
            total = sum(daily_counts[d].values())
            f.write(f"{d},{total}\n")


def write_daily_counts_csv(path: str, daily_counts: Dict[str, Dict[str, int]]) -> None:
    """Write daily ticker counts to CSV (date rows, ticker columns)."""
    if not daily_counts:
        return

    sorted_dates = sorted(daily_counts.keys())
    ticker_totals: Dict[str, int] = defaultdict(int)
    for day_counts in daily_counts.values():
        for t, c in day_counts.items():
            ticker_totals[t] += int(c)

    all_tickers = sorted(
        ticker_totals.keys(),
        key=lambda t: (-ticker_totals.get(t, 0), t),
    )

    if not sorted_dates or not all_tickers:
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write("date," + ",".join(all_tickers) + "\n")
        for d in sorted_dates:
            row = [str(daily_counts.get(d, {}).get(t, 0)) for t in all_tickers]
            f.write(f"{d}," + ",".join(row) + "\n")


def write_ticker_stats_csv(
    path: str,
    rows: List[Dict[str, Any]],
    global_mode: bool = False,
) -> None:
    """Write ticker stats to CSV.

    Args:
        path: Output file path
        rows: List of ticker stat dicts
        global_mode: If True, use dataset_coverage columns instead of subs/coms
    """
    if not rows:
        return

    if global_mode:
        header = (
            "ticker,total_mentions,active_days,total_days,dataset_coverage,num_datasets,"
            "mean_per_day,median_per_day,p90_per_day,p95_per_day,max_per_day\n"
        )
        fmt = (
            "{ticker},{total_mentions},{active_days},{total_days},"
            "{dataset_coverage},{num_datasets},{mean_per_day},"
            "{median_per_day},{p90_per_day},{p95_per_day},{max_per_day}\n"
        )
    else:
        header = (
            "ticker,total_mentions,submissions,comments,active_days,total_days,"
            "mean_per_day,median_per_day,p90_per_day,p95_per_day,max_per_day\n"
        )
        fmt = (
            "{ticker},{total_mentions},{submissions},{comments},"
            "{active_days},{total_days},{mean_per_day},"
            "{median_per_day},{p90_per_day},{p95_per_day},{max_per_day}\n"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for row in rows:
            f.write(fmt.format(**row))


def write_json(path: str, data: Any) -> None:
    """Write data to JSON file with pretty formatting."""
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
