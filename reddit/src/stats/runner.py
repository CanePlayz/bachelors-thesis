"""Unified stats runner for Stage 1 outputs.

Entry point for computing all statistics from Stage 1 mention files.
Writes per-dataset and global aggregated stats to out/stats/.

Output structure:
    out/stats/
        global/
            aggregated/                    <- Combined data files
                match_counts.json
                daily_totals.csv
                ticker_stats.csv
                daily_ticker_counts.csv
            samples/                       <- Sample files by match type
                all_samples.json           <- Combined samples
                ticker_dollar_samples.json
                ticker_samples.json
                name_samples.json
                alias_samples.json
            plots/                         <- All plots
                daily_total_mentions.png
                top_tickers_over_time.png
                ...
        datasets/{dataset}/                <- Per-dataset stats
            stats.json
            daily_ticker_counts.csv
            ticker_stats.csv
            samples/
                all_samples.json
                ticker_dollar_samples.json
                ticker_samples.json
                name_samples.json
                alias_samples.json

Usage:
    from stats.runner import run_stats_pass
    run_stats_pass()
"""

from __future__ import annotations

import os
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from data.config import STAGE1_OUT_DIR, STATS_OUT_DIR
from helpers.logging_utils import format_duration, print_banner

from .io import (MATCH_TYPE_CATEGORIES, aggregate_mention_file,
                 compute_ticker_stats, sample_file, sample_file_by_match_type,
                 write_daily_counts_csv, write_daily_totals_csv, write_json,
                 write_ticker_stats_csv)
from .models import DatasetStats
from .plotting import generate_plots_from_global_csv
from .utils import percentile


# Type definitions for match counts
class MatchCountsDict(TypedDict):
    submissions: int
    comments: int
    total: int
    target_pairs: int
    by_match_type: Dict[str, int]


# Type for samples organized by match type
SamplesByType = Dict[
    str, Dict[str, List[Dict[str, Any]]]
]  # match_type -> source -> samples


# =============================================================================
# Per-dataset stats
# =============================================================================


def compute_dataset_stats(
    ds_name: str,
    stage1_dir: str,
    stats_datasets_dir: str,
    sample_size: int = 20,
) -> Tuple[
    Optional[DatasetStats],
    Dict[str, Dict[str, int]],
    MatchCountsDict,
    Dict[str, List[Dict[str, Any]]],
    SamplesByType,
]:
    """Compute and write stats for a single dataset.

    Returns:
        - DatasetStats or None
        - daily_counts: date -> ticker -> count
        - match_counts: MatchCountsDict
        - all_samples: source -> samples (for all match types combined)
        - samples_by_type: match_type -> source -> samples
    """
    ds_input = os.path.join(stage1_dir, ds_name)
    ds_output = os.path.join(stats_datasets_dir, ds_name)
    os.makedirs(ds_output, exist_ok=True)

    sub_path = os.path.join(ds_input, "s1_submission_mentions.jsonl.zst")
    com_path = os.path.join(ds_input, "s1_comment_mentions.jsonl.zst")

    # Return empty results if no files exist
    empty_match_counts: MatchCountsDict = {
        "submissions": 0,
        "comments": 0,
        "total": 0,
        "target_pairs": 0,
        "by_match_type": {mt: 0 for mt in MATCH_TYPE_CATEGORIES},
    }
    if not os.path.exists(sub_path) and not os.path.exists(com_path):
        return None, {}, empty_match_counts, {}, {}

    # Aggregate both files
    daily_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ticker_subs: Dict[str, int] = defaultdict(int)
    ticker_coms: Dict[str, int] = defaultdict(int)
    sub_count = com_count = 0

    # All samples combined (for backwards compatibility)
    all_samples: Dict[str, List[Dict[str, Any]]] = {"submissions": [], "comments": []}

    # Samples organized by match type
    samples_by_type: SamplesByType = {
        mt: {"submissions": [], "comments": []} for mt in MATCH_TYPE_CATEGORIES
    }

    sub_target_pairs = com_target_pairs = 0

    # Track match type statistics
    match_type_totals: Dict[str, int] = {mt: 0 for mt in MATCH_TYPE_CATEGORIES}

    if os.path.exists(sub_path):
        # Combined samples
        all_samples["submissions"], _ = sample_file(sub_path, sample_size)

        # Per-match-type samples
        for mt in MATCH_TYPE_CATEGORIES:
            samples_by_type[mt]["submissions"], _ = sample_file_by_match_type(
                sub_path, mt, sample_size
            )

        sub_daily, sub_totals, sub_count, sub_target_pairs, sub_match_types = (
            aggregate_mention_file(sub_path)
        )
        for d, tc in sub_daily.items():
            for t, c in tc.items():
                daily_counts[d][t] += c
        for t, c in sub_totals.items():
            ticker_subs[t] += c
        for mt, cnt in sub_match_types.items():
            match_type_totals[mt] += cnt

    if os.path.exists(com_path):
        # Combined samples
        all_samples["comments"], _ = sample_file(com_path, sample_size)

        # Per-match-type samples
        for mt in MATCH_TYPE_CATEGORIES:
            samples_by_type[mt]["comments"], _ = sample_file_by_match_type(
                com_path, mt, sample_size
            )

        com_daily, com_totals, com_count, com_target_pairs, com_match_types = (
            aggregate_mention_file(com_path)
        )
        for d, tc in com_daily.items():
            for t, c in tc.items():
                daily_counts[d][t] += c
        for t, c in com_totals.items():
            ticker_coms[t] += c
        for mt, cnt in com_match_types.items():
            match_type_totals[mt] += cnt

    # Convert defaultdicts
    daily_counts_final = {d: dict(tc) for d, tc in daily_counts.items()}

    # Compute ticker stats
    ticker_rows = compute_ticker_stats(
        daily_counts_final, dict(ticker_subs), dict(ticker_coms)
    )

    # Build match counts with match type breakdown
    match_counts: MatchCountsDict = {
        "submissions": sub_count,
        "comments": com_count,
        "total": sub_count + com_count,
        "target_pairs": sub_target_pairs + com_target_pairs,
        "by_match_type": match_type_totals,
    }

    # Build DatasetStats
    sorted_dates = sorted(daily_counts_final.keys()) if daily_counts_final else []
    all_tickers = sorted(set(t for day in daily_counts_final.values() for t in day))
    top_tickers = sorted(
        [(t, ticker_subs.get(t, 0) + ticker_coms.get(t, 0)) for t in all_tickers],
        key=lambda x: -x[1],
    )[:20]

    ds_stats = DatasetStats(
        dataset=ds_name,
        total_submissions=sub_count,
        matched_submissions=sub_count,
        total_comments=com_count,
        matched_comments=com_count,
        unique_tickers_subs=len(ticker_subs),
        unique_tickers_comments=len(ticker_coms),
        date_range=(sorted_dates[0], sorted_dates[-1]) if sorted_dates else None,
        top_tickers=top_tickers,
    )

    # Write outputs
    write_json(os.path.join(ds_output, "stats.json"), ds_stats.to_dict())
    write_daily_counts_csv(
        os.path.join(ds_output, "daily_ticker_counts.csv"), daily_counts_final
    )
    write_ticker_stats_csv(os.path.join(ds_output, "ticker_stats.csv"), ticker_rows)

    # Write samples to samples/ subdirectory
    samples_dir = os.path.join(ds_output, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    write_json(os.path.join(samples_dir, "all_samples.json"), all_samples)
    for mt in MATCH_TYPE_CATEGORIES:
        write_json(os.path.join(samples_dir, f"{mt}_samples.json"), samples_by_type[mt])

    return (
        ds_stats,
        daily_counts_final,
        match_counts,
        all_samples,
        samples_by_type,
    )


# =============================================================================
# Global aggregation
# =============================================================================


def compute_global_stats(
    all_daily: Dict[str, Dict[str, Dict[str, int]]],
    all_match: Dict[str, MatchCountsDict],
    all_samples: Dict[str, Dict[str, List[Dict[str, Any]]]],
    all_samples_by_type: Dict[str, SamplesByType],
    global_dir: str,
) -> None:
    """Aggregate cross-dataset stats and write global outputs.

    Creates subdirectories:
        global/aggregated/  <- data files
        global/samples/     <- sample files by match type
        global/plots/       <- plots (created by plotting.py)
    """
    # Create subdirectories
    aggregated_dir = os.path.join(global_dir, "aggregated")
    samples_dir = os.path.join(global_dir, "samples")
    os.makedirs(aggregated_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Merge daily counts
    merged: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ds_daily in all_daily.values():
        for d, tc in ds_daily.items():
            for t, c in tc.items():
                merged[d][t] += c
    merged_final = {d: dict(tc) for d, tc in merged.items()}

    # Build global ticker stats with dataset coverage
    all_dates = sorted(merged_final.keys())
    all_tickers = sorted(set(t for day in merged_final.values() for t in day))
    total_days = len(all_dates)
    num_datasets = len(all_daily)

    global_rows = []
    for ticker in all_tickers:
        counts = [merged_final.get(d, {}).get(ticker, 0) for d in all_dates]
        dataset_coverage = sum(
            1
            for ds_daily in all_daily.values()
            if any(ticker in day for day in ds_daily.values())
        )

        global_rows.append(
            {
                "ticker": ticker,
                "total_mentions": sum(counts),
                "active_days": sum(1 for c in counts if c > 0),
                "total_days": total_days,
                "dataset_coverage": dataset_coverage,
                "num_datasets": num_datasets,
                "mean_per_day": round(sum(counts) / total_days, 3) if total_days else 0,
                "median_per_day": round(statistics.median(counts), 3) if counts else 0,
                "p90_per_day": round(percentile(counts, 90), 3),
                "p95_per_day": round(percentile(counts, 95), 3),
                "max_per_day": max(counts) if counts else 0,
            }
        )

    # Write aggregated outputs
    csv_path = os.path.join(aggregated_dir, "daily_ticker_counts.csv")
    write_daily_counts_csv(csv_path, merged_final)
    print(f"  Wrote: {csv_path}")

    totals_path = os.path.join(aggregated_dir, "daily_totals.csv")
    write_daily_totals_csv(totals_path, merged_final)
    print(f"  Wrote: {totals_path}")

    stats_path = os.path.join(aggregated_dir, "ticker_stats.csv")
    write_ticker_stats_csv(stats_path, global_rows, global_mode=True)
    print(f"  Wrote: {stats_path}")

    # Add global totals to match counts (including match type breakdown)
    global_submissions = sum(m["submissions"] for m in all_match.values())
    global_comments = sum(m["comments"] for m in all_match.values())
    global_target_pairs = sum(m["target_pairs"] for m in all_match.values())

    # Aggregate match type counts across all datasets
    global_match_types: Dict[str, int] = {mt: 0 for mt in MATCH_TYPE_CATEGORIES}
    for m in all_match.values():
        by_type = m["by_match_type"]
        for mt, cnt in by_type.items():
            if mt in global_match_types:
                global_match_types[mt] += cnt

    all_match_with_global: Dict[str, MatchCountsDict] = {
        "_global": {
            "submissions": global_submissions,
            "comments": global_comments,
            "total": global_submissions + global_comments,
            "target_pairs": global_target_pairs,
            "by_match_type": global_match_types,
        },
        **all_match,
    }

    counts_path = os.path.join(aggregated_dir, "match_counts.json")
    write_json(counts_path, all_match_with_global)
    print(f"  Wrote: {counts_path}")

    # Write combined samples
    samples_path = os.path.join(samples_dir, "all_samples.json")
    write_json(samples_path, all_samples)
    print(f"  Wrote: {samples_path}")

    # Write per-match-type samples (merged across datasets)
    for mt in MATCH_TYPE_CATEGORIES:
        merged_mt_samples: Dict[str, List[Dict[str, Any]]] = {
            "submissions": [],
            "comments": [],
        }
        for ds_samples_by_type in all_samples_by_type.values():
            if mt in ds_samples_by_type:
                merged_mt_samples["submissions"].extend(
                    ds_samples_by_type[mt].get("submissions", [])
                )
                merged_mt_samples["comments"].extend(
                    ds_samples_by_type[mt].get("comments", [])
                )

        mt_path = os.path.join(samples_dir, f"{mt}_samples.json")
        write_json(mt_path, merged_mt_samples)
        print(f"  Wrote: {mt_path}")


# =============================================================================
# Main entry point
# =============================================================================


def run_stats_pass(
    stage1_dir: str = STAGE1_OUT_DIR,
    stats_dir: str = STATS_OUT_DIR,
    datasets: Optional[List[str]] = None,
    sample_size: int = 20,
    generate_plots_flag: bool = True,
) -> Tuple[List[DatasetStats], Dict[str, MatchCountsDict]]:
    """Run the complete stats pass on Stage 1 outputs.

    Both the normal pipeline and --stats-plots-only mode call this.
    """
    print_banner("STATS PASS: Aggregating Stage 1 outputs")
    start = time.time()

    global_dir = os.path.join(stats_dir, "global")
    datasets_dir = os.path.join(stats_dir, "datasets")
    plots_dir = os.path.join(global_dir, "plots")
    os.makedirs(global_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Auto-discover datasets
    if datasets is None:
        datasets = []
        if os.path.isdir(stage1_dir):
            for name in sorted(os.listdir(stage1_dir)):
                if os.path.isdir(os.path.join(stage1_dir, name)):
                    datasets.append(name)

    if not datasets:
        print("No datasets found in Stage 1 outputs")
        return [], {}

    print(f"Processing {len(datasets)} datasets: {', '.join(datasets)}")

    # Process each dataset
    all_stats: List[DatasetStats] = []
    all_daily: Dict[str, Dict[str, Dict[str, int]]] = {}
    all_match: Dict[str, MatchCountsDict] = {}
    all_samples: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    all_samples_by_type: Dict[str, SamplesByType] = {}

    for ds in datasets:
        print(f"\n  [{ds}] Computing stats...")
        ds_stats, daily, match, samples, samples_by_type = (
            compute_dataset_stats(ds, stage1_dir, datasets_dir, sample_size)
        )
        if ds_stats:
            all_stats.append(ds_stats)
            all_daily[ds] = daily
            all_match[ds] = match
            all_samples[ds] = samples
            all_samples_by_type[ds] = samples_by_type
            print(f"    Mentions: {match['total']:,}")
            # Print match type breakdown
            by_type = match["by_match_type"]
            if by_type:
                type_str = ", ".join(
                    f"{mt}: {cnt:,}" for mt, cnt in by_type.items() if cnt > 0
                )
                if type_str:
                    print(f"    By type: {type_str}")

    # Global aggregation
    print("\nComputing global aggregations...")
    compute_global_stats(
        all_daily,
        all_match,
        all_samples,
        all_samples_by_type,
        global_dir,
    )

    # Plots
    if generate_plots_flag:
        print("\nGenerating plots...")
        # Read from new aggregated location
        csv_path = os.path.join(global_dir, "aggregated", "daily_ticker_counts.csv")
        if os.path.exists(csv_path):
            generate_plots_from_global_csv(csv_path, plots_dir)

    print(f"\nStats pass complete in {format_duration(time.time() - start)}")
    print(
        f"  Datasets: {len(all_stats)}, Mentions: {sum(c['total'] for c in all_match.values()):,}"
    )

    return all_stats, all_match
