"""Stage 1: Pipeline orchestration and dataset processing for v2 (src).

This module is the core of Stage 1 (ticker mention extraction). It handles:
- Dataset discovery from .zst files
- Processing submissions and comments through ticker extraction
- Writing compressed mention outputs
- Progress reporting and timing

Note: Stats computation is now handled by the stats pass (stats/runner.py),
which runs after Stage 1 completes. Stage 1 only writes mention outputs.

Main functions:
- discover_datasets(): Find available datasets in the datasets directory
- process_dataset(): Process a single dataset (submissions + comments)
- run_stage1(): Orchestrate Stage 1 across all datasets

Output files (per dataset in stage1_mentions/{dataset}/):
- s1_submission_mentions.jsonl.zst: Submission mention records
- s1_comment_mentions.jsonl.zst: Comment mention records
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import orjson

# Import streaming IO utilities from common module
from common.io_utils import (
    StreamProgress,
    iter_zst_ndjson_with_progress,
    write_jsonl_zst,
)

# Import configuration paths
from data.config import DATASETS_DIR, STAGE1_OUT_DIR

# Import ticker universe for extraction
from data.ticker_universe import TickerUniverse, build_ticker_universe

# Import logging/formatting helpers
from helpers.logging_utils import (
    format_duration,
    format_progress_file_complete,
    format_progress_line,
    print_progress_header,
    timestamp_to_date,
)

# Import processing functions
from .processing import MentionRecord, process_comment, process_submission


# Simple stats tracking for Stage 1 runtime info
class Stage1Stats:
    """Minimal stats tracked during Stage 1 extraction."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.total_submissions = 0
        self.matched_submissions = 0
        self.total_comments = 0
        self.matched_comments = 0
        self.runtime_seconds = 0.0


def discover_datasets(datasets_dir: str) -> List[str]:
    """Discover available datasets from files in datasets_dir.

    Scans for files matching *_submissions.zst or *_comments.zst
    and extracts unique dataset names.

    Args:
        datasets_dir: Path to directory containing .zst files

    Returns:
        Sorted list of dataset names
    """
    # Return empty if directory doesn't exist
    if not os.path.isdir(datasets_dir):
        return []

    # Set to track unique dataset names
    found = set()

    # Scan directory for matching files
    for fname in os.listdir(datasets_dir):
        # Check for submission or comment files
        if fname.endswith("_submissions.zst") or fname.endswith("_comments.zst"):
            # Extract dataset name by removing suffix
            name = fname.replace("_submissions.zst", "").replace("_comments.zst", "")
            found.add(name)

    # Return sorted list
    return sorted(found)


def process_dataset(
    dataset_name: str,
    universe: TickerUniverse,
    datasets_dir: str = DATASETS_DIR,
    out_dir: str = STAGE1_OUT_DIR,
    progress_every: int = 50000,
    force: bool = False,
    skip_submissions: bool = False,
    tickers_only: bool = False,
) -> Optional[Stage1Stats]:
    """Process a single dataset (submissions + comments).

    Reads the .zst files, extracts ticker mentions, and writes
    output files with mention records. Stats computation is handled
    separately by the stats pass (stats/runner.py).

    Args:
        dataset_name: Name of the dataset (e.g., 'investing', 'wallstreetbets')
        universe: TickerUniverse for ticker matching
        datasets_dir: Directory containing source .zst files
        out_dir: Output directory for Stage 1 results
        progress_every: Print progress every N records
        force: If True, reprocess even if outputs exist
        skip_submissions: If True, skip submissions and only process comments
        tickers_only: If True, only match $-prefixed and bare caps tickers

    Returns:
        Stage1Stats with runtime info, or None if skipped
    """
    # Create output directory for this dataset
    dataset_out = os.path.join(out_dir, dataset_name)
    os.makedirs(dataset_out, exist_ok=True)

    # Check if outputs already exist (skip unless force=True)
    sub_out = os.path.join(dataset_out, "s1_submission_mentions.jsonl.zst")
    com_out = os.path.join(dataset_out, "s1_comment_mentions.jsonl.zst")
    if (
        not force
        and not skip_submissions
        and (os.path.exists(sub_out) or os.path.exists(com_out))
    ):
        print(f"\n[{dataset_name}] Skipping (outputs exist, use --force to reprocess)")
        return None

    # Initialize stats tracking
    stats = Stage1Stats(dataset=dataset_name)
    start_time = time.time()

    # Build paths to source files
    sub_file = os.path.join(datasets_dir, f"{dataset_name}_submissions.zst")
    com_file = os.path.join(datasets_dir, f"{dataset_name}_comments.zst")

    # --- Process Submissions ---
    if skip_submissions:
        print(f"\n[{dataset_name}] Skipping submissions (resuming from comments)")
    elif os.path.exists(sub_file):
        print(f"\n[{dataset_name}] Processing submissions...")
        print_progress_header(prefix="  ")

        # Collect mention records
        sub_mentions: List[dict] = []

        # Timing for progress calculation
        phase_start = time.time()
        processed = 0
        progress = StreamProgress()

        # Stream through submissions
        for idx, row in enumerate(iter_zst_ndjson_with_progress(sub_file, progress), 1):
            processed = idx

            # Try to extract tickers from this submission
            record = process_submission(row, universe, tickers_only=tickers_only)

            # If we got a match, record it
            if record:
                sub_mentions.append(record.to_dict())

            # Print progress periodically
            if idx % progress_every == 0:
                elapsed = time.time() - phase_start
                rate = idx / elapsed if elapsed > 0 else 0
                pct = progress.progress_pct
                eta = (elapsed / pct * (100 - pct)) if pct > 0 else 0
                print(
                    format_progress_line(
                        dataset_name,
                        "submissions",
                        idx,
                        len(sub_mentions),
                        rate,
                        pct,
                        eta,
                        prefix="  ",
                    )
                )

        # Update stats
        stats.total_submissions = processed
        stats.matched_submissions = len(sub_mentions)

        # Print final progress line
        elapsed = time.time() - phase_start
        rate = processed / elapsed if elapsed > 0 else 0
        print(
            format_progress_line(
                dataset_name,
                "submissions",
                processed,
                len(sub_mentions),
                rate,
                100.0,
                0,
                prefix="  ",
            )
        )

        print(
            format_progress_file_complete(
                dataset_name,
                "submissions",
                processed,
                len(sub_mentions),
                prefix="  ",
            )
        )

        # Write mention records to compressed JSONL
        if sub_mentions:
            out_path = os.path.join(dataset_out, "s1_submission_mentions.jsonl.zst")
            write_jsonl_zst(out_path, sub_mentions)
            print(
                f"  Wrote {len(sub_mentions):,} submission mentions → {os.path.basename(out_path)}"
            )

    # --- Process Comments ---
    if os.path.exists(com_file):
        print(f"\n[{dataset_name}] Processing comments...")
        print_progress_header(prefix="  ")

        # Collect mention records
        com_mentions: List[dict] = []

        # Timing for progress calculation
        phase_start = time.time()
        processed = 0
        progress = StreamProgress()

        # Stream through comments
        for idx, row in enumerate(iter_zst_ndjson_with_progress(com_file, progress), 1):
            processed = idx

            # Try to extract tickers from this comment
            record = process_comment(row, universe, tickers_only=tickers_only)

            # If we got a match, record it
            if record:
                com_mentions.append(record.to_dict())

            # Print progress periodically
            if idx % progress_every == 0:
                elapsed = time.time() - phase_start
                rate = idx / elapsed if elapsed > 0 else 0
                pct = progress.progress_pct
                eta = (elapsed / pct * (100 - pct)) if pct > 0 else 0
                print(
                    format_progress_line(
                        dataset_name,
                        "comments",
                        idx,
                        len(com_mentions),
                        rate,
                        pct,
                        eta,
                        prefix="  ",
                    )
                )

        # Update stats
        stats.total_comments = processed
        stats.matched_comments = len(com_mentions)

        # Print final progress line
        elapsed = time.time() - phase_start
        rate = processed / elapsed if elapsed > 0 else 0
        print(
            format_progress_line(
                dataset_name,
                "comments",
                processed,
                len(com_mentions),
                rate,
                100.0,
                0,
                prefix="  ",
            )
        )

        print(
            format_progress_file_complete(
                dataset_name,
                "comments",
                processed,
                len(com_mentions),
                prefix="  ",
            )
        )

        # Write mention records to compressed JSONL
        if com_mentions:
            out_path = os.path.join(dataset_out, "s1_comment_mentions.jsonl.zst")
            write_jsonl_zst(out_path, com_mentions)
            print(
                f"  Wrote {len(com_mentions):,} comment mentions → {os.path.basename(out_path)}"
            )

    # Record total runtime
    stats.runtime_seconds = time.time() - start_time

    return stats


def run_stage1(
    datasets: Optional[List[str]] = None,
    datasets_dir: str = DATASETS_DIR,
    out_dir: str = STAGE1_OUT_DIR,
    progress_every: int = 50000,
    start_from: Optional[Tuple[str, str]] = None,
    force: bool = False,
    tickers_only: bool = False,
) -> List[Stage1Stats]:
    """Run Stage 1: Ticker mention extraction.

    This is the main entry point for Stage 1 of the pipeline.
    It handles dataset discovery, ticker universe loading, processing
    each dataset, and writing mention outputs.

    Stats computation is handled separately by the stats pass (stats/runner.py).

    Args:
        datasets: List of dataset names. If None, auto-discover from datasets_dir.
        datasets_dir: Directory containing .zst dataset files.
        out_dir: Output directory for Stage 1 results.
        progress_every: Print progress every N items.
        start_from: Optional (dataset, phase) tuple to resume from.
        force: If True, reprocess datasets even if outputs exist.
        tickers_only: If True, only match $-prefixed and bare caps tickers.

    Returns:
        List of Stage1Stats for each processed dataset.
    """
    # Print banner
    print("\n" + "=" * 70)
    print("STAGE 1: TICKER MENTION EXTRACTION")
    print("=" * 70)

    # Discover datasets if not provided
    print(f"Looking for datasets in: {datasets_dir}")
    if datasets is None:
        datasets = discover_datasets(datasets_dir)
        print(f"Discovered {len(datasets)} datasets: {', '.join(datasets)}")
    else:
        print(f"Processing {len(datasets)} datasets: {', '.join(datasets)}")

    # Exit if no datasets found
    if not datasets:
        print("No datasets found!")
        return []

    # Handle resume functionality
    skip_until = None
    if start_from:
        skip_until = start_from
        print(f"  Resuming from: {start_from[0]} ({start_from[1]})")

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Initialize aggregation structures
    all_stats: List[Stage1Stats] = []

    # Build ticker universe for matching
    print("\nBuilding ticker universe...")
    universe = build_ticker_universe()
    print(
        f"  Loaded {len(universe.all_symbols):,} symbols, {len(universe.name_automaton):,} name patterns"
    )

    # Process each dataset
    for ds in datasets:
        # Handle resume: skip until we reach the target dataset
        if skip_until and ds != skip_until[0]:
            print(f"\n[{ds}] Skipping (resuming from {skip_until[0]})...")
            continue

        # Determine if we should skip submissions for this dataset
        skip_submissions = False
        if skip_until and ds == skip_until[0] and skip_until[1] == "comments":
            skip_submissions = True

        # Process this dataset
        stats = process_dataset(
            ds,
            universe,
            datasets_dir=datasets_dir,
            out_dir=out_dir,
            progress_every=progress_every,
            force=force or (skip_until is not None),  # Force when resuming
            skip_submissions=skip_submissions,
            tickers_only=tickers_only,
        )

        # Skip if dataset was skipped (already processed)
        if stats is None:
            continue

        all_stats.append(stats)

        # Print dataset summary
        print(f"\n[{ds}] Done in {format_duration(stats.runtime_seconds)}")
        print(
            f"  Submissions: {stats.matched_submissions:,}/{stats.total_submissions:,}"
        )
        print(f"  Comments: {stats.matched_comments:,}/{stats.total_comments:,}")

        # Clear skip_until after first processed dataset
        skip_until = None

    # --- Summary ---
    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETE")
    print("=" * 70)

    if all_stats:
        total_runtime = sum(s.runtime_seconds for s in all_stats)
        total_subs = sum(s.matched_submissions for s in all_stats)
        total_coms = sum(s.matched_comments for s in all_stats)

        print(f"Total runtime: {format_duration(total_runtime)}")
        print(f"Datasets processed: {len(all_stats)}")
        print(f"Total submissions with mentions: {total_subs:,}")
        print(f"Total comments with mentions: {total_coms:,}")
    else:
        print("No datasets were processed (all skipped or none found)")

    return all_stats
