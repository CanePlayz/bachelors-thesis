"""CLI entry point for v2 pipeline (src).

This module provides the command-line interface for running the ticker mention
extraction and sentiment analysis pipeline. The pipeline has three stages:

Stage 1: Ticker Mention Extraction
    Streams through Reddit archives and extracts ticker mentions.
    Outputs: stage1_mentions/{dataset}/s1_*.jsonl.zst

Stage 2: Sentiment Scoring
    Scores all extracted mentions with multiple sentiment models.
    Uses GPU acceleration and caching for efficiency.
    Outputs: stage2_sentiment/{dataset}/s2_*.jsonl.zst

Stage 3: Daily Aggregation
    Aggregates sentiment by day, ticker, and model.
    Computes weighted average and dispersion.
    Outputs: stage3_aggregated/daily_sentiment.csv

Re-run Logic:
    By default, stages are skipped if their outputs already exist.
    - No flags: auto-discover which stages need running, run from first missing
    - --force: re-run all stages from the beginning
    - --stage N: start discovery from stage N (skip earlier stages)
    - --stage N --force: re-run from stage N onwards (requires stage N-1 outputs)

Usage:
    python main.py                    # Auto-discover and run needed stages
    python main.py --stage 2          # Check from Stage 2 onwards
    python main.py --force            # Re-run everything
    python main.py --stage 2 --force  # Force re-run from Stage 2
    python main.py investing stocks   # Process specific datasets
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path so absolute imports like `from common.io_utils import ...`
# work when running this script directly (`python main.py`).
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import time
from typing import List, Optional, Tuple

# Import configuration
from data.config import (
    DATASETS_DIR,
    DEFAULT_BATCH_SIZE,
    STAGE1_OUT_DIR,
    STAGE2_OUT_DIR,
    STAGE3_OUT_DIR,
    STATS_OUT_DIR,
)

# Import pipeline stages
from extraction.pipeline import run_stage1

# Import helpers
from helpers.logging_utils import format_duration
from sentiment import run_stage2, run_stage3

# Import unified state management
from state import (
    Check,
    Fingerprints,
    Mark,
    PipelineState,
    clear_state,
    load_state,
    save_state,
)

# Import unified stats runner
from stats.runner import run_stats_pass

# -----------------------------------------------------------------------------
# Pipeline State Directory
# -----------------------------------------------------------------------------

# State file lives in the pipeline output root (parent of stage dirs)
STATE_DIR = os.path.dirname(STAGE1_OUT_DIR)


# -----------------------------------------------------------------------------
# Stage Output Discovery Functions (using unified fingerprint system)
# -----------------------------------------------------------------------------


def discover_raw_datasets(datasets_dir: str) -> List[str]:
    """Discover datasets from raw .zst files in the datasets directory."""
    if not os.path.isdir(datasets_dir):
        return []

    found: set[str] = set()
    for fname in os.listdir(datasets_dir):
        if fname.endswith("_submissions.zst"):
            found.add(fname[: -len("_submissions.zst")])
        elif fname.endswith("_comments.zst"):
            found.add(fname[: -len("_comments.zst")])

    return sorted(found)


def stage1_has_outputs(
    state: PipelineState, datasets: Optional[List[str]] = None
) -> bool:
    """Check if Stage 1 outputs are valid using fingerprint system.

    A stage is VALID if ALL three conditions hold:
        1. output_fp is not None (proves completion)
        2. input_fp == expected_input_fp (proves correct inputs)
        3. output_fp == current_fingerprint(outputs) (proves unchanged)
    """
    result = Check.stage1(state, DATASETS_DIR, STAGE1_OUT_DIR)
    if not result.valid:
        return False

    # If specific datasets requested, also verify those outputs exist
    if datasets:
        for ds in datasets:
            ds_out = os.path.join(STAGE1_OUT_DIR, ds)
            if not os.path.isdir(ds_out):
                return False
            # Check at least one output file exists
            has_any = any(
                os.path.exists(os.path.join(ds_out, f))
                for f in [
                    "s1_submission_mentions.jsonl.zst",
                    "s1_comment_mentions.jsonl.zst",
                ]
            )
            if not has_any:
                return False

    return True


def stage2_has_outputs(
    state: PipelineState, datasets: Optional[List[str]] = None
) -> bool:
    """Check if Stage 2 outputs are valid using fingerprint system.

    A stage is VALID if ALL three conditions hold:
        1. output_fp is not None (proves completion)
        2. input_fp == expected_input_fp (proves correct inputs)
        3. output_fp == current_fingerprint(outputs) (proves unchanged)
    """
    s2_result = Check.stage2(state, STAGE2_OUT_DIR, datasets)
    if not s2_result.valid:
        return False

    # If specific datasets requested, also verify those outputs exist
    if datasets:
        for ds in datasets:
            ds_out = os.path.join(STAGE2_OUT_DIR, ds)
            if not os.path.isdir(ds_out):
                return False
            # Check at least one output file exists
            has_any = any(
                os.path.exists(os.path.join(ds_out, f))
                for f in [
                    "s2_submission_mentions.jsonl.zst",
                    "s2_comment_mentions.jsonl.zst",
                ]
            )
            if not has_any:
                return False

    return True


def stage3_has_outputs(state: PipelineState) -> bool:
    """Check if Stage 3 outputs are valid using fingerprint system.

    A stage is VALID if ALL three conditions hold:
        1. output_fp is not None (proves completion)
        2. input_fp == expected_input_fp (proves correct inputs)
        3. output_fp == current_fingerprint(outputs) (proves unchanged)
    """
    s3_result = Check.stage3(state, STAGE2_OUT_DIR, STAGE3_OUT_DIR)
    return s3_result.valid


def datasets_missing_stage1_outputs(
    state: PipelineState, datasets: List[str]
) -> List[str]:
    """Return datasets whose Stage 1 outputs are missing/incomplete."""
    # If stage1 is globally invalid, all datasets need processing
    if not Check.stage1(state, DATASETS_DIR, STAGE1_OUT_DIR).valid:
        return datasets

    # Otherwise check individual datasets for missing files
    missing: List[str] = []
    for ds in datasets:
        ds_out = os.path.join(STAGE1_OUT_DIR, ds)
        if not os.path.isdir(ds_out):
            missing.append(ds)
            continue

        # Check expected outputs based on raw inputs
        expected_exists = False
        if os.path.exists(os.path.join(DATASETS_DIR, f"{ds}_submissions.zst")):
            if not os.path.exists(
                os.path.join(ds_out, "s1_submission_mentions.jsonl.zst")
            ):
                missing.append(ds)
                continue
            expected_exists = True
        if os.path.exists(os.path.join(DATASETS_DIR, f"{ds}_comments.zst")):
            if not os.path.exists(
                os.path.join(ds_out, "s1_comment_mentions.jsonl.zst")
            ):
                missing.append(ds)
                continue
            expected_exists = True

        if not expected_exists:
            missing.append(ds)

    return missing


def datasets_missing_stage2_outputs(
    state: PipelineState, datasets: List[str]
) -> List[str]:
    """Return datasets whose Stage 2 outputs are missing or stale."""
    # If stage2 is globally invalid, all datasets need processing
    if not Check.stage2(state, STAGE2_OUT_DIR, datasets).valid:
        return datasets

    # Otherwise check individual datasets for missing files
    missing: List[str] = []
    for ds in datasets:
        ds_out = os.path.join(STAGE2_OUT_DIR, ds)
        if not os.path.isdir(ds_out):
            missing.append(ds)
            continue

        # Check at least one output file exists
        has_any = any(
            os.path.exists(os.path.join(ds_out, f))
            for f in [
                "s2_submission_mentions.jsonl.zst",
                "s2_comment_mentions.jsonl.zst",
            ]
        )
        if not has_any:
            missing.append(ds)

    return missing


def discover_first_missing_stage(
    state: PipelineState,
    datasets: Optional[List[str]] = None,
) -> int:
    """Discover which stage needs to run first.

    Checks stages 1, 2, 3 in order using fingerprint validation.
    Returns the first stage that is invalid or missing outputs.

    Args:
        state: Current pipeline state
        datasets: Optional list of datasets to check

    Returns:
        Stage number (1, 2, or 3), or 4 if all stages complete
    """
    if not stage1_has_outputs(state, datasets):
        return 1
    if not stage2_has_outputs(state, datasets):
        return 2
    if not stage3_has_outputs(state):
        return 3
    return 4  # All stages complete


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Parses command-line arguments and runs the pipeline stages
    based on the provided flags.

    Args:
        argv: Optional list of command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    # Set up argument parser with description and examples
    parser = argparse.ArgumentParser(
        description="Reddit Ticker Mention Extraction & Sentiment Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                          # Auto-discover and run needed stages
    python main.py --stage 1                # Check from Stage 1, run missing onwards
    python main.py --stage 2                # Check from Stage 2, run missing onwards
    python main.py --force                  # Re-run all stages (1, 2, 3)
    python main.py --stage 2 --force        # Force re-run stages 2 and 3
    python main.py investing stocks         # Process specific datasets only
    python main.py --batch-size 64          # Use larger batch size for GPU scoring
    python main.py --stats-plots-only       # Regenerate stats and plots only
    python main.py --low-memory             # Use disk storage for texts/scores (saves RAM)
    python main.py --continue-stage1 wallstreetbets comments  # Resume Stage 1
    python main.py --stage1-only            # Run only Stage 1 (for filtering experiments)
    python main.py --tickers-only           # Skip company name/alias matching (robustness check)

Re-run Logic:
    - No flags: auto-detect first missing stage, run from there onwards
    - --force: re-run everything from Stage 1
    - --stage N: start checking from Stage N, run from first missing onwards
    - --stage N --force: re-run from Stage N (requires Stage N-1 outputs)

Stages:
    1. Ticker Mention Extraction - Find stock mentions in Reddit posts
    2. Sentiment Scoring - Score mentions with multiple ML models (GPU)
    3. Daily Aggregation - Aggregate sentiment by day/stock/model
        """,
    )

    # Positional argument: list of datasets to process
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names to process (default: auto-discover)",
    )

    # Stage selection
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=None,
        metavar="N",
        help="Start discovery from stage N (with --force: re-run from N)",
    )

    # Force reprocessing
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even if outputs exist",
    )

    # Batch size for sentiment scoring
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help=f"Batch size for GPU sentiment scoring (default: {DEFAULT_BATCH_SIZE})",
    )

    # Continue Stage 1 from a specific dataset/type
    parser.add_argument(
        "--continue-stage1",
        nargs=2,
        metavar=("DATASET", "TYPE"),
        help=(
            "Resume Stage 1 from a specific dataset and type. "
            "TYPE must be 'submissions' or 'comments'. "
            "Example: --continue-stage1 wallstreetbets comments"
        ),
    )

    # Stats/plots only mode
    parser.add_argument(
        "--stats-plots-only",
        action="store_true",
        help="Only regenerate stats and plots from existing Stage 1 outputs (skip pipeline)",
    )

    # Low-memory mode for Stage 2
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Use disk-based storage for unique texts in Stage 2 (reduces RAM, slower)",
    )

    # Stage 1 only mode (for extraction/filtering experiments)
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run only Stage 1 (ticker extraction), skip sentiment and aggregation",
    )

    # Tickers-only mode (skip company name and alias matching)
    parser.add_argument(
        "--tickers-only",
        action="store_true",
        help="Only match $-prefixed and bare caps tickers (skip company names/aliases)",
    )

    # Parse the command-line arguments
    args = parser.parse_args(argv)

    # Get datasets list (None means auto-discover)
    datasets: Optional[List[str]] = args.datasets if args.datasets else None

    # Parse --continue-stage1 if provided
    start_from: Optional[Tuple[str, str]] = None
    if args.continue_stage1:
        resume_dataset, resume_type = args.continue_stage1
        if resume_type not in ("submissions", "comments"):
            print(
                f"ERROR: --continue-stage1 TYPE must be 'submissions' or 'comments', got '{resume_type}'"
            )
            return 1
        start_from = (resume_dataset, resume_type)

    # Print pipeline banner
    print("=" * 70)
    print("REDDIT TICKER MENTION & SENTIMENT PIPELINE")
    print("=" * 70)

    # Load pipeline state for fingerprint-based validation
    state = load_state(STATE_DIR)

    # -------------------------------------------------------------------------
    # Handle --stats-plots-only mode
    # -------------------------------------------------------------------------
    if args.stats_plots_only:
        print("Stats/plots mode: regenerating from Stage 1 outputs")
        print()

        run_stats_pass(
            stage1_dir=STAGE1_OUT_DIR,
            stats_dir=STATS_OUT_DIR,
            datasets=datasets,
        )
        return 0

    # -------------------------------------------------------------------------
    # Determine which stages to run based on --stage and --force flags
    # -------------------------------------------------------------------------
    #
    # Logic:
    # 1. No flags: auto-discover first missing stage, run from there onwards
    # 2. --force only: run all stages (1, 2, 3)
    # 3. --stage N only: check if stage N has outputs
    #    - If outputs exist, check N+1, etc.
    #    - If no outputs, run from N onwards
    # 4. --stage N --force: require stage N-1 outputs exist (unless N=1),
    #    then run from N onwards without checking
    # -------------------------------------------------------------------------

    # Handle --stage1-only mode
    if args.stage1_only:
        run_stages = [1]
        print("Stage 1 only mode: will run extraction only (no sentiment/aggregation)")
        if args.force:
            print("  (with --force: reprocessing even if outputs exist)")
            clear_state(STATE_DIR)
            state = PipelineState()

    elif args.force and args.stage is None:
        # --force without --stage: run everything from stage 1
        start_stage = 1
        run_stages = [1, 2, 3]
        print("Force mode: will reprocess all stages")
        # Clear state for fresh start
        clear_state(STATE_DIR)
        state = PipelineState()

    elif args.force and args.stage is not None:
        # --stage N --force: require prior stage outputs, run from N onwards
        start_stage = args.stage

        # Validate: prior stage must have outputs (unless starting at 1)
        if start_stage == 2 and not stage1_has_outputs(state, datasets):
            print("ERROR: --stage 2 --force requires Stage 1 outputs to exist")
            print(f"  Stage 1 output dir: {STAGE1_OUT_DIR}")
            return 1
        if start_stage == 3 and not stage2_has_outputs(state, datasets):
            print("ERROR: --stage 3 --force requires Stage 2 outputs to exist")
            print(f"  Stage 2 output dir: {STAGE2_OUT_DIR}")
            return 1

        run_stages = list(range(start_stage, 4))  # e.g., [2, 3] or [3]
        print(f"Force mode: reprocessing from Stage {start_stage} onwards")

    elif args.stage is not None:
        # --stage N without --force: start discovery from stage N
        start_stage = args.stage
        run_stages = []

        # Check each stage from N onwards
        for stage in range(start_stage, 4):
            if stage == 1:
                has_output = stage1_has_outputs(state, datasets)
            elif stage == 2:
                has_output = stage2_has_outputs(state, datasets)
            else:  # stage == 3
                has_output = stage3_has_outputs(state)

            if has_output:
                print(f"Stage {stage}: outputs valid, skipping")
            else:
                # No valid output - run this stage and all remaining stages
                run_stages = list(range(stage, 4))
                print(
                    f"Stage {stage}: outputs missing/stale, will run stages {run_stages}"
                )
                break

        if not run_stages:
            print("All stages have valid outputs! Use --force to reprocess.")
            return 0

    else:
        # No flags: auto-discover first missing stage
        first_missing = discover_first_missing_stage(state, datasets)

        if first_missing > 3:
            print("All stages have valid outputs! Use --force to reprocess.")
            return 0

        run_stages = list(range(first_missing, 4))
        print(f"Auto-discovered: Stage {first_missing} is first invalid/missing")
        print(f"Will run stages: {run_stages}")

    pipeline_start = time.time()

    # --- Stage 1: Ticker Mention Extraction ---
    if 1 in run_stages:
        stage1_candidates = datasets or discover_raw_datasets(DATASETS_DIR)
        if not stage1_candidates:
            print("No datasets found in Datasets/. Nothing to do.")
            return 1

        stage1_datasets = (
            stage1_candidates
            if args.force
            else datasets_missing_stage1_outputs(state, stage1_candidates)
        )

        if not stage1_datasets:
            print("Stage 1: all datasets already have complete outputs, skipping")
        else:
            # Mark stage started (records input fingerprint)
            Mark.stage1_started(state, DATASETS_DIR)
            save_state(STATE_DIR, state)

            print(
                f"Stage 1: processing {len(stage1_datasets)} missing dataset(s): {', '.join(stage1_datasets)}"
            )
            if args.tickers_only:
                print("  (tickers-only mode: skipping company name and alias matching)")
            run_stage1(
                datasets=stage1_datasets,
                out_dir=STAGE1_OUT_DIR,
                progress_every=50000,
                start_from=start_from,
                force=args.force,
                tickers_only=args.tickers_only,
            )

            # Mark stage complete (records output fingerprint)
            Mark.stage1_complete(state, STAGE1_OUT_DIR)
            save_state(STATE_DIR, state)

        # Run stats pass after Stage 1 (same as --stats-plots-only)
        run_stats_pass(
            stage1_dir=STAGE1_OUT_DIR,
            stats_dir=STATS_OUT_DIR,
            datasets=datasets or discover_raw_datasets(DATASETS_DIR),
        )

    # --- Stage 2: Sentiment Scoring ---
    if 2 in run_stages:
        stage2_candidates = datasets or discover_raw_datasets(DATASETS_DIR)
        if not stage2_candidates:
            print("No datasets found in Datasets/. Nothing to do.")
            return 1

        stage2_datasets = (
            stage2_candidates
            if args.force
            else datasets_missing_stage2_outputs(state, stage2_candidates)
        )

        if not stage2_datasets:
            print("Stage 2: all datasets already have valid outputs, skipping")
        else:
            # Mark stage started (records input fingerprint)
            Mark.stage2_started(state, STAGE1_OUT_DIR, stage2_datasets)
            save_state(STATE_DIR, state)

            run_stage2(
                datasets=stage2_datasets,
                stage1_dir=STAGE1_OUT_DIR,
                stage2_dir=STAGE2_OUT_DIR,
                batch_size=args.batch_size,
                force=True,
                low_memory=args.low_memory,
                # Pass state for lowmem pass tracking
                state=state,
                state_dir=STATE_DIR,
            )

            # Mark stage complete (records output fingerprint)
            Mark.stage2_complete(state, STAGE2_OUT_DIR, stage2_datasets)
            save_state(STATE_DIR, state)

    # --- Stage 3: Daily Aggregation ---
    if 3 in run_stages:
        # Mark stage started
        Mark.stage3_started(state, STAGE2_OUT_DIR)
        save_state(STATE_DIR, state)

        stage3_output_path = os.path.join(STAGE3_OUT_DIR, "daily_sentiment.csv")
        run_stage3(
            datasets=datasets,
            stage2_dir=STAGE2_OUT_DIR,
            stage3_dir=STAGE3_OUT_DIR,
            force=args.force or os.path.exists(stage3_output_path),
        )

        # Mark stage complete
        Mark.stage3_complete(state, STAGE3_OUT_DIR)
        save_state(STATE_DIR, state)

    # Final summary
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print("ALL REQUESTED STAGES COMPLETE")
    print("=" * 70)
    print(f"Stages run: {run_stages}")
    print(f"Total pipeline time: {format_duration(total_time)}")

    return 0


# Allow running as a script
if __name__ == "__main__":
    sys.exit(main())
