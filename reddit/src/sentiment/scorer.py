"""Stage 2: Sentiment scoring for ticker mentions.

Architecture (v2 - Streamlined)
-------------------------------
Pass 1 (Score): Stream Stage 1 outputs → batch texts → check LMDB cache →
                score uncached with GPU → write scores to LMDB

Pass 2 (Apply): Stream Stage 1 outputs → lookup scores from LMDB →
                write Stage 2 outputs with sentiment fields

This eliminates the 2+ hour text collection pass from v1. The ~10% deduplication
ratio didn't justify the I/O cost. Instead, we stream directly and leverage
LMDB's ~10x faster lookups (vs SQLite) to minimize cache check overhead.

Module structure:
- scorer.py: Entry point (run_stage2) and exports
- utils.py: Shared utilities (model keys, dataset discovery)
- pass1.py: GPU scoring and LMDB caching
- pass2.py: Score application and output writing

Key functions:
- run_stage2(): Main entry point
- score_mentions_streaming(): Pass 1 - stream mentions, score, cache to LMDB
- apply_scores_streaming(): Pass 2 - stream mentions, lookup scores, write output
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Set

from data.config import (
    DEFAULT_BATCH_SIZE,
    SENTIMENT_MODELS,
    STAGE1_OUT_DIR,
    STAGE2_OUT_DIR,
)
from helpers.logging_utils import format_duration, indent, print_banner

from .lmdb_store import LMDBScoreStore
from .pass1 import score_mentions_streaming
from .pass2 import apply_scores_streaming
from .utils import discover_stage1_datasets, get_model_key

# Re-export commonly used functions for backwards compatibility
__all__ = [
    "run_stage2",
    "get_model_key",
    "discover_stage1_datasets",
    "score_mentions_streaming",
    "apply_scores_streaming",
]


def run_stage2(
    datasets: Optional[List[str]] = None,
    stage1_dir: str = STAGE1_OUT_DIR,
    stage2_dir: str = STAGE2_OUT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    low_memory: bool = False,
    low_memory_db_dir: Optional[str] = None,
    low_memory_reuse: bool = True,
    state: Optional[Any] = None,
    state_dir: Optional[str] = None,
) -> Dict[str, int]:
    """Run Stage 2: Sentiment scoring.

    Orchestrates the two-pass sentiment scoring pipeline:
    1. Pass 1: Stream mentions, score with GPU, cache to LMDB
    2. Pass 2: Stream mentions, lookup scores, write enriched outputs

    Args:
        datasets: List of dataset names to process. If None, auto-discover
                  from stage1_dir.
        stage1_dir: Path to Stage 1 outputs (stage1_mentions directory)
        stage2_dir: Path for Stage 2 outputs (stage2_sentiment directory)
        batch_size: Batch size for GPU inference (default from config)
        force: If True, reprocess even if outputs exist
        low_memory: Kept for API compatibility (always uses LMDB now)
        low_memory_db_dir: Optional override for LMDB storage location
        low_memory_reuse: If True and Pass 1 completed, skip to Pass 2
        state: Optional pre-loaded pipeline state dict
        state_dir: Directory for pipeline state file

    Returns:
        Dict mapping dataset name to number of records processed

    Note:
        The new architecture always uses LMDB-based streaming.
        The low_memory flags are kept for API compatibility.
    """
    print_banner("STAGE 2: SENTIMENT SCORING")
    print("Architecture: Streaming with LMDB cache")

    start_time = time.time()

    # Discover datasets
    if datasets is None:
        datasets = discover_stage1_datasets(stage1_dir)
    if not datasets:
        print("No Stage 1 outputs found!")
        return {}

    print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    print(f"Using {len(SENTIMENT_MODELS)} sentiment models")

    # Filter to datasets needing processing
    datasets_to_process = []
    for ds in datasets:
        out_dir = os.path.join(stage2_dir, ds)
        has_output = os.path.exists(
            os.path.join(out_dir, "s2_submission_mentions.jsonl.zst")
        ) or os.path.exists(os.path.join(out_dir, "s2_comment_mentions.jsonl.zst"))
        if force or not has_output:
            datasets_to_process.append(ds)
        else:
            print(f"[{ds}] Skipping (outputs exist, use --force to reprocess)")

    if not datasets_to_process:
        print("All datasets already processed!")
        return {}

    # Prepare model metadata
    model_keys = [get_model_key(name) for name, _ in SENTIMENT_MODELS]
    targeted_models = {
        get_model_key(name) for name, is_targeted in SENTIMENT_MODELS if is_targeted
    }

    # Setup LMDB store in unified location
    # Uses RUN_SCORES_DIR from config (~/reddit_sentiment_lmdb/run_scores/)
    from data.config import RUN_SCORES_DIR

    db_dir = low_memory_db_dir or RUN_SCORES_DIR
    lmdb_path = os.path.join(db_dir, "scores.lmdb")

    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    # Import state functions
    from state import Check, Mark, load_state, save_state

    # Load state
    if state is None:
        state_dir = state_dir or os.path.dirname(stage2_dir)
        state = load_state(state_dir)
    else:
        state_dir = state_dir or os.path.dirname(stage2_dir)

    # Check pass validity - but DON'T delete LMDB on invalid state
    # LMDB is persistent and has_score() will handle resumption of interrupted runs
    pass1_check = Check.lowmem_pass1(state, lmdb_path)

    if not pass1_check.valid:
        print(f"Pass 1 state: {pass1_check.reason}")
        # Don't delete LMDB - it may contain partial progress we can resume from
        # The has_score() checks in score_mentions_streaming handle deduplication

    results: Dict[str, int] = {}

    with LMDBScoreStore(lmdb_path) as score_store:
        # Check existing LMDB entries for resumption info
        existing_count = score_store.count()
        if existing_count > 0 and not pass1_check.valid:
            print(f"Resuming with {existing_count:,} existing LMDB entries")

        # Pass 1: Score mentions
        if low_memory_reuse and pass1_check.valid:
            print_banner(
                f"PASS 1: Reusing LMDB cache ({existing_count:,} scores)",
                prefix=indent(1),
            )
        else:
            Mark.lowmem_pass1_started(state, stage1_dir, datasets_to_process)
            save_state(state_dir, state)

            score_mentions_streaming(
                datasets_to_process,
                stage1_dir,
                batch_size,
                score_store,
            )

            Mark.lowmem_pass1_complete(state, lmdb_path)
            save_state(state_dir, state)

        # Pass 2: Apply scores and write outputs
        results = apply_scores_streaming(
            datasets_to_process,
            stage1_dir,
            stage2_dir,
            score_store,
            model_keys,
            targeted_models,
        )

    # Record stage2 completion
    Mark.stage2_complete(state, stage2_dir, datasets_to_process)
    save_state(state_dir, state)

    # Summary
    print_banner("STAGE 2 COMPLETE")
    print(f"Total time: {format_duration(time.time() - start_time)}")
    print(f"Datasets processed: {len(results)}")
    print(f"Total mentions scored: {sum(results.values()):,}")

    return results
