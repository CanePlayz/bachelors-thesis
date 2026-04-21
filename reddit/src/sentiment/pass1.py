"""Pass 1: Score mentions and cache to LMDB.

This module implements Pass 1 of the Stage 2 sentiment scoring pipeline.
It streams through Stage 1 mention files, scores texts with HuggingFace
models, and caches results to LMDB for fast lookup in Pass 2.

Architecture:
    Stream Stage 1 outputs → batch texts → check LMDB cache →
    score uncached with GPU → write scores to LMDB

Two-tier caching:
    - LMDB (score_store): Fast key-value store for this pipeline run.
      Checked first via has_score() to skip already-processed items.
    - SQLite (SentimentCache): Persistent cache in pipeline/caches/ that
      survives across runs. Checked for texts not in LMDB.

Key functions:
    - score_mentions_streaming(): Main Pass 1 entry point
    - _score_batch_nontargeted(): Score texts with non-targeted models
    - _score_batch_targeted(): Score (text, target) pairs with targeted models
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

from common.io_utils import StreamProgress, iter_zst_ndjson_with_progress
from data.config import MAX_TEXT_LEN, SENTIMENT_MODELS
from helpers.logging_utils import format_duration, indent, print_banner

from .cache import SentimentCache
from .lmdb_store import LMDBScoreStore
from .model_loader import (
    dispose_model,
    get_device,
    load_sentiment_model,
    score_batch_internal,
    score_batch_targeted_internal,
)
from .progress import format_pass1_file_complete, format_pass1_line, print_pass1_header
from .utils import (
    get_model_key,
    get_total_bytes,
    iter_stage1_files,
    load_global_match_counts,
)


def _score_batch_nontargeted(
    texts: List[str],
    model_name: str,
    model_key: str,
    clf,
    cache: SentimentCache,
    score_store: LMDBScoreStore,
    batch_size: int,
) -> Tuple[int, int, int]:
    """Score a batch of texts with a non-targeted model.

    Processes a batch of texts through the sentiment model, utilizing
    both LMDB (run-local) and SQLite (persistent) caches to avoid
    redundant GPU inference.

    Args:
        texts: List of text strings to score
        model_name: Full HuggingFace model name
        model_key: Short key for this model
        clf: Loaded HuggingFace pipeline
        cache: SQLite persistent cache
        score_store: LMDB run-local cache
        batch_size: Batch size for GPU inference

    Returns:
        Tuple of (items_written_to_lmdb, sqlite_cache_hits, gpu_scored)
    """
    if not texts:
        return 0, 0, 0

    # Deduplicate within batch
    unique_texts = list(set(texts))

    # Check persistent SQLite cache first
    to_score = []
    cached_scores = {}
    sqlite_hits = 0
    for text in unique_texts:
        cached = cache.get(text)
        if cached:
            cached_scores[text] = (cached.score, cached.label)
            sqlite_hits += 1
        else:
            to_score.append(text)

    # Score uncached texts using the model_loader function (GPU inference)
    gpu_scored = len(to_score)
    if to_score:
        results = score_batch_internal(clf, to_score, batch_size)
        for text, (score, label) in zip(to_score, results):
            cache.put(text, score, label)
            cached_scores[text] = (score, label)

    # Write all scores to LMDB
    items = [
        (text, model_key, score, label)
        for text, (score, label) in cached_scores.items()
    ]
    score_store.add_non_targeted(items)

    return len(items), sqlite_hits, gpu_scored


def _score_batch_targeted(
    pairs: List[Tuple[str, str]],
    model_name: str,
    model_key: str,
    clf,
    cache: SentimentCache,
    score_store: LMDBScoreStore,
    batch_size: int,
) -> Tuple[int, int, int]:
    """Score a batch of (text, target) pairs with a targeted model.

    Targeted models (like topic-sentiment) score sentiment toward a
    specific target entity mentioned in the text.

    Args:
        pairs: List of (text, target) tuples to score
        model_name: Full HuggingFace model name
        model_key: Short key for this model
        clf: Loaded HuggingFace pipeline
        cache: SQLite persistent cache
        score_store: LMDB run-local cache
        batch_size: Batch size for GPU inference

    Returns:
        Tuple of (items_written_to_lmdb, sqlite_cache_hits, gpu_scored)
    """
    if not pairs:
        return 0, 0, 0

    # Deduplicate within batch
    unique_pairs = list(set(pairs))

    # Check persistent SQLite cache first
    to_score = []
    cached_scores = {}
    sqlite_hits = 0
    for text, target in unique_pairs:
        cached = cache.get(text, target)
        if cached:
            cached_scores[(text, target)] = (cached.score, cached.label)
            sqlite_hits += 1
        else:
            to_score.append((text, target))

    # Score uncached pairs using the model_loader function (GPU inference)
    gpu_scored = len(to_score)
    if to_score:
        results = score_batch_targeted_internal(clf, to_score, batch_size)
        for (text, target), (score, label) in zip(to_score, results):
            cache.put(text, score, label, target)
            cached_scores[(text, target)] = (score, label)

    # Write all scores to LMDB
    items = [
        (text, model_key, target, score, label)
        for (text, target), (score, label) in cached_scores.items()
    ]
    score_store.add_targeted(items)

    return len(items), sqlite_hits, gpu_scored


def score_mentions_streaming(
    datasets: List[str],
    stage1_dir: str,
    batch_size: int,
    score_store: LMDBScoreStore,
) -> int:
    """Stream mentions from Stage 1, score with all models, cache to LMDB.

    This is Pass 1 of the streamlined architecture. It:
    1. Streams through Stage 1 mention files
    2. Batches texts for GPU efficiency
    3. Checks LMDB cache (fast, run-local) before scoring
    4. Scores uncached texts with each model (also checks SQLite persistent cache)
    5. Writes scores to LMDB for Pass 2 lookups

    Deduplication happens naturally via LMDB's idempotent put operations.

    Args:
        datasets: List of dataset names to process
        stage1_dir: Path to stage1_mentions directory
        batch_size: Batch size for GPU scoring
        score_store: LMDB store for scores (checked first for each item)

    Returns:
        Total mentions processed
    """
    print_banner("PASS 1: Scoring mentions (streaming to LMDB)", prefix=indent(1))

    # Calculate total bytes for progress tracking
    total_bytes = get_total_bytes(datasets, stage1_dir)
    progress_interval = 50000  # Update progress every N mentions
    total_mentions = 0

    # Load global totals from stats pass (for better ETA when caching is heavy)
    stats_total_mentions, stats_total_target_pairs = load_global_match_counts(
        stage1_dir
    )

    # Process each model separately (reuses same Stage 1 files)
    for model_idx, (model_name, is_targeted) in enumerate(SENTIMENT_MODELS, 1):
        model_key = get_model_key(model_name)

        # Print model info
        print(
            f"\n{indent(1)}--- Model {model_idx}/{len(SENTIMENT_MODELS)}: {model_key} ---"
        )
        print(f"{indent(1)}Full name: {model_name}")
        print(f"{indent(1)}Targeted: {is_targeted}")
        print(
            f"{indent(1)}Device: {'GPU' if get_device() >= 0 else 'CPU'}, Batch size: {batch_size}"
        )

        # Load SQLite persistent cache (in pipeline/caches/<model>.sqlite)
        cache = SentimentCache(model_name)
        cache_stats = cache.stats()
        print(f"{indent(1)}Persistent cache: {cache_stats['total_entries']:,} entries")

        # Load HuggingFace model pipeline
        load_start = time.time()
        clf = load_sentiment_model(model_name)
        print(f"{indent(1)}Model loaded in {format_duration(time.time() - load_start)}")

        print_pass1_header(prefix=indent(1))

        # Track metrics for this model
        model_mentions = 0  # Total records processed
        model_scored = 0  # Items written to LMDB this run
        model_gpu_scored = 0  # Items actually processed by GPU (for rate calc)
        model_lmdb_hits = 0  # Items already in LMDB (skipped entirely)
        model_sqlite_hits = 0  # Items from SQLite persistent cache
        bytes_read = 0  # Bytes processed (for progress %)
        scoring_start: Optional[float] = None  # Set when first GPU scoring happens
        batch_texts: List[str] = []  # Non-targeted: texts to score
        batch_targets: List[Tuple[str, str]] = []  # Targeted: (text, target) pairs

        # Stream through all Stage 1 files
        for ds, file_type, file_path in iter_stage1_files(datasets, stage1_dir):
            # Get file size for progress tracking
            file_size = os.path.getsize(file_path)
            progress = StreamProgress(total_bytes=file_size)

            # Stream records from compressed JSONL
            for record in iter_zst_ndjson_with_progress(file_path, progress):
                # Extract and truncate text to respect max model input length
                text = str(record.get("text", ""))[:MAX_TEXT_LEN]
                if not text.strip():
                    continue

                # Get matched tickers for this record from Stage 1
                matched_on = record.get("matched_on", {})

                if is_targeted:
                    # Targeted model: score each (text, target) pair separately
                    targets = {match_str for _, match_str in matched_on.items()}
                    for target in targets:
                        # Check LMDB first (fast lookup by hashed key)
                        if score_store.has_score(text, model_key, target):
                            model_lmdb_hits += 1
                        else:
                            # Queue for scoring (will check SQLite cache in batch func)
                            batch_targets.append((text, target))
                else:
                    # Non-targeted model: score text once
                    if score_store.has_score(text, model_key, None):
                        model_lmdb_hits += 1
                    else:
                        batch_texts.append(text)

                # Update counters
                model_mentions += 1

                # Score batch when full (non-targeted)
                if len(batch_texts) >= batch_size:
                    scored, sqlite_hits, gpu_scored = _score_batch_nontargeted(
                        batch_texts,
                        model_name,
                        model_key,
                        clf,
                        cache,
                        score_store,
                        batch_size,
                    )
                    model_scored += scored
                    model_sqlite_hits += sqlite_hits
                    model_gpu_scored += gpu_scored
                    if gpu_scored > 0 and scoring_start is None:
                        scoring_start = time.time()  # First actual GPU work
                    batch_texts.clear()

                # Score batch when full (targeted)
                if len(batch_targets) >= batch_size:
                    scored, sqlite_hits, gpu_scored = _score_batch_targeted(
                        batch_targets,
                        model_name,
                        model_key,
                        clf,
                        cache,
                        score_store,
                        batch_size,
                    )
                    model_scored += scored
                    model_sqlite_hits += sqlite_hits
                    model_gpu_scored += gpu_scored
                    if gpu_scored > 0 and scoring_start is None:
                        scoring_start = time.time()  # First actual GPU work
                    batch_targets.clear()

                # Progress update every N mentions
                if model_mentions % progress_interval == 0:
                    current_bytes = bytes_read + progress.bytes_read
                    overall_pct = (
                        100.0 * current_bytes / total_bytes if total_bytes > 0 else 0
                    )
                    # Rate based on GPU scoring speed (best ETA predictor)
                    if scoring_start is not None and model_gpu_scored > 0:
                        scoring_elapsed = time.time() - scoring_start
                        rate = (
                            model_gpu_scored / scoring_elapsed
                            if scoring_elapsed > 0
                            else 0
                        )
                        # ETA based on remaining items to potentially score on GPU.
                        # Use stats totals as upper bound, subtract cached items.
                        expected_total = (
                            stats_total_target_pairs
                            if is_targeted
                            else stats_total_mentions
                        )
                        if expected_total:
                            completed = (
                                model_lmdb_hits + model_sqlite_hits + model_gpu_scored
                            )
                            remaining = max(int(expected_total) - int(completed), 0)
                            eta = remaining / rate if rate > 0 else 0
                        else:
                            eta = 0
                    else:
                        rate = 0
                        eta = 0

                    print(
                        format_pass1_line(
                            model_mentions,
                            model_gpu_scored,
                            model_sqlite_hits,
                            rate,
                            overall_pct,
                            eta,
                            prefix=indent(1),
                        )
                    )

            # File complete - update bytes read
            bytes_read += file_size
            print(
                format_pass1_file_complete(
                    ds, file_type, model_mentions, prefix=indent(1)
                )
            )

        # Score remaining items in partial batches
        if batch_texts:
            scored, sqlite_hits, gpu_scored = _score_batch_nontargeted(
                batch_texts, model_name, model_key, clf, cache, score_store, batch_size
            )
            model_scored += scored
            model_sqlite_hits += sqlite_hits
            model_gpu_scored += gpu_scored

        if batch_targets:
            scored, sqlite_hits, gpu_scored = _score_batch_targeted(
                batch_targets,
                model_name,
                model_key,
                clf,
                cache,
                score_store,
                batch_size,
            )
            model_scored += scored
            model_sqlite_hits += sqlite_hits
            model_gpu_scored += gpu_scored

        # Flush LMDB to disk, close SQLite, free GPU memory
        score_store.sync()
        cache.close()
        dispose_model(clf)

        print(
            f"\n{indent(1)}✓ Model complete: {model_gpu_scored:,} GPU scored, "
            f"{model_sqlite_hits:,} persistent cache hits, {model_lmdb_hits:,} run cache hits"
        )

        # All models process same mentions, so final count is same
        total_mentions = model_mentions

    print(f"\n{indent(1)}Pass 1 complete: {total_mentions:,} mentions processed")
    return total_mentions
