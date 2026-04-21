"""Pass 2: Apply scores from LMDB and write Stage 2 outputs.

This module implements Pass 2 of the Stage 2 sentiment scoring pipeline.
It streams through Stage 1 files, looks up pre-computed scores from LMDB,
and writes enriched Stage 2 output files with sentiment fields.

Architecture:
    Stream Stage 1 outputs → lookup scores from LMDB →
    write Stage 2 outputs with sentiment fields

Memory efficiency:
    Files are processed in chunks (default 50k records) to avoid memory
    exhaustion on large files like wallstreetbets_comments. The chunked
    generator pattern bounds memory to ~200MB regardless of file size.

Key functions:
    - apply_scores_streaming(): Main Pass 2 entry point
    - _write_scored_file_chunked(): Process single file in memory-efficient chunks
    - _apply_scores_to_record(): Enrich a record with sentiment scores
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from common.io_utils import iter_zst_ndjson, write_jsonl_zst
from data.config import MAX_TEXT_LEN
from helpers.logging_utils import indent, print_banner

from .lmdb_store import LMDBScoreStore
from .progress import format_pass2_line, print_pass2_header
from .utils import get_total_bytes


def _apply_scores_to_record(
    record: Dict[str, Any],
    text_scores: Dict[
        str,
        Dict[str, Tuple[Optional[float], Optional[str], Optional[Dict[str, float]]]],
    ],
    model_keys: List[str],
    targeted_models: Set[str],
) -> Dict[str, Any]:
    """Apply sentiment scores to a single record.

    Enriches a Stage 1 record with sentiment scores from all configured
    models. For targeted models, also adds per-ticker sentiment scores.

    Args:
        record: The Stage 1 record to enrich with scores
        text_scores: Pre-fetched scores for texts in this chunk, structured as
                     text -> model_key -> (score, label, targeted_dict)
        model_keys: List of model keys (for field naming)
        targeted_models: Set of model keys that are targeted

    Returns:
        The record with sentiment fields added:
        - sentiment_{model_key}: float score
        - sentiment_label_{model_key}: string label
        - sentiment_targeted_{model_key}: dict of ticker -> score (targeted only)
    """
    text = str(record.get("text", ""))[:MAX_TEXT_LEN]
    matched_on = record.get("matched_on", {})
    record_scores = text_scores.get(text, {})

    for model_key in model_keys:
        model_data = record_scores.get(model_key, (None, None, None))
        score, label, targeted_dict = model_data

        record[f"sentiment_{model_key}"] = score
        record[f"sentiment_label_{model_key}"] = label

        if model_key in targeted_models:
            ticker_scores: Dict[str, float] = {}
            if targeted_dict:
                for ticker, match_str in matched_on.items():
                    if match_str in targeted_dict:
                        ticker_scores[ticker] = targeted_dict[match_str]
            record[f"sentiment_targeted_{model_key}"] = ticker_scores

    return record


def _write_scored_file_chunked(
    in_path: str,
    out_path: str,
    score_store: LMDBScoreStore,
    model_keys: List[str],
    targeted_models: Set[str],
    chunk_size: int,
) -> int:
    """Process a Stage 1 file in chunks and write scored Stage 2 output.

    This function streams through the input file, accumulating records into
    chunks. For each chunk, it batch-fetches scores from LMDB (memory-efficient)
    and then streams the scored records to the output file.

    Memory usage is bounded by: chunk_size * avg_record_size (~50k * 4KB = 200MB)

    Args:
        in_path: Path to Stage 1 input file
        out_path: Path to Stage 2 output file
        score_store: LMDB store with cached scores
        model_keys: List of model keys (for field naming)
        targeted_models: Set of model keys that are targeted
        chunk_size: Number of records to process at a time

    Returns:
        Total number of records processed
    """

    def scored_records_generator() -> Generator[Dict[str, Any], None, None]:
        """Generator that yields scored records, processing in chunks."""
        chunk_records: List[Dict[str, Any]] = []
        chunk_texts: Set[str] = set()

        for record in iter_zst_ndjson(in_path):
            text = str(record.get("text", ""))[:MAX_TEXT_LEN]
            chunk_texts.add(text)
            chunk_records.append(record)

            # Process chunk when full
            if len(chunk_records) >= chunk_size:
                # Batch lookup scores for this chunk
                text_scores = score_store.fetch_scores_for_texts(list(chunk_texts))

                # Apply scores and yield records
                for rec in chunk_records:
                    yield _apply_scores_to_record(
                        rec, text_scores, model_keys, targeted_models
                    )

                # Clear chunk for next batch
                chunk_records.clear()
                chunk_texts.clear()

        # Process remaining records
        if chunk_records:
            text_scores = score_store.fetch_scores_for_texts(list(chunk_texts))
            for rec in chunk_records:
                yield _apply_scores_to_record(
                    rec, text_scores, model_keys, targeted_models
                )

    # Count records during write (generator consumes input)
    record_count = 0

    def counting_generator() -> Generator[Dict[str, Any], None, None]:
        nonlocal record_count
        for rec in scored_records_generator():
            record_count += 1
            yield rec

    # Write output using generator (streaming, memory-efficient)
    write_jsonl_zst(out_path, counting_generator(), level=3)

    return record_count


def apply_scores_streaming(
    datasets: List[str],
    stage1_dir: str,
    stage2_dir: str,
    score_store: LMDBScoreStore,
    model_keys: List[str],
    targeted_models: Set[str],
) -> Dict[str, int]:
    """Stream mentions, lookup scores from LMDB, write Stage 2 outputs.

    This is Pass 2 of the sentiment scoring pipeline. LMDB's memory-mapped
    I/O makes score lookups very fast. Files are processed in chunks to
    avoid memory exhaustion on large files like wallstreetbets_comments
    (millions of records).

    Args:
        datasets: List of dataset names to process
        stage1_dir: Path to stage1_mentions directory
        stage2_dir: Path to stage2_sentiment directory
        score_store: LMDB store with cached scores from Pass 1
        model_keys: List of model keys (for field naming)
        targeted_models: Set of model keys that are targeted

    Returns:
        Dict mapping dataset name to records processed
    """
    print_banner("PASS 2: Applying scores and writing output", prefix=indent(1))

    total_bytes = get_total_bytes(datasets, stage1_dir)
    start_time = time.time()
    bytes_processed = 0
    total_records = 0

    # Chunk size for memory-efficient LMDB lookups (50k texts at a time)
    CHUNK_SIZE = 50000

    print_pass2_header(prefix=indent(1))

    results: Dict[str, int] = {}

    for ds in datasets:
        ds_in = os.path.join(stage1_dir, ds)
        ds_out = os.path.join(stage2_dir, ds)
        os.makedirs(ds_out, exist_ok=True)

        ds_records = 0

        for file_type, in_name, out_name in [
            (
                "submission",
                "s1_submission_mentions.jsonl.zst",
                "s2_submission_mentions.jsonl.zst",
            ),
            (
                "comment",
                "s1_comment_mentions.jsonl.zst",
                "s2_comment_mentions.jsonl.zst",
            ),
        ]:
            in_path = os.path.join(ds_in, in_name)
            if not os.path.exists(in_path):
                continue

            out_path = os.path.join(ds_out, out_name)
            file_size = os.path.getsize(in_path)

            # Write output using chunked generator (streaming, memory-efficient)
            # The generator processes CHUNK_SIZE records at a time for LMDB lookups
            file_record_count = _write_scored_file_chunked(
                in_path,
                out_path,
                score_store,
                model_keys,
                targeted_models,
                CHUNK_SIZE,
            )

            if file_record_count == 0:
                continue

            ds_records += file_record_count
            total_records += file_record_count
            bytes_processed += file_size

            # Progress
            elapsed = time.time() - start_time
            rate = total_records / elapsed if elapsed > 0 else 0
            pct = 100.0 * bytes_processed / total_bytes if total_bytes > 0 else 0
            eta = elapsed * (100 - pct) / pct if pct > 0 else 0

            print(
                format_pass2_line(
                    total_records,
                    ds,
                    file_type,
                    file_record_count,
                    rate,
                    pct,
                    eta,
                    prefix=indent(1),
                )
            )

        results[ds] = ds_records

    print(f"\n{indent(1)}✓ Pass 2 complete: {total_records:,} records written")
    return results
