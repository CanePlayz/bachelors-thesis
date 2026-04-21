"""Shared utilities for Stage 2 sentiment scoring.

This module provides common functions used across Pass 1 (scoring) and
Pass 2 (applying scores), including:
- Model key generation for field naming
- Dataset discovery and file iteration
- Global statistics loading
"""

from __future__ import annotations

import json
import os
from typing import Generator, List, Optional, Tuple

from .cache import sanitize_model_name


def get_model_key(model_name: str) -> str:
    """Get a short key for a model name (for field naming).

    Converts full HuggingFace model names to short, consistent keys
    used in output field names like `sentiment_roberta`.

    Args:
        model_name: Full model name (e.g., "cardiffnlp/twitter-roberta-base-sentiment")

    Returns:
        Short key (e.g., "roberta")
    """
    name = model_name.split("/")[-1].lower()

    if "topic-sentiment" in name:
        return "roberta_topic"
    if "roberta" in name and "sentiment" in name:
        return "roberta"
    if "bertweet" in name:
        return "bertweet"
    if "fintwit" in name:
        return "fintwit"

    return sanitize_model_name(name)[:20]


def discover_stage1_datasets(stage1_dir: str) -> List[str]:
    """Find datasets with Stage 1 outputs.

    Scans the Stage 1 output directory for subdirectories containing
    mention files (s1_submission_mentions.jsonl.zst or s1_comment_mentions.jsonl.zst).

    Args:
        stage1_dir: Path to stage1_mentions directory

    Returns:
        Sorted list of dataset names with Stage 1 outputs
    """
    if not os.path.isdir(stage1_dir):
        return []

    datasets = []
    for name in os.listdir(stage1_dir):
        ds_dir = os.path.join(stage1_dir, name)
        if not os.path.isdir(ds_dir):
            continue
        if os.path.exists(
            os.path.join(ds_dir, "s1_submission_mentions.jsonl.zst")
        ) or os.path.exists(os.path.join(ds_dir, "s1_comment_mentions.jsonl.zst")):
            datasets.append(name)

    return sorted(datasets)


def iter_stage1_files(
    datasets: List[str], stage1_dir: str
) -> Generator[Tuple[str, str, str], None, None]:
    """Iterate over Stage 1 files.

    Yields tuples of (dataset, file_type, file_path) for each existing
    Stage 1 output file across all specified datasets.

    Args:
        datasets: List of dataset names to iterate
        stage1_dir: Path to stage1_mentions directory

    Yields:
        (dataset_name, file_type, file_path) tuples where file_type is
        either "submission" or "comment"
    """
    for ds in datasets:
        ds_dir = os.path.join(stage1_dir, ds)
        for ftype, fname in [
            ("submission", "s1_submission_mentions.jsonl.zst"),
            ("comment", "s1_comment_mentions.jsonl.zst"),
        ]:
            path = os.path.join(ds_dir, fname)
            if os.path.exists(path):
                yield (ds, ftype, path)


def get_total_bytes(datasets: List[str], stage1_dir: str) -> int:
    """Calculate total bytes across all Stage 1 files.

    Used for progress percentage calculations during processing.

    Args:
        datasets: List of dataset names
        stage1_dir: Path to stage1_mentions directory

    Returns:
        Total compressed file size in bytes
    """
    total = 0
    for _, _, path in iter_stage1_files(datasets, stage1_dir):
        total += os.path.getsize(path)
    return total


def load_global_match_counts(stage1_dir: str) -> Tuple[Optional[int], Optional[int]]:
    """Load global totals from the stats pass.

    Reads pre-computed statistics from the stats directory to provide
    better ETA estimates during processing.

    Args:
        stage1_dir: Path to stage1_mentions directory

    Returns:
        (total_mentions, total_target_pairs) tuple where:
        - total_mentions counts Stage 1 records (comments + submissions)
        - total_target_pairs counts unique targets per record (sum over records),
          matching Stage 2's targeted loop over unique match strings
    """
    out_dir = os.path.dirname(stage1_dir.rstrip("\\/"))
    path = os.path.join(out_dir, "stats", "global", "match_counts.json")
    if not os.path.exists(path):
        return None, None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None

    if not (
        isinstance(data, dict)
        and "_global" in data
        and isinstance(data["_global"], dict)
    ):
        return None, None

    global_obj = data["_global"]
    total_mentions = global_obj.get("total")
    total_target_pairs = global_obj.get("target_pairs")
    return (
        int(total_mentions) if isinstance(total_mentions, (int, float)) else None,
        (
            int(total_target_pairs)
            if isinstance(total_target_pairs, (int, float))
            else None
        ),
    )
