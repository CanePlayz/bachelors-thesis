"""Stage 3: Aggregate sentiment by day, stock, and model.

This module implements the final stage of the pipeline, which aggregates
scored mentions into daily sentiment statistics per stock and model.

Aggregation formula:
    For each (day, ticker, model):
    - Weight w = max(1, log(1 + score))  (for both submissions and comments)
    - sentiment_avg = Σ(w * sentiment) / Σ(w)
    - sentiment_disp = weighted standard deviation

Output format (long-format CSV):
    day,ticker,model,sentiment_avg,sentiment_disp,mention_count,weight_sum,
    submission_count,comment_count

This long format allows easy pivoting/filtering in pandas for analysis.
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import orjson
from common.io_utils import iter_zst_ndjson
from data.config import SENTIMENT_MODELS, STAGE2_OUT_DIR, STAGE3_OUT_DIR
from helpers.logging_utils import (format_duration, indent, print_banner,
                                   timestamp_to_date)

from .scorer import get_model_key


@dataclass
class AggregationBucket:
    """Accumulator for weighted sentiment statistics.

    Used to compute weighted average and dispersion for a
    (day, ticker, model) combination.
    """

    weight_sum: float = 0.0
    weighted_sentiment_sum: float = 0.0
    weighted_sq_sum: float = 0.0  # For variance calculation
    mention_count: int = 0
    submission_count: int = 0
    comment_count: int = 0

    def add(self, sentiment: float, weight: float, is_submission: bool) -> None:
        """Add a mention to the bucket.

        Args:
            sentiment: Signed sentiment score
            weight: Importance weight for this mention
            is_submission: True if submission, False if comment
        """
        self.weight_sum += weight
        self.weighted_sentiment_sum += weight * sentiment
        self.weighted_sq_sum += weight * sentiment * sentiment
        self.mention_count += 1

        if is_submission:
            self.submission_count += 1
        else:
            self.comment_count += 1

    @property
    def sentiment_avg(self) -> float:
        """Compute weighted average sentiment."""
        if self.weight_sum <= 0:
            return 0.0
        return self.weighted_sentiment_sum / self.weight_sum

    @property
    def sentiment_disp(self) -> float:
        """Compute weighted standard deviation (dispersion).

        Uses the formula: sqrt(Σ(w*x²)/Σw - (Σ(w*x)/Σw)²)
        """
        if self.weight_sum <= 0 or self.mention_count < 2:
            return 0.0

        mean = self.sentiment_avg
        variance = (self.weighted_sq_sum / self.weight_sum) - (mean * mean)

        # Handle numerical precision issues
        if variance < 0:
            variance = 0.0

        return math.sqrt(variance)


def compute_weight(score: int) -> float:
    """Compute importance weight for a mention.

    Uses log-transformed Reddit score as an approval-based weight,
    applied identically to submissions and comments.

    Formula:
        w = max(1, log(1 + score))

    Args:
        score: Reddit score (upvotes - downvotes)

    Returns:
        Importance weight (always >= 1.0)
    """
    return max(1.0, math.log(1.0 + max(0, score)))


def discover_stage2_datasets(stage2_dir: str) -> List[str]:
    """Find datasets with Stage 2 outputs.

    Args:
        stage2_dir: Path to stage2_sentiment directory

    Returns:
        List of dataset names with Stage 2 outputs
    """
    if not os.path.isdir(stage2_dir):
        return []

    datasets = []
    for name in os.listdir(stage2_dir):
        ds_dir = os.path.join(stage2_dir, name)
        if not os.path.isdir(ds_dir):
            continue

        has_subs = os.path.exists(
            os.path.join(ds_dir, "s2_submission_mentions.jsonl.zst")
        )
        has_coms = os.path.exists(os.path.join(ds_dir, "s2_comment_mentions.jsonl.zst"))

        if has_subs or has_coms:
            datasets.append(name)

    return sorted(datasets)


def aggregate_dataset(
    dataset: str,
    stage2_dir: str,
    model_keys: List[str],
) -> Dict[Tuple[str, str, str], AggregationBucket]:
    """Aggregate sentiment for a single dataset.

    Streams through scored mentions and accumulates into buckets.

    Args:
        dataset: Dataset name
        stage2_dir: Path to stage2_sentiment directory
        model_keys: List of model keys to aggregate

    Returns:
        Dict mapping (day, ticker, model_key) -> AggregationBucket
    """
    # Storage: (day, ticker, model_key) -> bucket
    buckets: Dict[Tuple[str, str, str], AggregationBucket] = defaultdict(
        AggregationBucket
    )

    ds_dir = os.path.join(stage2_dir, dataset)

    # Process both file types
    for fname, is_submission in [
        ("s2_submission_mentions.jsonl.zst", True),
        ("s2_comment_mentions.jsonl.zst", False),
    ]:
        fpath = os.path.join(ds_dir, fname)
        if not os.path.exists(fpath):
            continue

        # Stream through mentions
        for record in iter_zst_ndjson(fpath):
            # Extract metadata
            created_utc = record.get("created_utc", 0)
            day = timestamp_to_date(created_utc)
            tickers = record.get("tickers", [])
            score = record.get("score", 0) or 0
            num_comments = record.get("num_comments")

            # Compute weight for this mention
            weight = compute_weight(score)

            # Process each ticker mentioned
            for ticker in tickers:
                # Process each model's scores
                for model_key in model_keys:
                    # Get non-targeted sentiment
                    sentiment = record.get(f"sentiment_{model_key}")

                    # Try targeted sentiment if available
                    targeted = record.get(f"sentiment_targeted_{model_key}", {})
                    if ticker in targeted and targeted[ticker] is not None:
                        # Use targeted score for this specific ticker
                        sentiment = targeted[ticker]

                    # Skip if no valid sentiment
                    if sentiment is None or (
                        isinstance(sentiment, float) and math.isnan(sentiment)
                    ):
                        continue

                    # Add to bucket
                    key = (day, ticker, model_key)
                    buckets[key].add(sentiment, weight, is_submission)

    return dict(buckets)


def write_aggregated_csv(
    all_buckets: Dict[Tuple[str, str, str], AggregationBucket],
    output_path: str,
) -> None:
    """Write aggregated sentiment to CSV.

    Output columns:
        date, ticker, model, sentiment_avg, sentiment_disp,
        mention_count, weight_sum, submission_count, comment_count

    Args:
        all_buckets: Dict of (day, ticker, model) -> bucket
        output_path: Path to output CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write(
            "date,ticker,model,sentiment_avg,sentiment_disp,"
            "mention_count,weight_sum,submission_count,comment_count\n"
        )

        # Sort by (day, ticker, model) for consistent output
        for key in sorted(all_buckets.keys()):
            day, ticker, model = key
            bucket = all_buckets[key]

            # Format floats with reasonable precision
            avg = f"{bucket.sentiment_avg:.6f}"
            disp = f"{bucket.sentiment_disp:.6f}"
            wsum = f"{bucket.weight_sum:.3f}"

            f.write(
                f"{day},{ticker},{model},{avg},{disp},"
                f"{bucket.mention_count},{wsum},"
                f"{bucket.submission_count},{bucket.comment_count}\n"
            )


def write_global_stats(
    all_buckets: Dict[Tuple[str, str, str], AggregationBucket],
    datasets: List[str],
    output_path: str,
    runtime_seconds: float,
) -> None:
    """Write global statistics JSON.

    Args:
        all_buckets: Aggregation buckets
        datasets: List of processed datasets
        output_path: Path to output JSON file
        runtime_seconds: Total processing time
    """
    # Compute summary stats
    unique_days = set(day for day, _, _ in all_buckets.keys())
    unique_tickers = set(ticker for _, ticker, _ in all_buckets.keys())
    unique_models = set(model for _, _, model in all_buckets.keys())

    total_mentions = sum(b.mention_count for b in all_buckets.values())

    stats = {
        "datasets": datasets,
        "date_range": [min(unique_days), max(unique_days)] if unique_days else [],
        "unique_days": len(unique_days),
        "unique_tickers": len(unique_tickers),
        "models_used": sorted(unique_models),
        "total_aggregation_rows": len(all_buckets),
        "total_mentions_aggregated": total_mentions,
        "runtime_seconds": round(runtime_seconds, 2),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(orjson.dumps(stats, option=orjson.OPT_INDENT_2))


def run_stage3(
    datasets: Optional[List[str]] = None,
    stage2_dir: str = STAGE2_OUT_DIR,
    stage3_dir: str = STAGE3_OUT_DIR,
    force: bool = False,
) -> int:
    """Run Stage 3: Aggregate sentiment by day, stock, and model.

    Main entry point for the aggregation stage.

    Args:
        datasets: List of dataset names. If None, auto-discover.
        stage2_dir: Path to Stage 2 outputs
        stage3_dir: Path for Stage 3 outputs
        force: If True, re-aggregate even if outputs exist

    Returns:
        Number of aggregation rows written
    """
    print_banner("STAGE 3: DAILY AGGREGATION")

    start_time = time.time()

    # Check if outputs exist
    output_csv = os.path.join(stage3_dir, "daily_sentiment.csv")
    if os.path.exists(output_csv) and not force:
        print(f"Output exists: {output_csv}")
        print("Use --force to regenerate")
        return 0

    # Discover datasets
    if datasets is None:
        datasets = discover_stage2_datasets(stage2_dir)

    if not datasets:
        print("No Stage 2 outputs found!")
        return 0

    print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")

    # Get model keys from config
    model_keys = [get_model_key(name) for name, _ in SENTIMENT_MODELS]
    print(f"Aggregating for {len(model_keys)} models: {', '.join(model_keys)}")

    # Aggregate all datasets
    all_buckets: Dict[Tuple[str, str, str], AggregationBucket] = {}

    print("\nAggregating datasets...")
    for ds in datasets:
        ds_start = time.time()

        buckets = aggregate_dataset(ds, stage2_dir, model_keys)

        # Merge into global buckets
        for key, bucket in buckets.items():
            if key not in all_buckets:
                all_buckets[key] = AggregationBucket()

            # Merge bucket data
            all_buckets[key].weight_sum += bucket.weight_sum
            all_buckets[key].weighted_sentiment_sum += bucket.weighted_sentiment_sum
            all_buckets[key].weighted_sq_sum += bucket.weighted_sq_sum
            all_buckets[key].mention_count += bucket.mention_count
            all_buckets[key].submission_count += bucket.submission_count
            all_buckets[key].comment_count += bucket.comment_count

        ds_time = time.time() - ds_start
        print(
            f"{indent(1)}[{ds}] {len(buckets):,} combinations in {format_duration(ds_time)}"
        )

    # Write outputs
    print("\nWriting aggregated data...")
    write_aggregated_csv(all_buckets, output_csv)
    print(f"{indent(1)}→ {len(all_buckets):,} rows to {output_csv}")

    # Write global stats
    stats_path = os.path.join(stage3_dir, "global_stats.json")
    runtime = time.time() - start_time
    write_global_stats(all_buckets, datasets, stats_path, runtime)
    print(f"{indent(1)}→ Stats to {stats_path}")

    # Summary
    print_banner("STAGE 3 COMPLETE")
    print(f"Total time: {format_duration(runtime)}")
    print(f"Aggregation rows: {len(all_buckets):,}")

    # Show date range
    if all_buckets:
        days = set(day for day, _, _ in all_buckets.keys())
        print(f"Date range: {min(days)} to {max(days)}")

    return len(all_buckets)
