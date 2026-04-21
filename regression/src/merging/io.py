"""Panel I/O utilities for saving and loading panel data."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from config import RegressionConfig


def save_panel(
    df: pd.DataFrame,
    config: RegressionConfig,
    output_dir: Path,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    """Save panel data to Parquet with metadata.

    Args:
        df: Panel DataFrame
        config: Regression configuration used to build panel
        output_dir: Output directory
        metadata: Additional metadata to save

    Returns:
        Path to saved panel file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save panel data
    panel_path = output_dir / "panel_data.parquet"
    df.to_parquet(panel_path, index=False, compression="zstd")

    # Build metadata
    meta = {
        "created_at": datetime.now().isoformat(),
        "n_rows": len(df),
        "n_tickers": df["ticker"].nunique() if "ticker" in df.columns else 0,
        "n_dates": df["date"].nunique() if "date" in df.columns else 0,
        "date_range": {
            "start": str(df["date"].min()) if "date" in df.columns else None,
            "end": str(df["date"].max()) if "date" in df.columns else None,
        },
        "columns": df.columns.tolist(),
        "config": {
            "primary_sentiment_model": config.primary_sentiment_model,
            "horizons": config.horizons,
            "attention_measure": config.attention.measure,
            "attention_standardization": config.attention.standardization,
            "sentiment_split_pos_neg": config.sentiment.split_pos_neg,
            "sentiment_min_posts": config.sentiment.min_posts_for_valid_sentiment,
            "sentiment_standardization": config.sentiment.standardization,
            "sample_start_date": config.sample.start_date,
            "sample_end_date": config.sample.end_date,
            "sample_min_financial_days": config.sample.min_trading_days_financial,
            "sample_min_mention_days": config.sample.min_trading_days_mentioned,
            "sample_min_valid_sentiment_days": config.sample.min_trading_days_valid_sentiment,
        },
    }

    # Add any additional metadata
    if metadata:
        meta.update(metadata)

    # Save metadata
    meta_path = output_dir / "panel_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved panel data:")
    print(f"  → {panel_path}")
    print(f"  → {meta_path}")
    print(
        f"  → {len(df):,} rows, {meta['n_tickers']:,} tickers, {meta['n_dates']:,} dates"
    )

    return panel_path


def load_panel_data(
    output_dir: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load panel data from Parquet.

    Args:
        output_dir: Directory containing panel_data.parquet
        columns: Specific columns to load (None = all)

    Returns:
        Panel DataFrame
    """
    panel_path = output_dir / "panel_data.parquet"

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel data not found: {panel_path}")

    df = pd.read_parquet(panel_path, columns=columns)
    return df


def load_panel_metadata(output_dir: Path) -> Dict[str, Any]:
    """Load panel metadata.

    Args:
        output_dir: Directory containing panel_metadata.json

    Returns:
        Metadata dictionary
    """
    meta_path = output_dir / "panel_metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Panel metadata not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def panel_is_stale(
    output_dir: Path,
    config: RegressionConfig,
) -> bool:
    """Check if panel data needs to be rebuilt.

    Panel is considered stale if:
    - Panel files don't exist
    - Configuration has changed significantly

    Args:
        output_dir: Directory containing panel data
        config: Current regression configuration

    Returns:
        True if panel should be rebuilt
    """
    panel_path = output_dir / "panel_data.parquet"
    meta_path = output_dir / "panel_metadata.json"

    # Check if files exist
    if not panel_path.exists() or not meta_path.exists():
        return True

    # Load existing metadata
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, IOError):
        return True

    # Check if key configuration has changed
    existing_config = meta.get("config", {})

    # Key parameters that require rebuild if changed
    key_params = [
        ("primary_sentiment_model", config.primary_sentiment_model),
        ("horizons", config.horizons),
        ("attention_measure", config.attention.measure),
        ("attention_standardization", config.attention.standardization),
        ("sentiment_split_pos_neg", config.sentiment.split_pos_neg),
        ("sentiment_min_posts", config.sentiment.min_posts_for_valid_sentiment),
        ("sample_start_date", config.sample.start_date),
        ("sample_end_date", config.sample.end_date),
    ]

    for param_name, current_value in key_params:
        existing_value = existing_config.get(param_name)
        if existing_value != current_value:
            print(
                f"Panel stale: {param_name} changed from {existing_value} to {current_value}"
            )
            return True

    return False
