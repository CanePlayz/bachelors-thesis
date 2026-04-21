"""Configuration and path constants for the reddit ticker-mention pipeline.

This module defines all the directory paths used throughout the pipeline.
It uses relative path resolution from this file's location to find the
datasets folder and output directory.

Directory structure after running the pipeline:
    reddit/
        out/
            stage1_mentions/         <- Stage 1: extracted mentions per dataset
                {dataset}/
                    s1_submission_mentions.jsonl.zst
                    s1_comment_mentions.jsonl.zst
            stage2_sentiment/        <- Stage 2: scored mentions per dataset
                {dataset}/
                    s2_submission_mentions.jsonl.zst
                    s2_comment_mentions.jsonl.zst
            stage3_aggregated/       <- Stage 3: daily aggregation
                daily_sentiment.csv
            stats/                   <- Stats pass: aggregated statistics
                global/              <- Cross-dataset aggregations
                datasets/            <- Per-dataset stats
"""

import os

# Directory containing this file (src/data)
SRC_DATA_DIR = os.path.abspath(os.path.dirname(__file__))

# Parent directory (src)
SRC_DIR = os.path.abspath(os.path.join(SRC_DATA_DIR, ".."))

# Reddit module root (one level above src)
REDDIT_DIR = os.path.dirname(SRC_DIR)

# Path to the raw Reddit datasets (.zst files).
# Default location: reddit/datasets/  (user must place archives here, e.g.
# after extracting the torrent under reddit/torrent/).
DATASETS_DIR = os.environ.get(
    "REDDIT_DATASETS_DIR",
    os.path.join(REDDIT_DIR, "datasets"),
)

# =============================================================================
# Output directories - organized by pipeline stage
# =============================================================================

# Root output directory for pipeline results (reddit/out/)
DEFAULT_OUT_DIR = os.path.join(REDDIT_DIR, "out")

# Stage 1: Mention extraction outputs (per-dataset)
STAGE1_OUT_DIR = os.path.join(DEFAULT_OUT_DIR, "stage1_mentions")

# Stage 2: Sentiment-scored mentions (per-dataset)
STAGE2_OUT_DIR = os.path.join(DEFAULT_OUT_DIR, "stage2_sentiment")

# Stage 3: Aggregated daily sentiment (global)
STAGE3_OUT_DIR = os.path.join(DEFAULT_OUT_DIR, "stage3_aggregated")

# Stats/plots output directory (global)
STATS_OUT_DIR = os.path.join(DEFAULT_OUT_DIR, "stats")

# =============================================================================
# LMDB directory - unified location for all LMDB databases
# =============================================================================

# LMDB databases live in user home directory to:
# 1. Avoid LMDB Unicode path issues on Windows (project path has em-dash)
# 2. Persist across temp directory cleanups
# 3. Keep caches separate from project data
#
# Structure:
#   ~/reddit_sentiment_lmdb/
#       model_cache/           <- Persistent: cross-run sentiment scores per model
#           {model_name}.lmdb
#       run_scores/            <- Run-local: scores for current pipeline run
#           scores.lmdb

LMDB_BASE_DIR = os.path.join(os.path.expanduser("~"), "reddit_sentiment_lmdb")

# Persistent model caches (survive across pipeline runs)
# Each sentiment model gets its own LMDB directory
CACHES_DIR = os.path.join(LMDB_BASE_DIR, "model_cache")

# Run-local score store (for Stage 2 Pass 1 → Pass 2 handoff)
RUN_SCORES_DIR = os.path.join(LMDB_BASE_DIR, "run_scores")

# =============================================================================
# Ticker data paths
# =============================================================================

# Location of ticker data files
# Contains clean_tickers.csv and exchange listing files
TICKERS_DIR = os.path.join(REDDIT_DIR, "tickers")

# Path to the cleaned ticker CSV file
# Contains: ticker, name(s), is_etf flag
CLEAN_TICKERS_FILE = os.path.join(TICKERS_DIR, "clean_tickers.csv")

# =============================================================================
# Sentiment model configuration
# =============================================================================

# Models to use for sentiment scoring
# Each entry: (model_name, is_targeted)
# - Targeted models use text_pair for ticker-specific sentiment
# - Non-targeted models score the full text globally
SENTIMENT_MODELS = [
    # Targeted model: uses ticker as target for topic-specific sentiment
    ("cardiffnlp/twitter-roberta-base-topic-sentiment-latest", True),
    # BERTweet: trained on Twitter data, good for social media text
    ("finiteautomata/bertweet-base-sentiment-analysis", False),
    # FinTwitBERT: specialized for financial Twitter
    ("StephanAkkerman/FinTwitBERT-sentiment", False),
]

# Maximum token length for sentiment models (HuggingFace tokenizers)
MODEL_MAX_LEN = 512

# Maximum text length to keep (pre-tokenization truncation)
# Hedges against extremely long texts causing slowdowns
# Safe because text with more than this many characters will always exceed MODEL_MAX_LEN tokens
MAX_TEXT_LEN = 4000

# Default batch size for GPU scoring
DEFAULT_BATCH_SIZE = 32
