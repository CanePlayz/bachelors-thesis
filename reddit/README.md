# reddit/

Reddit ingestion, ticker matching, and sentiment scoring. The module
turns raw Pushshift dumps into a daily
ticker-level sentiment panel that feeds the volatility regressions.

## Layout

```
reddit/
├── README.md
├── src/             Pipeline source code (run via src/main.py)
│   ├── main.py        CLI entry point
│   ├── state.py       Pipeline state + fingerprints
│   ├── common/        Shared zstd/JSONL streaming helpers
│   ├── data/          Paths, configuration
│   ├── extraction/    Stage 1: stream archives, extract ticker mentions
│   ├── sentiment/     Stages 2–3: score mentions, daily aggregation
│   ├── helpers/       Logging utilities
│   └── stats/         Per-stage statistics
├── tickers/         Ticker universe (NASDAQ + NYSE listings)
├── torrent/         .torrent file pointing to the Pushshift dump
├── datasets/        (gitignored) Place extracted .zst dumps here
└── out/             (gitignored) Pipeline outputs
```

## Datasets

The pipeline expects raw Reddit dumps in `reddit/datasets/`. Files named `<dataset>_submissions.zst` / `<dataset>_comments.zst` are
auto-discovered by `src/main.py`. To use a different location, set
`REDDIT_DATASETS_DIR=/path/to/dumps`.

The Pushshift archives are distributed via the torrent file in `torrent/`.

## Pipeline stages

| Stage | Module          | Purpose                                              | Output                                              |
|-------|-----------------|------------------------------------------------------|-----------------------------------------------------|
| 1     | `extraction/`   | Stream `.zst` archives, extract ticker mentions      | `out/stage1_mentions/<dataset>/s1_*.jsonl.zst`      |
| 2     | `sentiment/`    | Score every mention with the sentiment models        | `out/stage2_sentiment/<dataset>/s2_*.jsonl.zst`     |
| 3     | `sentiment/`    | Aggregate to a (day, ticker, model) panel            | `out/stage3_aggregated/daily_sentiment.csv`         |

Each stage writes a fingerprint into the pipeline state file; subsequent runs
skip up-to-date stages automatically.

## Sentiment models

Three transformer-based models are scored on every mention, so the regression
robustness checks downstream do not require re-scoring:

- **`roberta_topic`** — primary, target-aware Twitter-RoBERTa
  (`cardiffnlp/twitter-roberta-base-topic-sentiment-latest`).
- **`bertweet`** — robustness, non-targeted
  (`finiteautomata/bertweet-base-sentiment-analysis`).
- **`fintwit`** — robustness, finance-domain
  (`StephanAkkerman/FinTwitBERT-sentiment`).

## Ticker universe

`tickers/clean_tickers.csv` is the curated symbol list used for mention
matching. Regenerate from the latest NASDAQ/NYSE listings with:

```bash
python tickers/fetch_listings.py
python tickers/fetch_historical_tickers.py
python tickers/generate_clean_tickers.py
```

## Usage

```bash
cd reddit/src

# Auto-discover and run the missing stages
python main.py

# Re-run from a specific stage (requires earlier outputs)
python main.py --stage 2

# Force a full re-run from scratch
python main.py --force

# Restrict to specific datasets (file stems under reddit/datasets/)
python main.py Trading options
```

## GPU notes

Stage 2 uses `transformers` and `torch`. A CUDA-enabled GPU is strongly
recommended; CPU-only runs over the full Pushshift dumps may take several days.

## Outputs consumed downstream

- `out/stage3_aggregated/daily_sentiment.csv` — daily (ticker, day, model)
  sentiment panel consumed by `regression/src/build_panel.py`.
- `out/stats/global/aggregated/ticker_stats.csv` — per-ticker total mention
  counts, used by `financial_data/src/fetch_stock_data.py --from-mentions`.
