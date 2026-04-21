"""Ticker and date-range loading utilities.

The financial-data fetcher always sources its ticker universe and its date
range from the Reddit pipeline outputs. There is no manual ticker selection
threshold (inclusion is decided downstream when the regression panel is
built) and no default date range (it tracks whatever the Reddit data covers).
"""

from __future__ import annotations

import re
from typing import List, Tuple

import pandas as pd
from config import MENTION_DAILY_TOTALS_CSV, MENTION_STATS_CSV, TICKERS_CSV


def normalize_ticker(raw: str) -> str:
    """Normalize a ticker symbol for Yahoo Finance.

    This primarily handles Reddit-style tickers like "$AAPL" and trims stray
    punctuation/whitespace.

    Notes:
        - Leaves leading '^' intact (indices like '^GSPC').
        - Converts class-share format like 'BRK.B' -> 'BRK-B' (common Yahoo format).
    """
    if raw is None:
        return ""

    ticker = str(raw).strip().upper()
    if ticker.startswith("$"):
        ticker = ticker[1:]

    # Trim common trailing punctuation from extraction artifacts
    ticker = ticker.strip(" \t\n\r\"'.,;:()[]{}<>")

    # Convert class-share tickers like BRK.B / BF.B to Yahoo's dash format
    if "." in ticker:
        parts = ticker.split(".")
        if len(parts) == 2 and len(parts[1]) == 1:
            ticker = parts[0] + "-" + parts[1]

    # Keep only a safe character set (letters/digits, plus '^', '-', '.')
    ticker = re.sub(r"[^A-Z0-9^\-\.]", "", ticker)
    return ticker


def normalize_tickers(raw_tickers: List[str]) -> List[str]:
    """Normalize + de-duplicate tickers while preserving order."""
    seen = set()
    out: List[str] = []
    for t in raw_tickers:
        nt = normalize_ticker(t)
        if not nt:
            continue
        if nt in seen:
            continue
        seen.add(nt)
        out.append(nt)
    return out


def filter_non_etf_tickers(tickers: List[str]) -> List[str]:
    """Filter out tickers flagged as ETFs in clean_tickers.csv.

    Any symbol not present in the ticker universe is kept (e.g., indices like '^GSPC').
    """
    if not tickers:
        return []
    if not TICKERS_CSV.exists():
        # Best-effort: if we can't check ETF flags, return as-is.
        return normalize_tickers(tickers)

    df = pd.read_csv(TICKERS_CSV, usecols=["ticker", "is_etf"])
    etf_set = set(
        df.loc[df["is_etf"] == "Y", "ticker"].dropna().astype(str).str.upper()
    )

    out: List[str] = []
    for t in tickers:
        tt = normalize_ticker(t)
        if not tt:
            continue
        if tt in etf_set:
            continue
        out.append(tt)
    return normalize_tickers(out)


def load_mentioned_tickers() -> List[str]:
    """Load every ticker that appears at least once in the Reddit pipeline stats.

    No mention-count threshold is applied here; downstream consumers (e.g. the
    regression panel builder) decide which tickers to ultimately include.
    """
    if not MENTION_STATS_CSV.exists():
        raise FileNotFoundError(
            f"Mention stats not found: {MENTION_STATS_CSV}\n"
            "Run the Reddit stats pipeline first."
        )

    df = pd.read_csv(MENTION_STATS_CSV)
    if "ticker" not in df.columns:
        raise ValueError(f"Mention stats missing 'ticker' column: {MENTION_STATS_CSV}")

    tickers = normalize_tickers(df["ticker"].dropna().unique().tolist())
    return filter_non_etf_tickers(tickers)


def load_reddit_date_range() -> Tuple[str, str]:
    """Auto-discover the (start, end) date range from the Reddit daily totals.

    Returns ISO date strings (YYYY-MM-DD).
    """
    if not MENTION_DAILY_TOTALS_CSV.exists():
        raise FileNotFoundError(
            f"Daily mention totals not found: {MENTION_DAILY_TOTALS_CSV}\n"
            "Run the Reddit stats pipeline first."
        )

    df = pd.read_csv(MENTION_DAILY_TOTALS_CSV, usecols=["date"])
    if df.empty:
        raise ValueError(f"Daily mention totals is empty: {MENTION_DAILY_TOTALS_CSV}")

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError(
            f"Daily mention totals has no parseable dates: {MENTION_DAILY_TOTALS_CSV}"
        )

    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")
