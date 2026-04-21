"""Data fetching utilities for Yahoo Finance."""

from __future__ import annotations

import contextlib
import io
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
from config import MARKET_BENCHMARK

try:
    # Optional: only used for defensive normalization if caller didn't normalize.
    from tickers import normalize_ticker
except Exception:  # pragma: no cover
    normalize_ticker = None  # type: ignore[assignment]


def fetch_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    include_benchmark: bool = True,
    progress_interval: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Fetch daily stock data from Yahoo Finance.

    Returns:
        Tuple of (prices_df, benchmark_df, errors_dict)
    """
    try:
        import yfinance as yf
    except ImportError:
        print("Error: yfinance not installed. Install with: pip install yfinance")
        sys.exit(1)

    all_data = []
    errors: Dict[str, str] = {}
    ok_count = 0
    empty_count = 0
    error_count = 0

    # Extend date range for market holidays
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=7)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    total = len(tickers)
    print(f"\nFetching data for {total} tickers...")
    print(f"Date range: {start_date} to {end_date}")

    start_ts = time.time()
    _print_fetch_progress_header(prefix="")

    for i, raw_ticker in enumerate(tickers, 1):
        if progress_interval > 0 and (i % progress_interval == 0 or i == total):
            elapsed = time.time() - start_ts
            rate = (i / elapsed) if elapsed > 0 else 0.0
            progress_pct = (i / total * 100.0) if total else 100.0
            eta_seconds = ((total - i) / rate) if rate > 0 else -1.0
            print(
                _format_fetch_progress_line(
                    checked=i,
                    total=total,
                    ok=ok_count,
                    no_data=empty_count,
                    error=error_count,
                    rate=rate,
                    progress_pct=progress_pct,
                    eta_seconds=eta_seconds,
                    prefix="",
                )
            )

        ticker = raw_ticker
        if normalize_ticker is not None:
            ticker = normalize_ticker(raw_ticker)
            if not ticker:
                # Skip empty/invalid post-normalization
                error_count += 1
                errors[str(raw_ticker)] = "Invalid ticker after normalization"
                continue

        try:
            stock = yf.Ticker(ticker)
            # yfinance can print noisy per-ticker lines (e.g. delisting warnings).
            # We summarize outcomes in the progress table + fetch_log.json instead.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                hist = stock.history(
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    auto_adjust=False,
                )

            if hist.empty:
                empty_count += 1
                errors[ticker] = "No data returned"
                continue

            hist = _process_ticker_history(hist, ticker)
            all_data.append(hist)
            ok_count += 1

        except Exception as e:
            error_count += 1
            errors[ticker] = str(e)

        if i % 10 == 0:
            time.sleep(0.1)

    # Combine and filter
    prices_df = pd.DataFrame()
    if all_data:
        prices_df = pd.concat(all_data, ignore_index=True)
        prices_df = prices_df[
            (prices_df["date"] >= start_date) & (prices_df["date"] <= end_date)
        ]

    # Fetch benchmark
    benchmark_df = pd.DataFrame()
    if include_benchmark:
        benchmark_df = _fetch_benchmark(start_dt, end_dt, start_date, end_date)

    # Final summary (useful when Yahoo prints many per-ticker lines)
    ok_rate = (ok_count / total) if total else 0.0
    print(
        f"\nFetch summary: attempted={total}, ok={ok_count}, no_data={empty_count}, "
        f"error={error_count}, ok_rate={ok_rate:.1%}"
    )

    return prices_df, benchmark_df, errors


def _process_ticker_history(hist: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Process raw Yahoo Finance history into standard format."""
    hist = hist.reset_index()
    hist["ticker"] = ticker
    hist = hist.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    hist = hist[[c for c in cols if c in hist.columns]]
    hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
    return hist


def _fetch_benchmark(
    start_dt: datetime, end_dt: datetime, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch market benchmark data."""
    import yfinance as yf

    print(f"\nFetching market benchmark ({MARKET_BENCHMARK})...")
    try:
        benchmark = yf.Ticker(MARKET_BENCHMARK)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            hist = benchmark.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                auto_adjust=False,
            )

        if hist.empty:
            return pd.DataFrame()

        hist = hist.reset_index()
        df = hist.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        df = df[[c for c in cols if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    except Exception as e:
        print(f"  Warning: Failed to fetch benchmark: {e}")
        return pd.DataFrame()


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS, similar to pipeline output."""
    if seconds < 0:
        return "--:--"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_fetch_progress_line(
    checked: int,
    total: int,
    ok: int,
    no_data: int,
    error: int,
    rate: float,
    progress_pct: float,
    eta_seconds: float,
    prefix: str = "",
) -> str:
    """Format a fixed-width progress line with pipe separators."""
    # Keep columns compact to avoid wrapping in narrow terminals.
    checked_col = f"{checked:>8,}"
    total_col = f"{total:>8,}"
    ok_col = f"{ok:>6,}"
    no_data_col = f"{no_data:>6,}"
    error_col = f"{error:>6,}"
    # Keep Rate column width consistent with header (including '/s').
    # This avoids the "one char off" look in PowerShell.
    RATE_WIDTH = 8
    rate_num = int(round(rate))
    rate_col = f"{rate_num:>{RATE_WIDTH-2},}/s"
    progress_col = f"{progress_pct:>5.1f}%"
    _ = eta_seconds  # computed by caller; not printed (keeps table narrow)
    return (
        f"{prefix}| {checked_col} | {total_col} | {ok_col} | {no_data_col} | {error_col} "
        f"| {rate_col} | {progress_col} |"
    )


def _print_fetch_progress_header(prefix: str = "") -> None:
    """Print header for the fetch progress table (pipeline-style)."""
    RATE_WIDTH = 8
    header = (
        f"| {'Checked':>8} | {'Total':>8} | {'OK':>6} | {'NoData':>6} | {'Err':>6} "
        f"| {'Rate':>{RATE_WIDTH}} | {'Prog':>6} |"
    )
    sep_line = "-" * len(header)
    print(prefix + sep_line)
    print(prefix + header)
    print(prefix + sep_line)
