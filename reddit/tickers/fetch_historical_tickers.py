"""Build survivorship-bias-corrected ticker union files from SEC EDGAR data.

Sources:
  * Current NASDAQ / NYSE listing files from the NASDAQ Trader FTP
    (downloaded by fetch_listings.py): nasdaqlisted.txt and otherlisted.txt.
  * SEC EDGAR company_tickers_exchange.json -- all SEC registrants with
    exchange information.  Companies remain in EDGAR records even after
    delisting, so this file contains many historically-listed stocks that
    no longer appear in the live listing snapshots.
  * A curated safety-net list of high-profile delistings 2018-2024
    (meme stocks, bank failures, major acquisitions) that are relevant to
    Reddit finance discussion but may have dropped out of EDGAR as well.

Outputs (written to the same directory as this script):
    nasdaqlisted_union.txt       -- baseline plus EDGAR plus curated delistings
    nyse_union.txt               -- baseline plus EDGAR in NYSE-union format

When generate_clean_tickers.py finds these files it prefers them over the
bare listing snapshots, expanding the Aho-Corasick ticker universe to reduce
survivorship bias in the Reddit mention pipeline.

Prerequisite:
    Run fetch_listings.py first to download the raw listing files.

Usage:
    python fetch_historical_tickers.py [--out-dir DIR] [--delay SECS]
"""

import argparse
import os
import re
import sys
import time

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths (relative to this script directory)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NASDAQ_FILE = os.path.join(BASE_DIR, "nasdaqlisted.txt")
NYSE_FILE = os.path.join(BASE_DIR, "otherlisted.txt")

# ---------------------------------------------------------------------------
# SEC EDGAR source
# ---------------------------------------------------------------------------
EDGAR_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

# Valid stock ticker: 1-6 uppercase letters, optional class suffix -A .. -Z
_TICKER_RE = re.compile(r"^[A-Z]{1,6}(-[A-Z])?$")

# ---------------------------------------------------------------------------
# Safety-net: curated delistings likely discussed on Reddit 2018-2024
# ---------------------------------------------------------------------------
KNOWN_DELISTINGS = {
    # Meme stocks / retail events
    "BBBY": "Bed Bath and Beyond Inc.",
    "TWTR": "Twitter Inc.",
    "GME": "GameStop Corp.",
    "AMC": "AMC Entertainment Holdings Inc.",
    "WISH": "ContextLogic Inc.",
    "CLOV": "Clover Health Investments Corp.",
    "WKHS": "Workhorse Group Inc.",
    # Major acquisitions 2018-2024
    "ATVI": "Activision Blizzard Inc.",
    "CTXS": "Citrix Systems Inc.",
    "WORK": "Slack Technologies Inc.",
    "MXIM": "Maxim Integrated Products Inc.",
    "XLNX": "Xilinx Inc.",
    "AVLR": "Avalara Inc.",
    "COUP": "Coupa Software Inc.",
    "POSH": "Poshmark Inc.",
    "SUMO": "Sumo Logic Inc.",
    "WLTW": "Willis Towers Watson PLC",
    "VIAC": "ViacomCBS Inc.",
    "DISCA": "Discovery Inc.",
    "DISCK": "Discovery Inc. Class C",
    # Bank failures 2023
    "SIVB": "SVB Financial Group",
    "FRC": "First Republic Bank",
    "SBNY": "Signature Bank",
    # Retail bankruptcies
    "JCP": "J. C. Penney Company Inc.",
    "GNC": "GNC Holdings Inc.",
    "HTZ": "Hertz Global Holdings Inc.",
    # EV / SPAC wave
    "RIDE": "Lordstown Motors Corp.",
    "NKLA": "Nikola Corporation",
    "SPCE": "Virgin Galactic Holdings Inc.",
    "FSR": "Fisker Inc.",
    # Energy bankruptcies
    "OAS": "Oasis Petroleum Inc.",
    "DNR": "Denbury Resources Inc.",
    "WPX": "WPX Energy Inc.",
    "CHK": "Chesapeake Energy Corp.",
    # Other notable
    "CS": "Credit Suisse Group AG",
    "LB": "L Brands Inc.",
    "RTN": "Raytheon Co.",
    "UTX": "United Technologies Corp.",
}


def fetch_edgar_tickers(session: requests.Session, delay: float = 1.0) -> pd.DataFrame:
    """Download SEC EDGAR company_tickers_exchange.json and return as DataFrame."""
    headers = {
        "User-Agent": "academic-research/1.0 thesis@university.edu",
        "Accept": "application/json",
    }
    data = None
    for attempt in range(3):
        try:
            resp = session.get(EDGAR_URL, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"Failed to fetch EDGAR data: {exc}") from exc
            print(f"  Retry {attempt + 1}/3 after error: {exc}")
            time.sleep(delay * 2)
    assert data is not None
    fields = data.get("fields", ["cik", "name", "ticker", "exchange"])
    rows = data.get("data", [])
    df = pd.DataFrame(rows, columns=fields)
    df["exchange"] = df["exchange"].fillna("").astype(str).str.strip().str.lower()
    print(f"[EDGAR] Downloaded {len(df):,} CIK records.")
    return df


def build_nasdaq_union(edgar_df: pd.DataFrame, current_path: str) -> pd.DataFrame:
    """Return union DataFrame in nasdaqlisted.txt format."""
    try:
        current = pd.read_csv(current_path, sep="|", on_bad_lines="skip", dtype=str)
        current["Symbol"] = current["Symbol"].fillna("").str.strip().str.upper()
    except Exception as exc:
        print(f"Warning: could not read {current_path}: {exc}", file=sys.stderr)
        current = pd.DataFrame(
            columns=[
                "Symbol",
                "Security Name",
                "Market Category",
                "Test Issue",
                "Financial Status",
                "Round Lot Size",
                "ETF",
                "NextShares",
            ]
        )

    known_syms = set(current["Symbol"].dropna())
    extras = []

    def _add(sym, name):
        sym = sym.strip().upper()
        if not sym or not _TICKER_RE.match(sym) or sym in known_syms:
            return
        extras.append(
            {
                "Symbol": sym,
                "Security Name": str(name or sym),
                "Market Category": "Q",
                "Test Issue": "N",
                "Financial Status": "N",
                "Round Lot Size": "100",
                "ETF": "N",
                "NextShares": "N",
            }
        )
        known_syms.add(sym)

    for _, row in edgar_df[edgar_df["exchange"] == "nasdaq"].iterrows():
        _add(str(row.get("ticker") or ""), str(row.get("name") or ""))

    for sym, name in KNOWN_DELISTINGS.items():
        _add(sym, name)

    if extras:
        union = pd.concat([current, pd.DataFrame(extras)], ignore_index=True)
    else:
        union = current

    union = union.drop_duplicates(subset=["Symbol"], keep="first")
    union = union.sort_values("Symbol").reset_index(drop=True)
    return union


def build_nyse_union(edgar_df: pd.DataFrame, current_nyse_path: str) -> pd.DataFrame:
    """Return union DataFrame in nyse_union.txt format (pipe-delimited)."""
    try:
        current = pd.read_csv(
            current_nyse_path, sep="|", on_bad_lines="skip", dtype=str
        )
        current = current.rename(columns={"Company Name": "Security Name"})
        current["ACT Symbol"] = current["ACT Symbol"].fillna("").str.strip().str.upper()
    except Exception as exc:
        print(f"Warning: could not read {current_nyse_path}: {exc}", file=sys.stderr)
        current = pd.DataFrame(columns=["ACT Symbol", "Security Name"])

    if "ETF" not in current.columns:
        current["ETF"] = "N"

    known_syms = set(current["ACT Symbol"].dropna())
    extras = []

    def _add(sym, name):
        sym = sym.strip().upper()
        if not sym or not _TICKER_RE.match(sym) or sym in known_syms:
            return
        extras.append(
            {"ACT Symbol": sym, "Security Name": str(name or sym), "ETF": "N"}
        )
        known_syms.add(sym)

    for _, row in edgar_df[edgar_df["exchange"] == "nyse"].iterrows():
        _add(str(row.get("ticker") or ""), str(row.get("name") or ""))

    if extras:
        union = pd.concat([current, pd.DataFrame(extras)], ignore_index=True)
    else:
        union = current

    union = union.drop_duplicates(subset=["ACT Symbol"], keep="first")
    union = union.sort_values("ACT Symbol").reset_index(drop=True)
    return union[["ACT Symbol", "Security Name", "ETF"]]


def print_stats(nasdaq_union: pd.DataFrame, nyse_union: pd.DataFrame) -> None:
    """Print a summary comparing union sizes against current listing files."""
    try:
        curr_nasdaq = len(pd.read_csv(NASDAQ_FILE, sep="|", on_bad_lines="skip"))
    except Exception:
        curr_nasdaq = 0
    try:
        curr_nyse = len(pd.read_csv(NYSE_FILE, sep="|", on_bad_lines="skip"))
    except Exception:
        curr_nyse = 0

    added_n = len(nasdaq_union) - curr_nasdaq
    added_y = len(nyse_union) - curr_nyse
    print()
    print("=" * 60)
    print("Survivorship-bias correction summary")
    print("=" * 60)
    print(f"  NASDAQ current:     {curr_nasdaq:>6,}")
    print(f"  NASDAQ union:       {len(nasdaq_union):>6,}  (+{added_n:,})")
    print(f"  NYSE current:       {curr_nyse:>6,}")
    print(f"  NYSE union:         {len(nyse_union):>6,}  (+{added_y:,})")
    print()
    print("Known-delisting spot check:")
    nasdaq_syms = set(nasdaq_union["Symbol"].str.upper())
    nyse_syms = set(nyse_union["ACT Symbol"].str.upper())
    all_syms = nasdaq_syms | nyse_syms
    for sym in [
        "BBBY",
        "TWTR",
        "FRC",
        "SIVB",
        "SBNY",
        "ATVI",
        "JCP",
        "HTZ",
        "CTXS",
        "GME",
    ]:
        status = "present" if sym in all_syms else "MISSING"
        print(f"  {sym:<8} {status}")
    print("=" * 60)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch SEC EDGAR tickers and build union listing files."
    )
    parser.add_argument(
        "--out-dir",
        default=BASE_DIR,
        help="Directory to write union files (default: script directory)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between network retries (default: 1.0)",
    )
    args = parser.parse_args(argv)

    out_nasdaq = os.path.join(args.out_dir, "nasdaqlisted_union.txt")
    out_nyse = os.path.join(args.out_dir, "nyse_union.txt")

    session = requests.Session()

    print("Fetching SEC EDGAR company_tickers_exchange.json ...")
    edgar_df = fetch_edgar_tickers(session, delay=args.delay)

    print("\nBuilding NASDAQ union ...")
    nasdaq_union = build_nasdaq_union(edgar_df, NASDAQ_FILE)
    print(f"  NASDAQ union: {len(nasdaq_union):,} entries")

    print("\nBuilding NYSE union ...")
    nyse_union = build_nyse_union(edgar_df, NYSE_FILE)
    print(f"  NYSE union:   {len(nyse_union):,} entries")

    nasdaq_union.to_csv(out_nasdaq, sep="|", index=False)
    print(f"\nWrote: {out_nasdaq}")
    nyse_union.to_csv(out_nyse, sep="|", index=False)
    print(f"Wrote: {out_nyse}")

    print_stats(nasdaq_union, nyse_union)


if __name__ == "__main__":
    main()
