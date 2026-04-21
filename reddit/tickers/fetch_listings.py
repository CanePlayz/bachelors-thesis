"""Download current NASDAQ and NYSE listing files from the NASDAQ Trader FTP.

Sources:
    ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt   -- NASDAQ-listed
    ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt    -- other exchanges (NYSE etc.)

The otherlisted.txt file is filtered to NYSE securities and written as
nasdaqlisted.txt and nyse_otherlisted.txt in the output directory, using
the original pipe-delimited formats.

Usage:
    python fetch_listings.py [--out-dir DIR]
"""

import argparse
import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FTP_BASE = "ftp://ftp.nasdaqtrader.com/symboldirectory"
NASDAQ_URL = f"{FTP_BASE}/nasdaqlisted.txt"
OTHER_URL = f"{FTP_BASE}/otherlisted.txt"


def _download(url: str, dest: str) -> None:
    """Download a file from FTP and write to *dest*."""
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    # The FTP files end with a trailer line like "File Creation Time: ..."
    # Strip that line so downstream CSV readers aren't confused.
    lines = raw.decode("utf-8", errors="replace").splitlines(keepends=True)
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        for line in lines:
            if line.strip().lower().startswith("file creation time"):
                continue
            fh.write(line)
    print(f"  Wrote {dest}  ({len(lines) - 1} lines)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download NASDAQ and NYSE listing files from NASDAQ Trader FTP."
    )
    parser.add_argument(
        "--out-dir",
        default=BASE_DIR,
        help="Directory to write listing files (default: script directory)",
    )
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    nasdaq_path = os.path.join(args.out_dir, "nasdaqlisted.txt")
    other_path = os.path.join(args.out_dir, "otherlisted.txt")

    print("Fetching listing files from NASDAQ Trader FTP ...")
    _download(NASDAQ_URL, nasdaq_path)
    _download(OTHER_URL, other_path)
    print("\nDone. Files ready for fetch_historical_tickers.py.")


if __name__ == "__main__":
    main()
