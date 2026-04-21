"""
Generate clean_tickers.csv from NASDAQ and NYSE source files.

Usage:
    python generate_clean_tickers.py
"""

import sys

import os
import re

import pandas as pd

# Look for files in the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "clean_tickers.csv")

# Use only historical union files produced by fetch_historical_tickers.py.
_NASDAQ_UNION = os.path.join(BASE_DIR, "nasdaqlisted_union.txt")
_NYSE_UNION = os.path.join(BASE_DIR, "nyse_union.txt")
if not os.path.exists(_NASDAQ_UNION):
    raise FileNotFoundError(
        f"Required union file missing: {_NASDAQ_UNION}. "
        "Run fetch_historical_tickers.py first."
    )
if not os.path.exists(_NYSE_UNION):
    raise FileNotFoundError(
        f"Required union file missing: {_NYSE_UNION}. "
        "Run fetch_historical_tickers.py first."
    )

NASDAQ_PATH = _NASDAQ_UNION
NYSE_PATH = _NYSE_UNION
NYSE_IS_OTHER_FMT = True
print(f"[ticker universe] Using historical union: {_NASDAQ_UNION}")
print(f"[ticker universe] Using historical union: {_NYSE_UNION}")


def get_common_names(formal_name):
    """Extract common name variations from formal company name."""
    if not isinstance(formal_name, str):
        return ""

    # Set to collect variations
    variations = set()

    # 1. Base clean: remove things after " - "
    base_name = formal_name.split(" - ")[0]

    # 2. Remove parenthetical info (e.g. "Agilent Technologies, Inc. (The)")
    base_name = re.sub(r"\(.*?\)", "", base_name)

    # 3. Add the cleaned base name to variations (just stripped)
    base_name = base_name.strip(" ,.-")
    variations.add(base_name)

    # 4. Extract name before comma if present
    if "," in base_name:
        pre_comma = base_name.split(",")[0].strip()
        variations.add(pre_comma)

    # 5. Iterative stripping of suffixes
    suffixes = [
        r",?\s+Inc\.?$",
        r",?\s+Corp\.?$",
        r",?\s+Corporation$",
        r",?\s+Ltd\.?$",
        r",?\s+P\.?L\.?C\.?$",
        r",?\s+L\.?P\.?$",
        r",?\s+L\.?L\.?C\.?$",
        r",?\s+S\.?A\.?$",
        r",?\s+N\.?V\.?$",
        r",?\s+Company$",
        r",?\s+Co\.?$",
        r",?\s+Limited$",
        r",?\s+Group$",
        r",?\s+Holdings?$",
        r",?\s+Trust$",
        r",?\s+Fund$",
        r",?\s+REIT$",
        r",?\s+Hldgs?$",
    ]

    # Remove stock types first
    stock_types = [
        r"\s+Common Stock",
        r"\s+Class [A-Z]",
        r"\s+Ordinary Shares",
        r"\s+Depositary Shares",
    ]

    clean = base_name
    for st in stock_types:
        clean = re.sub(st, "", clean, flags=re.IGNORECASE)

    clean = clean.strip(" ,.-")

    # Remove legal entities iteratively until no change
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if re.search(suf, clean, flags=re.IGNORECASE):
                clean = re.sub(suf, "", clean, flags=re.IGNORECASE).strip(" ,.-")
                changed = True

    # Remove "The" from start
    if clean.lower().startswith("the "):
        clean = clean[4:].strip()

    variations.add(clean)

    # 6. Fallback: "Amazon.com" -> "Amazon"
    if "." in clean and " " not in clean:
        variations.add(clean.split(".")[0])

    # Remove empty or super short noise
    final_vars = [v for v in variations if len(v) > 1]

    # Common English words that should NOT become standalone company name variants.
    # These produce false positives when matched via Aho-Corasick in the pipeline.
    # We still allow longer variants like "Monday.com" but not the stripped "Monday".
    # Also terms that are actual company names but cause too many false positives.
    COMMON_WORD_BLACKLIST = {
        # Days of the week
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        # Common nouns/adjectives that are company names
        "ally",
        "ball",
        "banner",
        "bill",
        "block",
        "bloom",
        "bridge",
        "buckle",
        "cable",
        "core",
        "crown",
        "decent",
        "delta",
        "dish",
        "door",
        "drive",
        "eagle",
        "earth",
        "edge",
        "elastic",
        "employers",
        "equitable",
        "ever",
        "five",
        "flex",
        "floor",
        "forge",
        "form",
        "fossil",
        "freedom",
        "fresh",
        "gambling",
        "global",
        "gold",
        "google",
        "green",
        "grow",
        "guess",
        "guess?",  # GES variant with punctuation
        "here",
        "honest",
        "intelligent",
        "jazz",
        "keys",
        "lake",
        "land",
        "leap",
        "life",
        "light",
        "lines",
        "live",
        "local",
        "main",
        "match",
        "max",
        "nasdaq",
        "national",
        "net",
        "new",
        "next",
        "nice",
        "nova",
        "open",
        "oracle",
        "park",
        "path",
        "pattern",
        "peak",
        "plus",
        "plug",
        "pool",
        "popular",
        "post",
        "power",
        "press",
        "prime",
        "pure",
        "quantum",
        "quest",
        "range",
        "real",
        "reddit",
        "root",
        "sage",
        "salt",
        "sea",
        "shell",
        "sky",
        "smart",
        "snap",
        "solar",
        "solo",
        "sound",
        "south",
        "spot",
        "spring",
        "square",
        "star",
        "stem",
        "step",
        "stone",
        "store",
        "strategy",
        "strive",
        "sun",
        "sure",
        "taiwan",
        "target",
        "titan",
        "top",
        "tower",
        "tree",
        "trip",
        "true",
        "trust",
        "uber",
        "ultra",
        "union",
        "united",
        "unity",
        "urban",
        "us",
        "use",
        "vale",
        "value",
        "via",
        "view",
        "waters",
        "west",
        "win",
        "wing",
        "wish",
        "wolf",
        "world",
        "zone",
        "zoom",
        # Finance jargon
        "bull",
        "bear",
        "bullish",
        "bearish",
        "bcg",  # BCG - Boston Consulting Group conspiracy posts, not Binah Capital Group
        "rally",
        "chart",
        "trade",
        "trend",
        # Very common words on these subreddits
        "coffee",  # JVA company name
        # Finance/options terminology
        "arm",  # Adjustable-Rate Mortgage
        "dte",  # Days To Expiration
        # Other problematic words
        "bandwidth",  # BAND - generic tech term
        "commerce",  # CMRC - "e-commerce" is everywhere
        "fly",  # FLY - "fly to the moon", "let it fly"
        "founder",  # FGL - "company founder" is common
        "gravity",
        "highway",  # HIHO - common word
        "interface",
        "national bank",  # NBHC - generic term for any national bank
        "best buy",  # Common phrase, not just the store
        # Words found in false positive analysis
        "lottery",  # LTRYW - extremely common word
        "troops",  # TROO - "send troops", "troops deployed"
        "graham",  # GHC - Benjamin Graham (investing legend)
        "universal",  # UVV - extremely common adjective
        "northern",  # NTRS - common directional word
        "progressive",  # PGR - common word
        "coherent",  # COHR - common adjective
        "citizens",  # CIA - common word
        "belive",  # BLIV - typo for "believe"
        "first bank",  # FRBA - matches any "First Bank" globally
        "american financial",  # AFG - too generic
        "financial institutions",  # FISI - generic industry term
        "my size",  # MYSZ - common phrase
        # Crypto company names (these cause false positives, not stock discussions)
        "solana",  # HSDT - Solana crypto is 99% of mentions
        "celsius",  # CELH - Celsius Network crypto
        # Brokerage names (people mention brokers, not the stock)
        "interactive brokers",  # IBKR - almost always broker context
        # Other common words
        "urgent",  # ULY (Urgent.ly) - extremely common word
    }

    # Filter out common words that would cause false positives
    final_vars = [v for v in final_vars if v.lower() not in COMMON_WORD_BLACKLIST]

    # Also filter out single-word names that are too short (high false positive risk)
    # Multi-word names like "Post Holdings" are fine, but "Post" alone is risky.
    # Exception: well-known company names that are commonly used standalone.
    # Focus on short names (≤5 chars) since longer names pass the length filter anyway.
    # Prioritize stocks with strong retail/social media attention.
    KNOWN_COMPANY_NAMES = {
        # Big tech
        "tesla",
        "nvidia",
        "apple",
        "amazon",
        "google",
        "meta",  # Facebook parent
        "intel",
        "cisco",
        "adobe",
        "dell",
        "amd",  # Advanced Micro Devices - huge WSB following
        "nvidia",
        # Streaming/social
        "netflix",
        "spotify",
        "roku",
        "yelp",
        # E-commerce/fintech
        "paypal",
        "ebay",
        "etsy",
        # Retail
        "costco",
        "starbucks",
        "walmart",
        "disney",
        "nike",
        "pepsi",
        "kraft",
        # Meme stocks / high retail attention
        "amc",  # AMC Entertainment - meme stock
        "nokia",  # Meme stock era attention
        "chewy",  # Ryan Cohen connection
        "lucid",  # Lucid Motors - EV hype
        "hertz",  # Meme attention after bankruptcy
        # Automotive
        "ford",
        # Rideshare
        "lyft",
        # Tech/electronics
        "sony",
        "sega",
        "xerox",
        # Consumer staples / iconic brands
        "coca",  # Coca-Cola (KO) often referenced as "Coca"
    }
    final_vars = [
        v
        for v in final_vars
        if " " in v or len(v) >= 6 or v.lower() in KNOWN_COMPANY_NAMES
    ]

    # Sort distinct, shortest first (most "common")
    final_vars.sort(key=len)

    seen = set()
    unique_vars = []

    for v in final_vars:
        v_clean = v.lower().strip()
        if v_clean not in seen:
            seen.add(v_clean)
            unique_vars.append(v.strip())

    return "|".join(unique_vars)


def main():
    print(f"Reading {NASDAQ_PATH}...")
    try:
        nasdaq = pd.read_csv(NASDAQ_PATH, sep="|", on_bad_lines="skip")
        if "Symbol" not in nasdaq.columns:
            nasdaq = pd.read_csv(NASDAQ_PATH, sep="|")

        nasdaq = nasdaq.dropna(subset=["Symbol"])
        df_nasdaq = nasdaq[["Symbol", "Security Name", "ETF"]].copy()
        df_nasdaq.columns = ["ticker", "raw_name", "is_etf"]
    except Exception as e:
        print(f"Failed parsing Nasdaq: {e}")
        return

    print(f"Reading {NYSE_PATH}...")
    try:
        if NYSE_IS_OTHER_FMT:
            # otherlisted_union.txt: pipe-delimited, columns include
            # "ACT Symbol", "Security Name", "ETF" (from fetch_historical_tickers.py)
            nyse = pd.read_csv(NYSE_PATH, sep="|", on_bad_lines="skip", dtype=str)
            nyse = nyse.dropna(subset=["ACT Symbol"])
            df_nyse = nyse[["ACT Symbol", "Security Name"]].copy()
            df_nyse["is_etf"] = (
                nyse["ETF"].values if "ETF" in nyse.columns else float("nan")
            )
            df_nyse.columns = ["ticker", "raw_name", "is_etf"]
        else:
            # Original nyse-listed.csv: comma-delimited, columns ACT Symbol / Company Name
            nyse = pd.read_csv(NYSE_PATH)
            nyse = nyse.dropna(subset=["ACT Symbol"])
            df_nyse = nyse[["ACT Symbol", "Company Name"]].copy()
            df_nyse["is_etf"] = float("nan")
            df_nyse.columns = ["ticker", "raw_name", "is_etf"]
    except Exception as e:
        print(f"Failed parsing NYSE: {e}")
        return

    print("Concatenating and processing...")
    combined = pd.concat([df_nasdaq, df_nyse], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker"], keep="first")
    combined["name"] = combined["raw_name"].apply(get_common_names)

    final_df = combined[["ticker", "name", "raw_name", "is_etf"]]

    print(f"Saving {len(final_df)} rows to {OUTPUT_PATH}")
    final_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
