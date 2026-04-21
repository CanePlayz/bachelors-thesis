"""Ticker universe loading and filtering for v2 pipeline (src).

This module handles loading stock ticker symbols from clean_tickers.csv,
filtering out ETFs and crypto, and building efficient lookup structures
for fast ticker matching during text extraction.

Key components:
- TickerInfo: Data class for individual ticker metadata
- TickerUniverse: Container with all lookup structures
- Aho-Corasick automaton for O(n) company name matching
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Set, Tuple

import ahocorasick

from .config import CLEAN_TICKERS_FILE

# -------------------------------------------------------------------
# STOPLIST_TOKENS (for bare ticker matching)
# -------------------------------------------------------------------
# This list contains common words and abbreviations (≤5 characters) that
# look like stock tickers but are almost always false positives when they
# appear as bare ALL-CAPS tokens in Reddit text.
#
# These are filtered during bare-ticker matching (Phase 2 in extraction).
# IMPORTANT: Even if a ticker is on this list, it can still be matched
# via $-prefixed cashtag (e.g., $DRS) since that is an explicit reference.
#
# We keep this list focused on ≤5 character words because:
# 1. Bare ALL-CAPS regex only matches 3-5 letter tokens anyway
# 2. Longer words are unlikely to appear in ALL-CAPS naturally
STOPLIST_TOKENS: Set[str] = {
    # ----------------------------------------------------------
    # Common English words (1-5 letters)
    # ----------------------------------------------------------
    # 1-2 letters
    "A",
    "I",
    "AI",
    "AM",
    "AN",
    "AS",
    "AT",
    "BE",
    "BY",
    "DO",
    "GO",
    "HE",
    "IF",
    "IN",
    "IS",
    "IT",
    "ME",
    "MY",
    "NO",
    "OF",
    "OK",
    "ON",
    "OR",
    "SO",
    "TO",
    "UP",
    "US",
    "WE",
    # 3 letters
    "ALL",
    "AND",
    "ANY",
    "ARE",
    "BAD",
    "BIG",
    "BUT",
    "CAN",
    "CIA",  # Central Intelligence Agency - very common
    "DAY",
    "DID",
    "END",
    "FAR",
    "FEW",
    "FOR",
    "GET",
    "GOT",
    "GUY",
    "HAD",
    "HAS",
    "HER",
    "HIM",
    "HIS",
    "HOT",
    "HOW",
    "ITS",
    "JOB",
    "LET",
    "LOT",
    "LOW",
    "MAN",
    "MAY",
    "MEN",
    "NAH",
    "NEW",
    "NOT",
    "NOW",
    "OLD",
    "ONE",
    "OUR",
    "OUT",
    "OWN",
    "PAY",
    "RAN",
    "RUN",
    "SAW",
    "SAY",
    "SEE",
    "SET",
    "SHE",
    "THE",
    "TOO",
    "TOP",
    "TRY",
    "TWO",
    "USE",
    "WAR",
    "WAS",
    "WAY",
    "WHO",
    "WHY",
    "WIN",
    "WON",
    "YET",
    "YOU",
    # 4 letters
    "ALSO",
    "BACK",
    "BEEN",
    "BEST",
    "BOTH",
    "CALL",
    "CAME",
    "COME",
    "DOES",
    "DONE",
    "DOWN",
    "EACH",
    "EDIT",
    "ELSE",
    "EVEN",
    "FACT",
    "FEEL",
    "FIND",
    "FROM",
    "GAVE",
    "GIVE",
    "GOES",
    "GONE",
    "GOOD",
    "HALF",
    "HAND",
    "HARD",
    "HAVE",
    "HEAD",
    "HEAR",
    "HELP",
    "HERE",
    "HIGH",
    "HOLD",
    "HOME",
    "HOPE",
    "HUGE",
    "IDEA",
    "INTO",
    "JUST",
    "KEEP",
    "KIND",
    "KNEW",
    "KNOW",
    "LAST",
    "LATE",
    "LEFT",
    "LESS",
    "LIFE",
    "LIKE",
    "LINE",
    "LIST",
    "LIVE",
    "LONG",
    "LOOK",
    "LOSE",
    "LOSS",
    "LOST",
    "LOVE",
    "MADE",
    "MAKE",
    "MANY",
    "MORE",
    "MOST",
    "MOVE",
    "MUCH",
    "MUST",
    "NAME",
    "NEED",
    "NEXT",
    "NICE",
    "ONLY",
    "OPEN",
    "OVER",
    "PAID",
    "PART",
    "PAST",
    "PICK",
    "PLAY",
    "POST",
    "PULL",
    "PUSH",
    "READ",
    "REAL",
    "REST",
    "RISK",
    "RULE",
    "SAFE",
    "SAID",
    "SAME",
    "SAVE",
    "SEEM",
    "SEEN",
    "SELL",
    "SENT",
    "SHIT",
    "SHOW",
    "SIDE",
    "SOME",
    "SOON",
    "STOP",
    "SUCH",
    "SURE",
    "TAKE",
    "TALK",
    "TELL",
    "THAN",
    "THAT",
    "THEM",
    "THEN",
    "THEY",
    "THIS",
    "THUS",
    "TIME",
    "TOLD",
    "TOOK",
    "TRUE",
    "TURN",
    "VERY",
    "WAIT",
    "WANT",
    "WEEK",
    "WELL",
    "WENT",
    "WERE",
    "WHAT",
    "WHEN",
    "WILL",
    "WITH",
    "WONT",
    "WORD",
    "WORK",
    "YEAR",
    "YOUR",
    # 5 letters
    "ABOUT",
    "AFTER",
    "AGAIN",
    "BEING",
    "COULD",
    "EVERY",
    "FIRST",
    "FOUND",
    "GOING",
    "GREAT",
    "MIGHT",
    "MONEY",
    "NEVER",
    "OTHER",
    "PLACE",
    "POINT",
    "PRICE",
    "RIGHT",
    "SINCE",
    "STILL",
    "STOCK",
    "THEIR",
    "THERE",
    "THESE",
    "THING",
    "THINK",
    "THOSE",
    "THREE",
    "TODAY",
    "UNDER",
    "WATCH",
    "WHERE",
    "WHICH",
    "WHILE",
    "WORLD",
    "WOULD",
    "YEARS",
    # ----------------------------------------------------------
    # Reddit/internet abbreviations
    # ----------------------------------------------------------
    "AMP",
    "CEO",
    "CFO",
    "COO",
    "CTO",
    "DD",
    "DM",
    "DP",
    "EDI",
    "EG",
    "EOD",
    "ETA",
    "ETC",
    "FAQ",
    "FYI",
    "GIF",
    "GPS",
    "GT",
    "HTML",
    "HTTP",
    "IE",
    "IIRC",
    "IMHO",
    "IMG",
    "IMO",
    "IRL",
    "ISP",
    "JK",
    "LMAO",
    "LOL",
    "NBC",
    "NGL",
    "NFT",
    "NPC",
    "NSFW",
    "OMG",
    "OP",
    "OC",
    "PDF",
    "PM",
    "PS",
    "ROFL",
    "SMH",
    "TBD",
    "TBH",
    "TIL",
    "TL",
    "TLDR",
    "TMI",
    "USA",
    "USD",
    "UTC",
    "VS",
    "WTF",
    "WWW",
    "XML",
    "YOLO",
    # ----------------------------------------------------------
    # Finance jargon
    # ----------------------------------------------------------
    "APE",
    "APES",
    "ATH",
    "ATM",
    "BAG",
    "BAGS",
    "BEAR",
    "BOT",
    "BOTS",
    "BREAK",
    "BULL",
    "CALLS",
    "CHART",
    "CRAP",
    "DIP",
    "DIPS",
    "ENTRY",
    "FED",
    "FUD",
    "FOMO",
    "GAIN",
    "GAINS",
    "GG",
    "GREEN",
    "HODL",
    "IPO",
    "ITM",
    "IV",
    "LEVEL",
    "MOON",
    "OTM",
    "PDT",
    "PUMP",
    "PUTS",
    "RALLY",
    "RH",
    "SEC",
    "SHARE",
    "SHORT",
    "SPAM",
    "SPAC",
    "STONK",
    "TD",
    "TRADE",
    "TREND",
    "WSB",
    # Finance/trading abbreviations (not stock symbols)
    "AKA",
    "BTO",  # Buy To Open
    "CPT",  # Consistently Profitable Trader
    "DMA",  # Direct Market Access
    "DRS",  # Direct Registration System
    "DTE",  # Days to Expiration
    "DYOR",  # Do Your Own Research
    "EFC",
    "ETB",  # Easy To Borrow
    "EXP",  # Expiration (options)
    "FCF",  # Free Cash Flow
    "FOUR",
    "HIT",
    "HTB",  # Hard To Borrow
    "MSM",  # Mainstream Media
    "NPV",  # Net Present Value
    "PRE",  # pre-market
    "PSA",  # Public Service Announcement
    "STC",  # Sell To Close
    "EPS",
    "PE",
    "ROI",
    "YOY",
    "QOQ",
    "MOM",
    "YTD",
    "NAV",
    "AUM",
    # Technical analysis indicators
    "ATR",
    "SMC",  # Smart Money Concept
    "EMA",
    "SMA",
    "MACD",
    "VWAP",
    "ADX",
    "CCI",
    "OBV",
    "MFI",
    "CMF",
    "ADL",
    "DMI",
    "SAR",
    "ICT",  # Inner Circle Trader
    # ----------------------------------------------------------
    # Ambiguous symbols (countries, tech terms, etc.) ≤5 chars
    # ----------------------------------------------------------
    "UK",
    "EU",
    "CD",
    "IRS",
    "TV",
    "NYC",
    "IP",
    "PC",
    "TFSA",
    "XYZ",
    "WTI",
    "RSI",
    "FT",
    "UI",
    "API",
    "PR",
    "CPA",
    "CC",
    "FA",
    "EM",
    "SF",
    "IQ",
    "SA",
    "AR",
    "HR",
    "OS",
    # Hardware/tech terms
    "SSD",
    "RTX",  # NVIDIA GPU
    "GTX",  # NVIDIA GPU
    "GPU",
    "CPU",
    "RAM",
    "USB",
    "LED",
    "LCD",
    "ADC",
    "DAC",
    "CSV",
    "JSON",
    "SQL",
    "CLS",
    "ATS",  # Alternative Trading System
    "IEX",  # Investors Exchange
    # Medical/institutional
    "ICU",
    "ER",
    "OR",
    "CNS",
    # Ambiguous words that are also tickers
    "WAVE",  # "Elliott Wave" in TA
    "BEAT",
    "INFO",
    "KEY",
    "EARN",
    "LITE",
    "EFT",  # Typo for ETF
    "ULY",
    # ----------------------------------------------------------
    # Common first names that are tickers
    # ----------------------------------------------------------
    "RYAN",
    "BILL",
    "JACK",
    "ADAM",
    "JOHN",
    "MIKE",
    "MARK",
    "DAVE",
    # ----------------------------------------------------------
    # Other common words that are also tickers
    # ----------------------------------------------------------
    "BALL",  # Crystal ball, drop the ball, etc.
    "FIVE",  # Number in caps-lock posts
    "GOLD",  # Metal, not Barrick Gold
    "IBKR",  # Interactive Brokers - broker discussion
    "MLM",
    "NET",  # Netto, net gain, etc.
    "POOL",  # Financial pools, liquidity pool
    "SOS",
    "III",
    "DISH",
    "DOOR",
    "EDGE",
    "FLEX",
    "FORM",
    "KEYS",
    "LAKE",
    "LAND",
    "LEAP",
    "MAIN",
    "MAX",
    "NEXT",
    "NICE",
    "NOVA",
    "OPEN",
    "PARK",
    "PATH",
    "PEAK",
    "PLUS",
    "PLUG",
    "PURE",
    "ROOT",
    "SAGE",
    "SALT",
    "SKY",
    "SOLO",
    "STAR",
    "STEM",
    "STEP",
    "STORE",
    "SUN",
    "TREE",
    "TRIP",
    "ULTRA",
    "VIA",
    "VIEW",
    "WEST",
    "WING",
    "WISH",
    "WOLF",
    "ZONE",
    "ASX",  # Australian Stock Exchange
    # ----------------------------------------------------------
    # Dictionary-checked false positives (new audit)
    # Common English words whose mention counts are implausible
    # for the underlying companies. Genuine references via $-prefix
    # cashtag are still captured in Phase 1.
    # ----------------------------------------------------------
    "WEN",  # Wendy's — "wen moon" / "wen lambo" Reddit slang
    "EVER",  # EverQuote (micro-cap) — ubiquitous word
    "BOOM",  # DMC Global (micro-cap) — exclamation
    "GAME",  # GameSquare Holdings (micro-cap) — common noun
    "HOUR",  # Hour Loop (micro-cap) — time word
    "CASH",  # Pathward Financial — finance vocabulary
    "MOD",  # Modine Manufacturing — "moderator" on Reddit
    "RUM",  # Rumble Inc — drink name
    "FLY",  # Firefly Aerospace — common verb
    "KEN",  # Kenon Holdings — first name / Ken Griffin memes
    "FUN",  # Six Flags — ubiquitous adjective
    "FAT",  # FAT Brands (micro-cap) — common adjective
    "NOTE",  # FiscalNote (micro-cap) — common noun
    "SHO",  # Sunstone Hotel Investors — slang for "sure"
    "RIG",  # Transocean — "rigged" / "oil rig" in market context
    "CARE",  # Carter Bankshares — common verb
    "AGO",  # Assured Guaranty — time word ("3 years AGO")
    "SON",  # Sonoco Products — family word
    "CALM",  # Cal-Maine Foods — emotional descriptor
    "MIND",  # MIND Technology (nano-cap) — common noun
    "MET",  # MetLife (S&P 500) — past tense of "meet"
    "LUCK",  # Lucky Strike Entertainment — "good luck" everywhere
    "EML",  # Eastern Company (small-cap) — email abbreviation
    "DOW",  # Dow Inc — "the DOW" mostly references DJIA index
    "UPS",  # United Parcel Service — "ups and downs" + shipping
}

# Common cryptocurrency tickers to exclude
# These are not stocks and should not be counted
# Patterns to identify closed-end funds (CEFs)
# These are excluded because they are funds, not operating companies
_CEF_PATTERNS = [
    r"\bCLOSED[- ]?END\b",
    r"\bCEF\b",
    r"\bFUND\b",  # Generic fund (but not "FUNDED" or company names ending in fund-like words)
    r"\bINVESTMENT TRUST\b",
    r"\bMUNICIPAL\s+(INCOME|BOND|FUND)\b",
    r"\bINCOME FUND\b",
    r"\bEQUITY FUND\b",
    r"\bBOND FUND\b",
    r"\bOPPORTUNITIES FUND\b",
    r"\bDIVIDEND FUND\b",
    r"\bGLOBAL FUND\b",
    r"\bSTRATEGIC FUND\b",
]
_CEF_REGEX = re.compile("|".join(_CEF_PATTERNS), re.IGNORECASE)


def _is_closed_end_fund(raw_name: str) -> bool:
    """Check if a security name indicates a closed-end fund."""
    if not raw_name:
        return False
    return bool(_CEF_REGEX.search(raw_name))


CRYPTO_TICKERS: Set[str] = {
    "BTC",
    "ETH",
    "DOGE",
    "XRP",
    "ADA",
    "SOL",
    "DOT",
    "AVAX",
    "MATIC",
    "LINK",
    "SHIB",
    "LTC",
    "BCH",
    "UNI",
    "ATOM",
    "XLM",
    "VET",
    "FIL",
    "ALGO",
    "MANA",
    "SAND",
    "AXS",
    "AAVE",
    "COMP",
    "MKR",
    "SUSHI",
    "YFI",
    "SNX",
    "CRV",
    "BAL",
    "BNB",
    "USDT",
    "USDC",
    "BUSD",
    "DAI",
    "WBTC",
}


@dataclass
class TickerInfo:
    """Information about a stock ticker.

    Attributes:
        symbol: The stock ticker symbol (e.g., 'AAPL', 'MSFT')
        company_names: List of company name variants for matching
        is_etf: Whether this is an ETF (we exclude these)
    """

    symbol: str
    company_names: List[str]
    is_etf: bool


@dataclass
class TickerUniverse:
    """Container for all loaded tickers with efficient lookup structures.

    Attributes:
        by_symbol: Map from ticker symbol to TickerInfo
        all_symbols: Set of all valid ticker symbols
        name_to_symbol: Map from normalized company name to (symbol, original_name)
        name_automaton: Aho-Corasick automaton for O(n) name matching
    """

    by_symbol: Dict[str, TickerInfo] = field(default_factory=dict)
    all_symbols: Set[str] = field(default_factory=set)
    name_to_symbol: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    name_automaton: Any = None


def load_clean_tickers(path: str) -> Iterator[TickerInfo]:
    """Load tickers from clean_tickers.csv.

    Filters out:
    - ETFs (is_etf = 'Y')
    - Blacklisted symbols
    - Non-alphabetic symbols (no numbers or special chars)
    - Cryptocurrency tickers

    Args:
        path: Path to the clean_tickers.csv file

    Yields:
        TickerInfo objects for each valid ticker
    """
    # Check if file exists
    if not os.path.exists(path):
        print(f"[WARN] Clean tickers file not found: {path}")
        return

    # Open and parse CSV
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Extract and normalize symbol
            symbol = row.get("ticker", "").strip().upper()
            names_raw = row.get("name", "").strip()
            is_etf_raw = row.get("is_etf", "N").strip().upper()

            # Skip empty symbols
            if not symbol:
                continue

            # Skip ETFs
            is_etf = is_etf_raw == "Y"
            if is_etf:
                continue

            # Skip symbols with numbers or special characters
            if not symbol.isalpha():
                continue

            # Skip cryptocurrency tickers
            if symbol in CRYPTO_TICKERS:
                continue

            # Skip closed-end funds (detect by name patterns)
            # We keep REITs, BDCs, and SPACs as they are proper stocks
            raw_name = row.get("raw_name", "").upper()
            if _is_closed_end_fund(raw_name):
                continue

            # Parse company names (pipe-separated)
            names = [n.strip() for n in names_raw.split("|") if n.strip()]

            # Yield the ticker info
            yield TickerInfo(
                symbol=symbol,
                company_names=names,
                is_etf=is_etf,
            )


def build_ticker_universe(tickers_file: str = CLEAN_TICKERS_FILE) -> TickerUniverse:
    """Load all tickers and build lookup structures.

    Creates:
    - Symbol lookup dictionary
    - Set of all valid symbols
    - Company name to symbol mapping
    - Aho-Corasick automaton for efficient name matching

    Args:
        tickers_file: Path to clean_tickers.csv

    Returns:
        Fully initialized TickerUniverse object
    """
    # Initialize empty universe
    universe = TickerUniverse()

    print(f"Loading tickers from {tickers_file}...")

    # Counters for progress reporting
    ticker_count = 0
    name_count = 0

    # Load each ticker from the CSV
    for info in load_clean_tickers(tickers_file):
        # Skip duplicates
        if info.symbol in universe.by_symbol:
            continue

        # Add to symbol lookup
        universe.by_symbol[info.symbol] = info
        universe.all_symbols.add(info.symbol)
        ticker_count += 1

        # Register each company name variant
        for name in info.company_names:
            # Normalize: uppercase and collapse whitespace
            normalized = " ".join(name.upper().split())

            # Skip very short names (too many false positives)
            if len(normalized) < 3:
                continue

            # Only keep first mapping for each name
            if normalized not in universe.name_to_symbol:
                universe.name_to_symbol[normalized] = (info.symbol, name)
                name_count += 1

    print(f"  Loaded {ticker_count} stock tickers (ETFs excluded)")
    print(f"  Registered {name_count} company name variants for matching")

    # Build Aho-Corasick automaton for O(n) multi-pattern matching
    print("  Building Aho-Corasick automaton for name matching...")
    automaton = ahocorasick.Automaton()

    # Add each name pattern to the automaton.
    # Note: the bare-caps stoplist is NOT applied here. Company names (e.g.
    # "Barrick Gold", "MetLife") are unambiguous multi-word patterns that do
    # not suffer from the same false-positive problem as bare ALL-CAPS tokens.
    # Name-level filtering is already handled by generate_clean_tickers.py
    # (common-word removal, short-name filtering).
    for normalized_name, (ticker, original_name) in universe.name_to_symbol.items():
        automaton.add_word(normalized_name, (normalized_name, ticker, original_name))

    # Finalize the automaton for searching
    automaton.make_automaton()
    universe.name_automaton = automaton
    print(f"  Automaton ready with {len(automaton)} patterns")

    # Print stats about short tickers
    short_1 = sum(1 for s in universe.all_symbols if len(s) == 1)
    short_2 = sum(1 for s in universe.all_symbols if len(s) == 2)
    print(f"  1-letter tickers (require $ prefix): {short_1}")
    print(f"  2-letter tickers (require $ prefix): {short_2}")
    print(f"  Stoplist size: {len(STOPLIST_TOKENS)}")

    return universe
