"""Daily return and volatility calculations.

This module computes daily log returns and a battery of daily volatility
estimators from raw OHLC price data.

Baseline estimator
------------------
The baseline daily volatility is the *jump-adjusted Garman--Klass* (JA-GK)
estimator, computed as the sum of two non-overlapping variance components:

    sigma^2_{JA-GK,t} = sigma^2_{overnight,t} + sigma^2_{GK,t}

where

    sigma^2_{overnight,t} = [ln(O_t / C_{t-1})]^2

is the squared overnight log return (capturing the close-to-open jump) and

    sigma^2_{GK,t} = 0.5 * [ln(H_t / L_t)]^2 - (2 ln 2 - 1) * [ln(C_t / O_t)]^2

is the Garman--Klass (1980) intraday range estimator.  This sum is exactly
the *jump-adjusted Garman--Klass estimator* of Molnár (2012, eq. 36), which
he derives and recommends as the way to apply Garman--Klass to assets with
overnight gaps.  Garman--Klass is the minimum-variance estimator within the
analytical class of range-based estimators (those expressible as a closed-form
function of OHLC) per Garman and Klass (1980).

Robustness estimators
---------------------
For each stock-day, we additionally compute five robustness alternatives so
that downstream regressions can vary the volatility target without re-running
the price pipeline:

    volatility_ja_rs : jump-adjusted Rogers--Satchell (1991)
    volatility_ja_pk : jump-adjusted Parkinson (1980)
    volatility_gk    : pure Garman--Klass (no overnight component)
    volatility_pk    : pure Parkinson (no overnight component)
    volatility_sqret : |r_t|, i.e. squared close-to-close log return as variance proxy

The first two vary the *intraday* estimator while preserving the
jump-adjustment structure; the latter three drop the overnight
component entirely and serve as benchmarks against the no-overnight
literature (cf. Molnár, 2012).

References
----------
Garman, M.B. and Klass, M.J. (1980). On the estimation of security price
    volatilities from historical data. Journal of Business 53(1), 67-78.
Molnár, P. (2012). Properties of range-based volatility estimators.
    International Review of Financial Analysis 23, 20-29.
Parkinson, M. (1980). The extreme value method for estimating the variance of
    the rate of return. Journal of Business 53(1), 61-65.
Rogers, L.C.G. and Satchell, S.E. (1991). Estimating variance from high, low
    and closing prices. Annals of Applied Probability 1(4), 504-512.
Yang, D. and Zhang, Q. (2000). Drift-independent volatility estimation based
    on high, low, open, and close prices. Journal of Business 73(3), 477-492.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from config import TRADING_DAYS_PER_YEAR

# ---------------------------------------------------------------------------
# Constants used by the range-based estimators
# ---------------------------------------------------------------------------
_LN2 = np.log(2.0)
_GK_C_COEF = 2.0 * _LN2 - 1.0  # coefficient on the close-to-open term in GK
_PK_DENOM = 4.0 * _LN2  # denominator of the Parkinson estimator


def _add_intraday_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add the log-price components used by all range-based estimators.

    Adds the columns o, c, u, d, where for each row

        o = ln(O_t / C_{t-1})   overnight return
        c = ln(C_t / O_t)       open-to-close return
        u = ln(H_t / O_t)       high-to-open log-price ratio
        d = ln(L_t / O_t)       low-to-open log-price ratio
    """
    df = df.copy()
    if "ticker" in df.columns:
        df["prev_close"] = df.groupby("ticker")["close"].shift(1)
    else:
        df["prev_close"] = df["close"].shift(1)
    df["o"] = np.log(df["open"] / df["prev_close"])
    df["c"] = np.log(df["close"] / df["open"])
    df["u"] = np.log(df["high"] / df["open"])
    df["d"] = np.log(df["low"] / df["open"])
    return df


def _compute_variance_components(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the daily variance components from the log-price ratios.

    Adds the columns:
        var_overnight : sigma^2_{overnight,t} = o^2
        var_gk        : sigma^2_{GK,t}        (Garman--Klass intraday)
        var_rs        : sigma^2_{RS,t}        (Rogers--Satchell intraday)
        var_pk        : sigma^2_{PK,t}        (Parkinson intraday)
    """
    df = df.copy()
    log_hl = df["u"] - df["d"]  # = ln(H/L)

    df["var_overnight"] = df["o"] ** 2
    df["var_gk"] = 0.5 * log_hl**2 - _GK_C_COEF * df["c"] ** 2
    df["var_rs"] = df["u"] * (df["u"] - df["c"]) + df["d"] * (df["d"] - df["c"])
    df["var_pk"] = (log_hl**2) / _PK_DENOM
    return df


def _assemble_volatilities(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Build the six daily volatility columns from the variance components.

    Args:
        df: DataFrame with var_overnight, var_gk, var_rs, var_pk and a
            log_return column (squared returns) already present.
        prefix: Optional prefix for the output columns (e.g. "market_").

    Adds the six daily-volatility columns; non-finite or negative variances
    (e.g. from numerical noise in Rogers--Satchell or Garman--Klass) are
    floored at zero before taking the square root.
    """
    df = df.copy()

    var_ja_gk = df["var_overnight"] + df["var_gk"]
    var_ja_rs = df["var_overnight"] + df["var_rs"]
    var_ja_pk = df["var_overnight"] + df["var_pk"]
    var_gk = df["var_gk"]
    var_pk = df["var_pk"]
    var_sqret = df["log_return"] ** 2

    df[f"{prefix}volatility_ja_gk"] = np.sqrt(np.maximum(var_ja_gk, 0.0))
    df[f"{prefix}volatility_ja_rs"] = np.sqrt(np.maximum(var_ja_rs, 0.0))
    df[f"{prefix}volatility_ja_pk"] = np.sqrt(np.maximum(var_ja_pk, 0.0))
    df[f"{prefix}volatility_gk"] = np.sqrt(np.maximum(var_gk, 0.0))
    df[f"{prefix}volatility_pk"] = np.sqrt(np.maximum(var_pk, 0.0))
    df[f"{prefix}volatility_sqret"] = np.sqrt(np.maximum(var_sqret, 0.0))
    return df


# Public column lists ------------------------------------------------------
DAILY_VOLATILITY_COLUMNS = [
    "volatility_ja_gk",  # baseline: jump-adjusted Garman-Klass
    "volatility_ja_rs",  # robustness: jump-adjusted Rogers-Satchell
    "volatility_ja_pk",  # robustness: jump-adjusted Parkinson
    "volatility_gk",  # robustness: pure Garman-Klass
    "volatility_pk",  # robustness: pure Parkinson
    "volatility_sqret",  # robustness: squared close-to-close log return
]

MARKET_VOLATILITY_COLUMN = "market_volatility_ja_gk"


def calculate_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns and the full battery of daily volatility measures.

    All variance components use raw (unadjusted) OHLC prices to preserve the
    intraday price relationships on which the range-based estimators rely.
    Log returns are computed from adjusted close prices so that they reflect
    true holding-period returns.

    Args:
        prices_df: DataFrame with columns
            [date, ticker, open, high, low, close, adj_close]

    Returns:
        DataFrame with columns
            [date, ticker, open, high, low, close, adj_close,
             simple_return, log_return,
             volatility_ja_gk, volatility_ja_rs, volatility_ja_pk,
             volatility_gk, volatility_pk, volatility_sqret]
    """
    if prices_df.empty:
        return pd.DataFrame()

    df = prices_df.sort_values(["ticker", "date"]).copy()

    for col in ["open", "high", "low", "close", "adj_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Returns (based on adjusted close for dividend/split adjustment)
    df["simple_return"] = df.groupby("ticker")["adj_close"].pct_change()
    df["log_return"] = df.groupby("ticker")["adj_close"].transform(
        lambda x: np.log(x / x.shift(1))
    )

    # Range-based variance components (raw OHLC)
    df = _add_intraday_components(df)
    df = _compute_variance_components(df)

    # Six daily volatility measures
    df = _assemble_volatilities(df)

    output_cols = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "simple_return",
        "log_return",
        *DAILY_VOLATILITY_COLUMNS,
    ]
    result = df[output_cols].copy()

    # Drop rows without valid returns (first row per ticker)
    return result.dropna(subset=["simple_return"])


def calculate_benchmark_returns(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market benchmark returns and the baseline daily volatility.

    Only the baseline JA-GK estimator is computed for the market control;
    downstream code uses the market series purely as a single aggregate
    volatility regressor.

    Args:
        benchmark_df: DataFrame with columns
            [date, open, high, low, close, adj_close]

    Returns:
        DataFrame with columns
            [date, market_close, market_return, market_log_return,
             market_volatility_ja_gk]
    """
    if benchmark_df.empty:
        return pd.DataFrame()

    df = benchmark_df.sort_values("date").copy()

    for col in ["open", "high", "low", "close", "adj_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["market_return"] = df["adj_close"].pct_change()
    df["market_log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))

    df = _add_intraday_components(df)
    df = _compute_variance_components(df)

    var_ja_gk = df["var_overnight"] + df["var_gk"]
    df[MARKET_VOLATILITY_COLUMN] = np.sqrt(np.maximum(var_ja_gk, 0.0))

    result = df[
        [
            "date",
            "adj_close",
            "market_return",
            "market_log_return",
            MARKET_VOLATILITY_COLUMN,
        ]
    ].copy()
    result = result.rename(columns={"adj_close": "market_close"})

    return result.dropna(subset=["market_return"])


def annualize_volatility(daily_vol: float | pd.Series) -> float | pd.Series:
    """Convert daily volatility to annualized volatility.

    Args:
        daily_vol: Daily volatility (sigma_daily)

    Returns:
        Annualized volatility: sigma_annual = sigma_daily * sqrt(252)
    """
    return daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
