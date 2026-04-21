"""Data loading and preparation for Panel-GARCH-X estimation.

Loads the existing panel data (built by the regression pipeline) and
prepares it for GARCH estimation: aligns returns and exogenous variables,
handles missing data, and splits into per-stock arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from config import PANEL_PARQUET, VIX_CACHE, GARCHSpec


@dataclass
class StockData:
    """Prepared time-series data for a single stock."""

    ticker: str
    dates: np.ndarray  # (T,) datetime
    returns: np.ndarray  # (T,) log returns
    exog: np.ndarray  # (T, K) exogenous variables (already lagged)
    exog_names: List[str]
    sample_var: float  # unconditional sample variance
    n_obs: int

    def __repr__(self) -> str:
        return f"StockData({self.ticker}, T={self.n_obs}, K={len(self.exog_names)})"


def load_panel(
    spec: GARCHSpec,
    panel_path: Optional[Path] = None,
    min_obs: int = 100,
) -> Tuple[List[StockData], pd.DataFrame]:
    """Load panel data and prepare per-stock arrays for GARCH estimation.

    Parameters
    ----------
    spec : GARCHSpec
        Model specification (determines which columns to load).
    panel_path : Path, optional
        Path to panel parquet. Defaults to PANEL_PARQUET from config.
    min_obs : int
        Minimum observations per stock after dropping NaN rows.

    Returns
    -------
    stocks : list of StockData
        Per-stock prepared data, sorted by ticker.
    panel : pd.DataFrame
        The raw loaded panel (for diagnostics).
    """
    path = panel_path or PANEL_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"Panel data not found at {path}.\n"
            "Run the regression pipeline first: python regression/src/build_panel.py"
        )

    panel = pd.read_parquet(path)

    # Ensure proper index
    if "ticker" in panel.columns and "date" in panel.columns:
        panel = panel.set_index(["ticker", "date"]).sort_index()
    elif panel.index.names != ["ticker", "date"]:
        raise ValueError(f"Expected (ticker, date) index, got {panel.index.names}")

    # Identify required columns
    required = [spec.return_col] + spec.exog_cols
    missing = [c for c in required if c not in panel.columns and c != "vix_std"]
    if missing:
        raise ValueError(
            f"Columns missing from panel data: {missing}\n"
            f"Available: {sorted(panel.columns.tolist())}"
        )

    # Merge VIX if requested
    if "vix_std" in spec.exog_cols:
        vix = _load_vix(panel)
        panel = panel.join(vix, on="date")

    # Select relevant columns
    cols = [spec.return_col] + spec.exog_cols
    df = panel[cols].copy()

    # Drop rows with NaN/inf in returns (required) — fill NaN exog with 0
    # Social media variables: NaN means no posts → 0 sentiment / 0 attention
    # The valid_sentiment indicator (also in exog) flags these days so the
    # GARCH model can distinguish "neutral sentiment" from "no posts" via
    # its own dedicated coefficient psi_i (mirrors the OLS ValidSent control)
    df = df.dropna(subset=[spec.return_col])
    df = df[np.isfinite(df[spec.return_col])]
    for col in spec.exog_cols:
        df[col] = df[col].fillna(0.0)
        df[col] = df[col].replace([np.inf, -np.inf], 0.0)

    # Build per-stock data
    stocks: List[StockData] = []
    for ticker, group in df.groupby(level="ticker"):
        group = group.droplevel("ticker").sort_index()

        if len(group) < min_obs:
            continue

        returns = np.asarray(group[spec.return_col].values, dtype=np.float64)
        exog = group[spec.exog_cols].values.astype(np.float64)
        dates = group.index.values

        stocks.append(
            StockData(
                ticker=str(ticker),
                dates=dates,
                returns=returns,
                exog=exog,
                exog_names=list(spec.exog_cols),
                sample_var=float(np.var(returns, ddof=1)),
                n_obs=len(returns),
            )
        )

    stocks.sort(key=lambda s: s.ticker)

    if not stocks:
        raise ValueError(f"No stocks with >= {min_obs} observations after cleaning.")

    return stocks, panel


def _load_vix(panel: pd.DataFrame) -> pd.Series:
    """Load VIX closing prices, standardize to z-score, return as Series indexed by date."""
    if VIX_CACHE.exists():
        vix_df = pd.read_parquet(VIX_CACHE)
    else:
        import yfinance as yf

        dates = pd.to_datetime(panel.index.get_level_values("date"))
        start = str(dates.min().date())
        end = str(dates.max().date())
        vix_raw = yf.download("^VIX", start=start, end=end, auto_adjust=True)
        if vix_raw is None or vix_raw.empty:
            raise ValueError("Failed to download VIX data")
        vix_df = vix_raw[["Close"]].rename(columns={"Close": "vix"})
        vix_df.index.name = "date"
        # Flatten MultiIndex columns if yfinance returns them
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        VIX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        vix_df.to_parquet(VIX_CACHE)
        print(f"  VIX data cached to {VIX_CACHE}", flush=True)

    # Standardize to z-score
    vix_df["vix_std"] = (vix_df["vix"] - vix_df["vix"].mean()) / vix_df["vix"].std()
    vix_df.index = vix_df.index.astype(str)
    return vix_df["vix_std"]


def summary_stats(stocks: List[StockData]) -> pd.DataFrame:
    """Compute summary statistics across the panel."""
    rows = []
    for s in stocks:
        rows.append(
            {
                "ticker": s.ticker,
                "n_obs": s.n_obs,
                "mean_return": np.mean(s.returns),
                "std_return": np.std(s.returns, ddof=1),
                "sample_var": s.sample_var,
                "min_return": np.min(s.returns),
                "max_return": np.max(s.returns),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")
