"""S&P 500 constituents and GICS sector mapping."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from config import (
    CACHE_DIR,
    CACHE_MAX_AGE_HOURS,
    ETF_TO_SECTOR,
    SECTOR_ETFS,
    SP500_WIKI_URL,
    WIKI_USER_AGENT,
)

UNIVERSE_PATH = CACHE_DIR / "universe.parquet"
UNIVERSE_META_PATH = CACHE_DIR / "universe_meta.json"


def _cache_is_fresh(meta_path: Path, max_age_hours: float = CACHE_MAX_AGE_HOURS) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(meta["fetched_at"])
        age_hours = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
        return age_hours < max_age_hours
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return False


def _yahoo_symbol(symbol: str) -> str:
    return symbol.strip().replace(".", "-")


def _write_meta(path: Path) -> None:
    path.write_text(
        json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )


def fetch_sp500_universe() -> pd.DataFrame:
    """Download the S&P 500 list from Wikipedia and map each name to its sector ETF."""
    response = requests.get(
        SP500_WIKI_URL,
        headers={"User-Agent": WIKI_USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise RuntimeError("Wikipedia returned no tables for the S&P 500 list.")

    raw = tables[0]
    columns = {str(c).strip().lower(): c for c in raw.columns}

    def col(*names: str) -> str:
        for name in names:
            if name in columns:
                return columns[name]
        raise KeyError(f"Missing expected column among {names}; got {list(raw.columns)}")

    symbol_col = col("symbol")
    name_col = col("security")
    sector_col = col("gics sector")
    sub_col = col("gics sub-industry", "gics sub industry")

    frame = pd.DataFrame(
        {
            "ticker": raw[symbol_col].astype(str).map(_yahoo_symbol),
            "name": raw[name_col].astype(str).str.strip(),
            "sector": raw[sector_col].astype(str).str.strip(),
            "sub_industry": raw[sub_col].astype(str).str.strip(),
        }
    )
    frame = frame.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    frame["etf"] = frame["sector"].map(SECTOR_ETFS)
    unknown = frame[frame["etf"].isna()]["sector"].unique().tolist()
    if unknown:
        frame = frame.dropna(subset=["etf"]).reset_index(drop=True)
    return frame


def load_universe(force: bool = False) -> pd.DataFrame:
    """Return ticker / name / GICS sector / sector ETF, using a Parquet cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if (
        not force
        and UNIVERSE_PATH.exists()
        and _cache_is_fresh(UNIVERSE_META_PATH)
    ):
        cached = pd.read_parquet(UNIVERSE_PATH)
        if "sub_industry" in cached.columns:
            return cached

    try:
        universe = fetch_sp500_universe()
        universe.to_parquet(UNIVERSE_PATH, index=False)
        _write_meta(UNIVERSE_META_PATH)
        return universe
    except Exception:
        if UNIVERSE_PATH.exists():
            return pd.read_parquet(UNIVERSE_PATH)
        raise


def sector_etf_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"etf": etf, "sector": sector} for etf, sector in ETF_TO_SECTOR.items()]
    )
