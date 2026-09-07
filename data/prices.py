"""Download and cache daily OHLCV from yfinance."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import (
    ALL_ETFS,
    CACHE_DIR,
    CACHE_MAX_AGE_HOURS,
    DOWNLOAD_CHUNK_SIZE,
    PRICE_HISTORY,
)

ETF_PRICES_PATH = CACHE_DIR / "etf_prices.parquet"
STOCK_PRICES_PATH = CACHE_DIR / "stock_prices.parquet"
ETF_META_PATH = CACHE_DIR / "etf_prices_meta.json"
STOCK_META_PATH = CACHE_DIR / "stock_prices_meta.json"


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


def _write_meta(path: Path, extra: dict | None = None) -> None:
    payload = {"fetched_at": datetime.now(timezone.utc).isoformat()}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _chunks(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _normalize_dates(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return dates.dt.normalize()


def _extract_ticker_ohlcv(data: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)
        if ticker in level0:
            sub = data[ticker].copy()
        elif ticker in level1:
            sub = data.xs(ticker, axis=1, level=1).copy()
        else:
            return None
    else:
        sub = data.copy()

    sub.columns = [str(c).strip().lower() for c in sub.columns]
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(set(sub.columns)):
        return None
    sub = sub.reset_index()
    lower_map = {str(c).lower(): c for c in sub.columns}
    date_col = lower_map.get("date", sub.columns[0])
    sub = sub.rename(columns={date_col: "date"})
    sub.columns = [str(c).strip().lower() for c in sub.columns]
    sub["date"] = _normalize_dates(sub["date"])
    sub["ticker"] = ticker
    out = sub[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
    out = out.dropna(subset=["close"])
    if out.empty:
        return None
    return out


def download_ohlcv(tickers: list[str], chunk_size: int = DOWNLOAD_CHUNK_SIZE) -> pd.DataFrame:
    """Download adjusted OHLCV and return a long DataFrame."""
    unique = list(dict.fromkeys(tickers))
    frames: list[pd.DataFrame] = []
    for chunk in _chunks(unique, chunk_size):
        data = yf.download(
            tickers=chunk,
            period=PRICE_HISTORY,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if data is None or data.empty:
            time.sleep(0.4)
            continue
        for ticker in chunk:
            parsed = _extract_ticker_ohlcv(data, ticker)
            if parsed is not None:
                frames.append(parsed)
        time.sleep(0.25)
    if not frames:
        raise RuntimeError("No price data downloaded from yfinance.")
    prices = pd.concat(frames, ignore_index=True)
    prices["volume"] = prices["volume"].fillna(0)
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def _load_cached_or_download(
    tickers: list[str],
    cache_path: Path,
    meta_path: Path,
    force: bool,
) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not force and cache_path.exists() and _cache_is_fresh(meta_path):
        return pd.read_parquet(cache_path)
    try:
        prices = download_ohlcv(tickers)
        prices.to_parquet(cache_path, index=False)
        _write_meta(meta_path, {"tickers": len(tickers), "rows": int(len(prices))})
        return prices
    except Exception:
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        raise


def load_etf_prices(force: bool = False) -> pd.DataFrame:
    return _load_cached_or_download(list(ALL_ETFS), ETF_PRICES_PATH, ETF_META_PATH, force)


def load_stock_prices(tickers: list[str], force: bool = False) -> pd.DataFrame:
    return _load_cached_or_download(tickers, STOCK_PRICES_PATH, STOCK_META_PATH, force)


def pivot_field(prices: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Wide panel: dates x tickers."""
    wide = prices.pivot(index="date", columns="ticker", values=field).sort_index()
    return wide


def latest_as_of(prices: pd.DataFrame) -> pd.Timestamp:
    return pd.to_datetime(prices["date"]).max()
