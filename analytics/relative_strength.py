"""Multi-horizon relative strength versus SPY."""

from __future__ import annotations

import pandas as pd

from config import BENCHMARK, HORIZONS, SECTOR_ETFS, SECTOR_LABELS
from data.prices import pivot_field


def relative_strength(
    close: pd.DataFrame,
    items: list[dict[str, str]],
    benchmark: str = BENCHMARK,
) -> pd.DataFrame:
    """items: dicts with etf (id), sector (label), and optional gics/parent_etf."""
    if benchmark not in close.columns:
        raise ValueError(f"Benchmark {benchmark} missing from prices.")
    bench = close[benchmark]
    rows = []
    for item in items:
        ticker = item["etf"]
        if ticker not in close.columns:
            continue
        series = close[ticker]
        row: dict[str, float | str] = {
            "etf": ticker,
            "sector": item.get("sector", ticker),
            "gics": item.get("gics", item.get("sector", ticker)),
        }
        if "parent_etf" in item:
            row["parent_etf"] = item["parent_etf"]
        for name, periods in HORIZONS.items():
            if len(series.dropna()) <= periods or len(bench.dropna()) <= periods:
                abs_ret = float("nan")
                bench_ret = float("nan")
            else:
                abs_ret = float(series.iloc[-1] / series.iloc[-1 - periods] - 1)
                bench_ret = float(bench.iloc[-1] / bench.iloc[-1 - periods] - 1)
            row[f"ret_{name}"] = abs_ret
            row[f"rs_{name}"] = abs_ret - bench_ret
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["rs_1m_rank"] = frame["rs_1m"].rank(ascending=False, method="min")
    return frame.sort_values("rs_1m", ascending=False).reset_index(drop=True)


def sector_relative_strength(etf_prices: pd.DataFrame, benchmark: str = BENCHMARK) -> pd.DataFrame:
    close = pivot_field(etf_prices, "close")
    items = [
        {
            "etf": etf,
            "sector": SECTOR_LABELS.get(etf, etf),
            "gics": gics,
        }
        for gics, etf in SECTOR_ETFS.items()
        if etf in close.columns
    ]
    return relative_strength(close, items, benchmark=benchmark)


def style_returns(etf_prices: pd.DataFrame, benchmark: str = BENCHMARK) -> dict[str, float]:
    close = pivot_field(etf_prices, "close")
    out: dict[str, float] = {}
    for ticker in close.columns:
        for name, periods in HORIZONS.items():
            if len(close) <= periods:
                out[f"{ticker}_{name}"] = float("nan")
                continue
            out[f"{ticker}_{name}"] = float(close[ticker].iloc[-1] / close[ticker].iloc[-1 - periods] - 1)
    return out
