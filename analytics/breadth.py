"""Sector breadth and stock-level leadership versus SPY."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BENCHMARK,
    BREADTH_HIGH_LOW_WINDOW,
    BROAD_BREADTH,
    HORIZONS,
    MA_WINDOW,
    NARROW_BREADTH,
    SECTOR_ETFS,
    SECTOR_LABELS,
)
from data.prices import pivot_field


def _leadership_label(rs_1m: float, pct_beat_spy: float) -> str:
    """Participation vs SPY, not vs zero — a bull market makes almost every name green."""
    if pd.isna(rs_1m) or pd.isna(pct_beat_spy):
        return "Unknown"
    if rs_1m > 0 and pct_beat_spy < NARROW_BREADTH:
        return "Narrow"
    if rs_1m > 0 and pct_beat_spy >= BROAD_BREADTH:
        return "Broad"
    if rs_1m <= 0 and pct_beat_spy <= NARROW_BREADTH:
        return "Broad weakness"
    return "Mixed"


def grouped_breadth(
    stock_prices: pd.DataFrame,
    groups: dict[str, list[str]],
    labels: dict[str, str],
    index_ret_1m: dict[str, float] | None = None,
    extra: dict[str, dict] | None = None,
    spy_ret_1m: float | None = None,
    ma_window: int = MA_WINDOW,
    high_low_window: int = BREADTH_HIGH_LOW_WINDOW,
) -> pd.DataFrame:
    stock_close = pivot_field(stock_prices, "close")
    one_m = HORIZONS["1m"]
    rows = []
    for group_id, names in groups.items():
        names = [t for t in names if t in stock_close.columns]
        if not names:
            continue
        panel = stock_close[names]
        last = panel.iloc[-1]
        ma = panel.rolling(ma_window, min_periods=ma_window).mean().iloc[-1]
        above_ma = (last > ma).mean()
        roll_high = panel.rolling(high_low_window, min_periods=high_low_window).max().iloc[-1]
        roll_low = panel.rolling(high_low_window, min_periods=high_low_window).min().iloc[-1]
        new_highs = int((last >= roll_high).sum())
        new_lows = int((last <= roll_low).sum())
        if len(panel) > one_m:
            ret_1m = panel.iloc[-1] / panel.iloc[-1 - one_m] - 1
        else:
            ret_1m = pd.Series(index=names, dtype=float)
        pct_up = float((ret_1m > 0).mean()) if len(ret_1m) else np.nan
        if spy_ret_1m is None or pd.isna(spy_ret_1m) or len(ret_1m) == 0:
            pct_beat = pct_up
        else:
            pct_beat = float((ret_1m > spy_ret_1m).mean())
        group_1m = np.nan if not index_ret_1m else index_ret_1m.get(group_id, np.nan)
        group_rs = (
            group_1m - spy_ret_1m
            if spy_ret_1m is not None and pd.notna(spy_ret_1m) and pd.notna(group_1m)
            else group_1m
        )
        row = {
            "etf": group_id,
            "sector": labels.get(group_id, group_id),
            "n_stocks": len(names),
            "pct_above_50dma": float(above_ma),
            "new_highs_20": new_highs,
            "new_lows_20": new_lows,
            "pct_up_1m": pct_up,
            "pct_beat_spy": pct_beat,
            "etf_ret_1m": group_1m,
            "leadership": _leadership_label(group_rs, pct_beat),
        }
        if extra and group_id in extra:
            row.update(extra[group_id])
        rows.append(row)
    return pd.DataFrame(rows)


def sector_breadth(
    stock_prices: pd.DataFrame,
    universe: pd.DataFrame,
    etf_prices: pd.DataFrame,
    ma_window: int = MA_WINDOW,
    high_low_window: int = BREADTH_HIGH_LOW_WINDOW,
) -> pd.DataFrame:
    stock_close = pivot_field(stock_prices, "close")
    etf_close = pivot_field(etf_prices, "close")
    one_m = HORIZONS["1m"]
    rows = []
    for gics, etf in SECTOR_ETFS.items():
        names = universe.loc[universe["etf"] == etf, "ticker"].tolist()
        names = [t for t in names if t in stock_close.columns]
        if not names:
            continue
        panel = stock_close[names]
        last = panel.iloc[-1]
        ma = panel.rolling(ma_window, min_periods=ma_window).mean().iloc[-1]
        above_ma = (last > ma).mean()
        roll_high = panel.rolling(high_low_window, min_periods=high_low_window).max().iloc[-1]
        roll_low = panel.rolling(high_low_window, min_periods=high_low_window).min().iloc[-1]
        new_highs = int((last >= roll_high).sum())
        new_lows = int((last <= roll_low).sum())
        if len(panel) > one_m:
            ret_1m = panel.iloc[-1] / panel.iloc[-1 - one_m] - 1
        else:
            ret_1m = pd.Series(index=names, dtype=float)
        pct_up = float((ret_1m > 0).mean()) if len(ret_1m) else np.nan
        etf_1m = np.nan
        spy_ret = np.nan
        if etf in etf_close.columns and len(etf_close) > one_m:
            etf_1m = float(etf_close[etf].iloc[-1] / etf_close[etf].iloc[-1 - one_m] - 1)
        if BENCHMARK in etf_close.columns and len(etf_close) > one_m:
            spy_ret = float(etf_close[BENCHMARK].iloc[-1] / etf_close[BENCHMARK].iloc[-1 - one_m] - 1)
        if pd.isna(spy_ret) or len(ret_1m) == 0:
            pct_beat = pct_up
        else:
            pct_beat = float((ret_1m > spy_ret).mean())
        etf_rs = etf_1m - spy_ret if pd.notna(etf_1m) and pd.notna(spy_ret) else etf_1m
        rows.append(
            {
                "etf": etf,
                "sector": SECTOR_LABELS.get(etf, etf),
                "gics": gics,
                "n_stocks": len(names),
                "pct_above_50dma": float(above_ma),
                "new_highs_20": new_highs,
                "new_lows_20": new_lows,
                "pct_up_1m": pct_up,
                "pct_beat_spy": pct_beat,
                "etf_ret_1m": etf_1m,
                "leadership": _leadership_label(etf_rs, pct_beat),
            }
        )
    return pd.DataFrame(rows)


def grouped_leaders(
    stock_prices: pd.DataFrame,
    members: pd.DataFrame,
    spy_prices: pd.DataFrame,
    group_id: str,
    label: str,
    n: int = 5,
) -> pd.DataFrame:
    """Top and bottom names in a group by 1-month relative strength vs SPY."""
    stock_close = pivot_field(stock_prices, "close")
    spy_close = pivot_field(spy_prices, "close")
    if BENCHMARK not in spy_close.columns:
        raise ValueError("SPY missing from prices.")
    tickers = [t for t in members["ticker"].tolist() if t in stock_close.columns]
    if not tickers:
        return pd.DataFrame()
    one_m = HORIZONS["1m"]
    if len(stock_close) <= one_m or len(spy_close) <= one_m:
        return pd.DataFrame()
    spy_ret = float(spy_close[BENCHMARK].iloc[-1] / spy_close[BENCHMARK].iloc[-1 - one_m] - 1)
    panel = stock_close[tickers]
    ret = panel.iloc[-1] / panel.iloc[-1 - one_m] - 1
    frame = pd.DataFrame({"ticker": ret.index, "ret_1m": ret.values, "rs_1m": ret.values - spy_ret})
    name_cols = [c for c in ["ticker", "name", "sub_industry"] if c in members.columns]
    frame = frame.merge(members[name_cols], on="ticker", how="left")
    frame = frame.dropna(subset=["rs_1m"])
    if frame.empty:
        return frame
    take = min(n, max(1, (len(frame) + 1) // 2))
    top = frame.nlargest(take, "rs_1m").copy()
    bottom = frame.nsmallest(take, "rs_1m").copy()
    top["rank_type"] = "Leader"
    bottom["rank_type"] = "Laggard"
    out = pd.concat([top, bottom], ignore_index=True).drop_duplicates(subset=["ticker"], keep="first")
    out["sector"] = label
    out["etf"] = group_id
    rank_order = pd.Categorical(out["rank_type"], categories=["Leader", "Laggard", "Mid"], ordered=True)
    out["rank_type"] = rank_order
    return out.sort_values(["rank_type", "rs_1m"], ascending=[True, False]).reset_index(drop=True)


def sector_leaders(
    stock_prices: pd.DataFrame,
    universe: pd.DataFrame,
    spy_prices: pd.DataFrame,
    etf: str,
    n: int = 5,
) -> pd.DataFrame:
    """Top and bottom n stocks in a sector by 1-month relative strength vs SPY."""
    stock_close = pivot_field(stock_prices, "close")
    spy_close = pivot_field(spy_prices, "close")
    if BENCHMARK not in spy_close.columns:
        raise ValueError("SPY missing from prices.")
    names = universe.loc[universe["etf"] == etf, ["ticker", "name"]]
    tickers = [t for t in names["ticker"].tolist() if t in stock_close.columns]
    if not tickers:
        return pd.DataFrame()
    one_m = HORIZONS["1m"]
    if len(stock_close) <= one_m or len(spy_close) <= one_m:
        return pd.DataFrame()
    spy_ret = float(spy_close[BENCHMARK].iloc[-1] / spy_close[BENCHMARK].iloc[-1 - one_m] - 1)
    panel = stock_close[tickers]
    ret = panel.iloc[-1] / panel.iloc[-1 - one_m] - 1
    frame = pd.DataFrame(
        {
            "ticker": ret.index,
            "ret_1m": ret.values,
            "rs_1m": ret.values - spy_ret,
        }
    )
    frame = frame.merge(names, on="ticker", how="left")
    frame = frame.dropna(subset=["rs_1m"])
    frame["rank_type"] = "Mid"
    top = frame.nlargest(n, "rs_1m").copy()
    bottom = frame.nsmallest(n, "rs_1m").copy()
    top["rank_type"] = "Leader"
    bottom["rank_type"] = "Laggard"
    out = pd.concat([top, bottom], ignore_index=True)
    out["sector"] = SECTOR_LABELS.get(etf, etf)
    out["etf"] = etf
    rank_order = pd.Categorical(out["rank_type"], categories=["Leader", "Laggard", "Mid"], ordered=True)
    out["rank_type"] = rank_order
    return out.sort_values(["rank_type", "rs_1m"], ascending=[True, False]).reset_index(drop=True)
