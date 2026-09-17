"""Volume expansion and Chaikin-style money-flow proxies (not true ETF creations)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CMF_WINDOW, REL_DVOL_SMOOTH, SECTOR_ETFS, SECTOR_LABELS
from data.prices import pivot_field


def _money_flow_multiplier(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    spread = high - low
    mfm = np.where(spread == 0, 0.0, ((close - low) - (high - close)) / spread)
    return pd.Series(mfm, index=close.index, dtype=float)


def _panel(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    return pivot_field(prices, field)


def ticker_flow_metrics(prices: pd.DataFrame, tickers: list[str], window: int = CMF_WINDOW) -> pd.DataFrame:
    close = _panel(prices, "close")
    high = _panel(prices, "high")
    low = _panel(prices, "low")
    volume = _panel(prices, "volume")
    rows = []
    for ticker in tickers:
        if ticker not in close.columns:
            continue
        c = close[ticker]
        h = high[ticker]
        l = low[ticker]
        v = volume[ticker].fillna(0)
        mfm = _money_flow_multiplier(h, l, c)
        dollar_vol = c * v
        mfv = mfm * dollar_vol
        cmf = mfv.rolling(window, min_periods=window).sum() / dollar_vol.rolling(
            window, min_periods=window
        ).sum().replace(0, np.nan)
        avg_dvol = dollar_vol.rolling(window, min_periods=window).mean()
        rel_dvol = dollar_vol / avg_dvol.replace(0, np.nan)
        rel_dvol_smooth = rel_dvol.rolling(REL_DVOL_SMOOTH, min_periods=max(3, REL_DVOL_SMOOTH // 2)).mean()
        flow_proxy = mfv.rolling(window, min_periods=window).sum()
        last_idx = c.dropna().index[-1] if c.dropna().shape[0] else None
        if last_idx is None:
            continue
        rows.append(
            {
                "ticker": ticker,
                "dollar_volume": float(dollar_vol.loc[last_idx]),
                "rel_dollar_volume": float(rel_dvol_smooth.loc[last_idx])
                if pd.notna(rel_dvol_smooth.loc[last_idx])
                else np.nan,
                "rel_dollar_volume_1d": float(rel_dvol.loc[last_idx]) if pd.notna(rel_dvol.loc[last_idx]) else np.nan,
                "cmf": float(cmf.loc[last_idx]) if pd.notna(cmf.loc[last_idx]) else np.nan,
                "flow_proxy": float(flow_proxy.loc[last_idx]) if pd.notna(flow_proxy.loc[last_idx]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def etf_money_flow(etf_prices: pd.DataFrame, window: int = CMF_WINDOW) -> pd.DataFrame:
    tickers = list(SECTOR_ETFS.values())
    frame = ticker_flow_metrics(etf_prices, tickers, window=window)
    if frame.empty:
        return frame
    frame["etf"] = frame["ticker"]
    frame["sector"] = frame["etf"].map(SECTOR_LABELS)
    return frame.drop(columns=["ticker"])


def grouped_constituent_flow(
    stock_prices: pd.DataFrame,
    groups: dict[str, list[str]],
    labels: dict[str, str],
    window: int = CMF_WINDOW,
    id_column: str = "etf",
) -> pd.DataFrame:
    """Dollar-volume-weighted money flow aggregated from named stock groups."""
    close = _panel(stock_prices, "close")
    high = _panel(stock_prices, "high")
    low = _panel(stock_prices, "low")
    volume = _panel(stock_prices, "volume")

    rows = []
    for group_id, names in groups.items():
        names = [t for t in names if t in close.columns]
        if not names:
            continue
        mfv_sum = None
        dvol_sum = None
        for ticker in names:
            mfm = _money_flow_multiplier(high[ticker], low[ticker], close[ticker])
            dollar_vol = close[ticker] * volume[ticker].fillna(0)
            mfv = mfm * dollar_vol
            mfv_sum = mfv if mfv_sum is None else mfv_sum.add(mfv, fill_value=0)
            dvol_sum = dollar_vol if dvol_sum is None else dvol_sum.add(dollar_vol, fill_value=0)
        if mfv_sum is None or dvol_sum is None:
            continue
        min_periods = max(5, window // 2)
        stock_cmf = mfv_sum.rolling(window, min_periods=min_periods).sum() / dvol_sum.rolling(
            window, min_periods=min_periods
        ).sum().replace(0, np.nan)
        avg_dvol = dvol_sum.rolling(window, min_periods=min_periods).mean()
        rel_dvol = dvol_sum / avg_dvol.replace(0, np.nan)
        rel_dvol_smooth = rel_dvol.rolling(REL_DVOL_SMOOTH, min_periods=max(3, REL_DVOL_SMOOTH // 2)).mean()
        last_idx = dvol_sum.dropna().index[-1]
        rows.append(
            {
                id_column: group_id,
                "sector": labels.get(group_id, group_id),
                "cmf": float(stock_cmf.loc[last_idx]) if pd.notna(stock_cmf.loc[last_idx]) else np.nan,
                "rel_dollar_volume": float(rel_dvol_smooth.loc[last_idx])
                if pd.notna(rel_dvol_smooth.loc[last_idx])
                else np.nan,
                "rel_dollar_volume_1d": float(rel_dvol.loc[last_idx]) if pd.notna(rel_dvol.loc[last_idx]) else np.nan,
                "dollar_volume": float(dvol_sum.loc[last_idx]),
                "flow_proxy": float(mfv_sum.rolling(window, min_periods=min_periods).sum().loc[last_idx]),
                "n_stocks": len(names),
            }
        )
    return pd.DataFrame(rows)


def sector_constituent_flow(
    stock_prices: pd.DataFrame,
    universe: pd.DataFrame,
    window: int = CMF_WINDOW,
) -> pd.DataFrame:
    """Dollar-volume-weighted money flow aggregated from S&P 500 names in each sector."""
    groups = {
        etf: universe.loc[universe["etf"] == etf, "ticker"].tolist()
        for etf in SECTOR_ETFS.values()
    }
    frame = grouped_constituent_flow(stock_prices, groups, SECTOR_LABELS, window=window)
    if frame.empty:
        return frame
    return frame.rename(
        columns={
            "cmf": "stock_cmf",
            "rel_dollar_volume": "stock_rel_dollar_volume",
            "dollar_volume": "stock_dollar_volume",
            "flow_proxy": "stock_flow_proxy",
        }
    )
