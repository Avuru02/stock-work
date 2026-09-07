"""JdK-style Relative Rotation Graph metrics (RS-Ratio and RS-Momentum)."""

from __future__ import annotations

import pandas as pd

from config import (
    BENCHMARK,
    RRG_TRAIL_STEPS,
    RRG_TRAIL_STRIDE,
    RRG_WINDOW,
    SECTOR_ETFS,
    SECTOR_LABELS,
)
from data.prices import pivot_field


def _rs_ratio_series(rs: pd.Series, window: int) -> pd.Series:
    """Normalize relative strength around 100 using nested moving averages."""
    rs_sma = rs.rolling(window, min_periods=window).mean()
    rs_ratio = 100 * rs_sma / rs_sma.rolling(window, min_periods=window).mean()
    return rs_ratio


def _quadrant(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "Leading"
    if rs_ratio >= 100 and rs_momentum < 100:
        return "Weakening"
    if rs_ratio < 100 and rs_momentum < 100:
        return "Lagging"
    return "Improving"


def compute_rrg(
    etf_prices: pd.DataFrame,
    benchmark: str = BENCHMARK,
    window: int = RRG_WINDOW,
    ticker_labels: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return latest RRG points and a short trail for each series."""
    close = pivot_field(etf_prices, "close")
    return compute_rrg_from_close(close, benchmark=benchmark, window=window, ticker_labels=ticker_labels)


def compute_rrg_from_close(
    close: pd.DataFrame,
    benchmark: str = BENCHMARK,
    window: int = RRG_WINDOW,
    ticker_labels: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if benchmark not in close.columns:
        raise ValueError(f"Benchmark {benchmark} missing from prices.")

    labels = ticker_labels or {etf: SECTOR_LABELS.get(etf, etf) for etf in SECTOR_ETFS.values()}
    bench = close[benchmark]
    latest_rows = []
    trail_rows = []
    for ticker, label in labels.items():
        if ticker not in close.columns:
            continue
        rs = close[ticker] / bench
        rs_ratio = _rs_ratio_series(rs, window)
        rs_momentum = _rs_ratio_series(rs_ratio / 100.0, window)
        valid = pd.concat(
            {"rs_ratio": rs_ratio, "rs_momentum": rs_momentum},
            axis=1,
        ).dropna()
        if valid.empty:
            continue
        last = valid.iloc[-1]
        latest_rows.append(
            {
                "etf": ticker,
                "sector": label,
                "rs_ratio": float(last["rs_ratio"]),
                "rs_momentum": float(last["rs_momentum"]),
                "quadrant": _quadrant(float(last["rs_ratio"]), float(last["rs_momentum"])),
            }
        )
        stride_points = valid.iloc[::-RRG_TRAIL_STRIDE][::-1].tail(RRG_TRAIL_STEPS)
        for date, point in stride_points.iterrows():
            trail_rows.append(
                {
                    "etf": ticker,
                    "sector": label,
                    "date": date,
                    "rs_ratio": float(point["rs_ratio"]),
                    "rs_momentum": float(point["rs_momentum"]),
                }
            )

    latest = pd.DataFrame(latest_rows)
    trails = pd.DataFrame(trail_rows)
    return latest, trails
