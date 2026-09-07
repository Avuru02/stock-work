"""Equal-weight S&P 500 industry baskets (semis, memory, software, and a few peers)."""

from __future__ import annotations

import pandas as pd

from analytics.breadth import grouped_breadth, grouped_leaders
from analytics.money_flow import grouped_constituent_flow
from analytics.relative_strength import relative_strength
from analytics.rotation_score import composite_score
from analytics.rrg import compute_rrg_from_close
from config import BENCHMARK, HORIZONS, INDUSTRY_GROUPS, INDUSTRY_LABELS, SECTOR_LABELS
from data.prices import pivot_field

MIN_NAMES = 2


def _norm_subindustry(value: str) -> str:
    text = str(value).lower().replace("&", "and").replace(",", " ")
    return " ".join(text.split())


def resolve_industry_members(universe: pd.DataFrame) -> pd.DataFrame:
    """Map S&P names into curated industry baskets."""
    if "sub_industry" not in universe.columns:
        raise ValueError("Universe is missing sub_industry; refresh the Wikipedia cache.")

    available = set(universe["ticker"].tolist())
    sub_norm = universe["sub_industry"].map(_norm_subindustry)
    frames = []
    for group_id, spec in INDUSTRY_GROUPS.items():
        mask = pd.Series(False, index=universe.index)
        wanted = {_norm_subindustry(name) for name in spec.get("sub_industries", ())}
        if wanted:
            mask = mask | sub_norm.isin(wanted)
        explicit = [t for t in spec.get("tickers", ()) if t in available]
        if explicit:
            mask = mask | universe["ticker"].isin(explicit)
        excluded = set(spec.get("exclude_tickers", ()))
        if excluded:
            mask = mask & ~universe["ticker"].isin(excluded)
        members = universe.loc[mask].copy()
        if len(members) < MIN_NAMES:
            continue
        members["industry_id"] = group_id
        members["industry_label"] = spec["label"]
        members["parent_etf"] = spec["parent_etf"]
        members["parent_sector"] = SECTOR_LABELS.get(spec["parent_etf"], spec["parent_etf"])
        frames.append(members)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def equal_weight_index(stock_close: pd.DataFrame, tickers: list[str]) -> pd.Series:
    panel = stock_close[tickers].dropna(how="all")
    rets = panel.pct_change()
    ew = rets.mean(axis=1, skipna=True)
    idx = (1 + ew.fillna(0.0)).cumprod()
    first = idx.loc[idx.ne(0)].iloc[0]
    return idx / first * 100.0


def industry_close_panel(
    stock_prices: pd.DataFrame,
    etf_prices: pd.DataFrame,
    members: pd.DataFrame,
) -> pd.DataFrame:
    stock_close = pivot_field(stock_prices, "close")
    spy = pivot_field(etf_prices, "close")[BENCHMARK]
    pieces = {BENCHMARK: spy}
    for group_id, group in members.groupby("industry_id"):
        tickers = [t for t in group["ticker"].tolist() if t in stock_close.columns]
        if len(tickers) < MIN_NAMES:
            continue
        pieces[group_id] = equal_weight_index(stock_close, tickers)
    close = pd.concat(pieces, axis=1).sort_index()
    return close.dropna(how="all")


def compute_industry_rotation(
    stock_prices: pd.DataFrame,
    etf_prices: pd.DataFrame,
    universe: pd.DataFrame,
) -> dict:
    members = resolve_industry_members(universe)
    if members.empty:
        return {
            "members": members,
            "relative_strength": pd.DataFrame(),
            "rrg": pd.DataFrame(),
            "rrg_trails": pd.DataFrame(),
            "scored": pd.DataFrame(),
            "leaders": {},
            "narrative": "No industry baskets could be resolved from the S&P 500 list.",
        }

    close = industry_close_panel(stock_prices, etf_prices, members)
    items = []
    groups: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    extras: dict[str, dict] = {}
    for group_id, group in members.groupby("industry_id"):
        if group_id not in close.columns:
            continue
        tickers = group["ticker"].tolist()
        groups[group_id] = tickers
        labels[group_id] = INDUSTRY_LABELS[group_id]
        parent = group["parent_etf"].iloc[0]
        extras[group_id] = {
            "parent_etf": parent,
            "parent_sector": group["parent_sector"].iloc[0],
            "gics": INDUSTRY_LABELS[group_id],
        }
        items.append(
            {
                "etf": group_id,
                "sector": INDUSTRY_LABELS[group_id],
                "gics": INDUSTRY_LABELS[group_id],
                "parent_etf": parent,
            }
        )

    rs = relative_strength(close, items)
    rrg_latest, rrg_trails = compute_rrg_from_close(close, ticker_labels=labels)
    flow = grouped_constituent_flow(stock_prices, groups, labels)
    index_ret_1m = {}
    one_m = HORIZONS["1m"]
    for group_id in groups:
        series = close[group_id].dropna()
        if len(series) > one_m:
            index_ret_1m[group_id] = float(series.iloc[-1] / series.iloc[-1 - one_m] - 1)
    breadth = grouped_breadth(stock_prices, groups, labels, index_ret_1m=index_ret_1m, extra=extras)
    scored = composite_score(rs, rrg_latest, flow, breadth)
    if "parent_etf" not in scored.columns and not rs.empty:
        scored = scored.merge(rs[["etf", "parent_etf"]], on="etf", how="left")
    parent_map = {gid: extras[gid]["parent_etf"] for gid in extras}
    parent_sector_map = {gid: extras[gid]["parent_sector"] for gid in extras}
    scored["parent_etf"] = scored["etf"].map(parent_map)
    scored["parent_sector"] = scored["etf"].map(parent_sector_map)

    leaders = {}
    for group_id, group in members.groupby("industry_id"):
        leaders[group_id] = grouped_leaders(
            stock_prices,
            group,
            etf_prices,
            group_id,
            INDUSTRY_LABELS[group_id],
        )

    inflow = scored[scored["flow_bucket"] == "Inflow"]
    outflow = scored[scored["flow_bucket"] == "Outflow"]
    into = ", ".join(inflow["sector"].head(3).tolist()) or "no group"
    out_of = ", ".join(outflow["sector"].head(3).tolist()) or "no group"
    leader = scored.iloc[0]
    narrative = (
        f"Among industry groups, estimated pressure is strongest in {into} "
        f"and weakest in {out_of}. {leader['sector']} ranks first "
        f"({leader['score']:.0f}, {leader['quadrant']})."
    )
    tech = scored[scored["parent_etf"] == "XLK"]
    if len(tech) >= 2:
        ordered = tech.sort_values("score", ascending=False)
        leader = ordered.iloc[0]
        laggard = ordered.iloc[-1]
        if leader["etf"] != laggard["etf"]:
            narrative += (
                f" Inside Technology, {leader['sector']} is ahead of {laggard['sector']} "
                f"({leader['score']:.0f} vs {laggard['score']:.0f})."
            )

    return {
        "members": members,
        "relative_strength": rs,
        "rrg": rrg_latest,
        "rrg_trails": rrg_trails,
        "scored": scored,
        "leaders": leaders,
        "narrative": narrative,
    }
