"""Build the full dashboard payload from cache or a fresh download."""

from __future__ import annotations

from data.prices import latest_as_of, load_etf_prices, load_stock_prices
from data.universe import load_universe
from analytics.breadth import sector_breadth, sector_leaders
from analytics.industries import compute_industry_rotation
from analytics.money_flow import etf_money_flow, sector_constituent_flow
from analytics.relative_strength import sector_relative_strength, style_returns
from analytics.rotation_score import composite_score, risk_regime, rotation_narrative
from analytics.rrg import compute_rrg
from config import BENCHMARK, SECTOR_ETFS


def build_dashboard_data(force: bool = False) -> dict:
    universe = load_universe(force=force)
    etf_prices = load_etf_prices(force=force)
    stock_prices = load_stock_prices(universe["ticker"].tolist(), force=force)

    rs = sector_relative_strength(etf_prices)
    rrg_latest, rrg_trails = compute_rrg(etf_prices)
    etf_flow = etf_money_flow(etf_prices)
    stock_flow = sector_constituent_flow(stock_prices, universe)
    breadth = sector_breadth(stock_prices, universe, etf_prices)
    scored = composite_score(rs, rrg_latest, etf_flow, breadth)
    if not stock_flow.empty:
        scored = scored.merge(
            stock_flow[
                [
                    "etf",
                    "stock_cmf",
                    "stock_rel_dollar_volume",
                    "stock_dollar_volume",
                    "stock_flow_proxy",
                ]
            ],
            on="etf",
            how="left",
        )

    leaders = {}
    for etf in SECTOR_ETFS.values():
        leaders[etf] = sector_leaders(stock_prices, universe, etf_prices, etf)

    styles = style_returns(etf_prices)
    regime = risk_regime(rs)
    narrative = rotation_narrative(scored)
    industries = compute_industry_rotation(stock_prices, etf_prices, universe)
    as_of = latest_as_of(etf_prices)

    return {
        "as_of": as_of,
        "universe": universe,
        "etf_prices": etf_prices,
        "stock_prices": stock_prices,
        "relative_strength": rs,
        "rrg": rrg_latest,
        "rrg_trails": rrg_trails,
        "etf_flow": etf_flow,
        "stock_flow": stock_flow,
        "breadth": breadth,
        "scored": scored,
        "leaders": leaders,
        "styles": styles,
        "regime": regime,
        "narrative": narrative,
        "spy_1d": styles.get(f"{BENCHMARK}_1d"),
        "spy_1m": styles.get(f"{BENCHMARK}_1m"),
        "qqq_1m": styles.get("QQQ_1m"),
        "iwm_1m": styles.get("IWM_1m"),
        "industries": industries,
    }
