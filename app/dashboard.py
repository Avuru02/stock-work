"""US sector rotation dashboard — relative strength, RRG, estimated flows, leadership."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as _app_config

importlib.reload(_app_config)

from analytics.pipeline import build_dashboard_data
from config import BENCHMARK, CONFIRM_NEEDED, INFLOW_THRESHOLD, OUTFLOW_THRESHOLD, SCORE_WEIGHTS, SECTOR_ETFS, SECTOR_LABELS
from data.prices import pivot_field

st.set_page_config(page_title="US Sector Rotation", layout="wide")


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def _fmt_billions(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"${value / 1e9:.2f}B"


@st.cache_data(show_spinner="Loading market data and computing rotation metrics...")
def load_payload(nonce: int, force: bool, layout_version: int = 6) -> dict:
    payload = build_dashboard_data(force=force)
    # DataFrames cache cleanly; Timestamp becomes string for display.
    payload = dict(payload)
    payload["as_of"] = pd.Timestamp(payload["as_of"])
    return payload


SECTOR_COLORS = {
    "XLK": "#4C78A8",
    "XLF": "#F58518",
    "XLV": "#54A24B",
    "XLE": "#E45756",
    "XLY": "#B279A2",
    "XLP": "#72B7B2",
    "XLI": "#FF9DA6",
    "XLB": "#9D755D",
    "XLU": "#BAB0AC",
    "XLRE": "#D67195",
    "XLC": "#1F9E89",
}

INDUSTRY_COLORS = {
    "SEM": "#4C78A8",
    "MEM": "#E45756",
    "EQS": "#F58518",
    "SOFT": "#54A24B",
    "BANK": "#B279A2",
    "BIOT": "#72B7B2",
    "EXPL": "#9D755D",
    "RETL": "#FF9DA6",
    "HOME": "#BAB0AC",
    "NET": "#1F9E89",
}


def _series_color(etf: str) -> str:
    return SECTOR_COLORS.get(etf) or INDUSTRY_COLORS.get(etf) or "#6E7681"


def slice_trails(trails: pd.DataFrame, days: int) -> pd.DataFrame:
    if trails is None or trails.empty:
        return pd.DataFrame()
    frame = trails.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    keep = frame["date"].drop_duplicates().sort_values().tail(days)
    return frame[frame["date"].isin(keep)].sort_values(["etf", "date"])


def heatmap_figure(rs: pd.DataFrame, title: str, height: int = 420) -> go.Figure:
    horizons = ["1d", "1w", "1m", "3m"]
    z = rs[["rs_" + h for h in horizons]].values * 100
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["1 Day", "1 Week", "1 Month", "3 Month"],
            y=rs["sector"].tolist(),
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="vs SPY (pp)", ticksuffix=""),
            hovertemplate="%{y}<br>%{x}: %{z:.2f} pp<extra></extra>",
            text=[[f"{v:.1f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def rrg_figure(
    latest: pd.DataFrame,
    trails: pd.DataFrame,
    title: str = "Relative Rotation Graph (RS-Ratio vs RS-Momentum, 100 = SPY)",
    focus: list[str] | None = None,
) -> go.Figure:
    fig = go.Figure()
    if latest is None or latest.empty:
        return fig

    plot_trails = trails.copy() if trails is not None and not trails.empty else pd.DataFrame()
    if not plot_trails.empty:
        plot_trails["date"] = pd.to_datetime(plot_trails["date"])
    if focus:
        latest = latest[latest["etf"].isin(focus)]
        if not plot_trails.empty:
            plot_trails = plot_trails[plot_trails["etf"].isin(focus)]

    x_parts = [latest["rs_ratio"]]
    y_parts = [latest["rs_momentum"]]
    if not plot_trails.empty:
        x_parts.append(plot_trails["rs_ratio"])
        y_parts.append(plot_trails["rs_momentum"])
    x_vals = pd.concat(x_parts, ignore_index=True)
    y_vals = pd.concat(y_parts, ignore_index=True)
    pad = 1.5
    x0, x1 = float(x_vals.min()) - pad, float(x_vals.max()) + pad
    y0, y1 = float(y_vals.min()) - pad, float(y_vals.max()) + pad
    x0, x1 = min(x0, 98), max(x1, 102)
    y0, y1 = min(y0, 98), max(y1, 102)

    fig.update_layout(
        shapes=[
            dict(type="rect", x0=100, x1=x1, y0=100, y1=y1, fillcolor="rgba(61,154,106,0.12)", line=dict(width=0)),
            dict(type="rect", x0=100, x1=x1, y0=y0, y1=100, fillcolor="rgba(201,176,55,0.10)", line=dict(width=0)),
            dict(type="rect", x0=x0, x1=100, y0=y0, y1=100, fillcolor="rgba(201,80,66,0.10)", line=dict(width=0)),
            dict(type="rect", x0=x0, x1=100, y0=100, y1=y1, fillcolor="rgba(56,132,196,0.10)", line=dict(width=0)),
            dict(type="line", x0=100, x1=100, y0=y0, y1=y1, line=dict(color="#8B949E", width=1, dash="dot")),
            dict(type="line", x0=x0, x1=x1, y0=100, y1=100, line=dict(color="#8B949E", width=1, dash="dot")),
        ]
    )

    order = latest["etf"].tolist()
    for etf in order:
        color = _series_color(etf)
        label = latest.loc[latest["etf"] == etf, "sector"].iloc[0]
        trail = pd.DataFrame()
        if not plot_trails.empty:
            trail = plot_trails[plot_trails["etf"] == etf].sort_values("date")
        if not trail.empty:
            n = len(trail)
            sizes = [5 + 9 * (i / max(n - 1, 1)) for i in range(n)]
            opacities = [0.25 + 0.75 * (i / max(n - 1, 1)) for i in range(n)]
            texts = [""] * (n - 1) + [etf]
            hover = [
                f"{label} ({etf})<br>{pd.Timestamp(d).strftime('%Y-%m-%d')}<br>"
                f"RS-Ratio: {x:.2f}<br>RS-Momentum: {y:.2f}"
                for d, x, y in zip(trail["date"], trail["rs_ratio"], trail["rs_momentum"])
            ]
            fig.add_trace(
                go.Scatter(
                    x=trail["rs_ratio"],
                    y=trail["rs_momentum"],
                    mode="lines+markers+text",
                    name=label,
                    text=texts,
                    textposition="top center",
                    textfont=dict(size=11, color=color),
                    line=dict(color=color, width=1.5),
                    marker=dict(size=sizes, color=color, opacity=opacities, line=dict(width=0)),
                    hovertext=hover,
                    hoverinfo="text",
                    legendgroup=etf,
                )
            )
        else:
            row = latest[latest["etf"] == etf].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[row["rs_ratio"]],
                    y=[row["rs_momentum"]],
                    mode="markers+text",
                    name=label,
                    text=[etf],
                    textposition="top center",
                    marker=dict(size=12, color=color),
                    hovertemplate=f"{label} ({etf})<br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>",
                    legendgroup=etf,
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="RS-Ratio (relative trend vs SPY)",
        yaxis_title="RS-Momentum (acceleration of relative trend)",
        height=560,
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0, font=dict(size=11)),
        annotations=[
            dict(x=1, y=1, xref="paper", yref="paper", text="Leading", showarrow=False, font=dict(size=11, color="#3D9A6A"), xanchor="right", yanchor="top"),
            dict(x=1, y=0, xref="paper", yref="paper", text="Weakening", showarrow=False, font=dict(size=11, color="#C9B037"), xanchor="right", yanchor="bottom"),
            dict(x=0, y=0, xref="paper", yref="paper", text="Lagging", showarrow=False, font=dict(size=11, color="#C95042"), xanchor="left", yanchor="bottom"),
            dict(x=0, y=1, xref="paper", yref="paper", text="Improving", showarrow=False, font=dict(size=11, color="#3884C4"), xanchor="left", yanchor="top"),
        ],
        xaxis=dict(range=[x0, x1]),
        yaxis=dict(range=[y0, y1]),
    )
    return fig


def relative_path_figure(
    prices: pd.DataFrame,
    tickers: list[str],
    labels: dict[str, str],
    days: int,
    title: str,
) -> go.Figure:
    close = pivot_field(prices, "close")
    if BENCHMARK not in close.columns:
        return go.Figure()
    window = close[[BENCHMARK] + [t for t in tickers if t in close.columns]].dropna().tail(days + 1)
    if window.empty or len(window) < 2:
        return go.Figure()
    spy = window[BENCHMARK]
    fig = go.Figure()
    fig.add_hline(y=100, line_dash="dot", line_color="#8B949E")
    for ticker in tickers:
        if ticker not in window.columns:
            continue
        rel = (window[ticker] / spy) / (window[ticker].iloc[0] / spy.iloc[0]) * 100
        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=rel,
                mode="lines",
                name=labels.get(ticker, ticker),
                line=dict(color=_series_color(ticker), width=2),
                hovertemplate="%{fullData.name}<br>%{x|%Y-%m-%d}<br>vs SPY (start=100): %{y:.1f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price vs SPY (100 = start of window)",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0, font=dict(size=11)),
    )
    return fig


def rrg_ratio_history_figure(trails: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if trails is None or trails.empty:
        return fig
    frame = trails.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    fig.add_hline(y=100, line_dash="dot", line_color="#8B949E")
    for etf, trail in frame.groupby("etf"):
        trail = trail.sort_values("date")
        label = trail["sector"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=trail["date"],
                y=trail["rs_ratio"],
                mode="lines",
                name=label,
                line=dict(color=_series_color(etf), width=2),
                hovertemplate="%{fullData.name}<br>%{x|%Y-%m-%d}<br>RS-Ratio: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="RS-Ratio (100 = in line with SPY)",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0, font=dict(size=11)),
    )
    return fig


def ranking_table(scored: pd.DataFrame, name_header: str = "Sector") -> pd.DataFrame:
    cols = [
        "sector",
        "etf",
        "flow_bucket",
        "conviction",
        "score",
        "quadrant",
        "rs_1m",
        "persist_1m",
        "signed_rel_dvol",
        "cmf_used",
        "pct_beat_spy",
        "pct_above_50dma",
        "leadership",
        "confirm_flags",
        "n_stocks",
    ]
    have = [c for c in cols if c in scored.columns]
    table = scored[have].copy()
    if "rs_1m" in table.columns:
        table["rs_1m"] = table["rs_1m"].map(lambda v: _fmt_pct(v))
    if "persist_1m" in table.columns:
        table["persist_1m"] = table["persist_1m"].map(lambda v: _fmt_pct(v, 0))
    if "signed_rel_dvol" in table.columns:
        table["signed_rel_dvol"] = table["signed_rel_dvol"].map(lambda v: _fmt_num(v, 2) + "x")
    if "rel_dollar_volume" in table.columns:
        table["rel_dollar_volume"] = table["rel_dollar_volume"].map(lambda v: _fmt_num(v, 2) + "x")
    if "cmf_used" in table.columns:
        table["cmf_used"] = table["cmf_used"].map(lambda v: _fmt_num(v, 2))
    if "cmf" in table.columns:
        table["cmf"] = table["cmf"].map(lambda v: _fmt_num(v, 2))
    if "pct_beat_spy" in table.columns:
        table["pct_beat_spy"] = table["pct_beat_spy"].map(lambda v: _fmt_pct(v, 0))
    if "pct_above_50dma" in table.columns:
        table["pct_above_50dma"] = table["pct_above_50dma"].map(lambda v: _fmt_pct(v, 0))
    if "score" in table.columns:
        table["score"] = table["score"].map(lambda v: _fmt_num(v, 0))
    rename = {
        "sector": name_header,
        "etf": "Id",
        "flow_bucket": "Flow",
        "conviction": "Confirms",
        "score": "Score",
        "quadrant": "RRG",
        "rs_1m": "RS 1M",
        "persist_1m": "Persist",
        "signed_rel_dvol": "Signed $ vol",
        "rel_dollar_volume": "Rel $ vol",
        "cmf_used": "CMF",
        "cmf": "CMF",
        "pct_beat_spy": "% beat SPY",
        "pct_above_50dma": "% > 50DMA",
        "leadership": "Leadership",
        "confirm_flags": "Tests passed",
        "n_stocks": "Names",
    }
    if "parent_sector" in scored.columns:
        table.insert(1, "Parent", scored["parent_sector"].to_list())
    return table.rename(columns=rename)


def leaders_table(leaders: pd.DataFrame, rank_type: str) -> pd.DataFrame:
    cols = [c for c in ["ticker", "name", "sub_industry", "ret_1m", "rs_1m"] if c in leaders.columns]
    subset = leaders[leaders["rank_type"] == rank_type][cols].copy()
    subset = subset.rename(
        columns={
            "ticker": "Ticker",
            "name": "Name",
            "sub_industry": "Sub-industry",
            "ret_1m": "1M return",
            "rs_1m": "1M RS",
        }
    )
    if "1M return" in subset.columns:
        subset["1M return"] = subset["1M return"].map(lambda v: _fmt_pct(v))
    if "1M RS" in subset.columns:
        subset["1M RS"] = subset["1M RS"].map(lambda v: _fmt_pct(v))
    return subset


def main() -> None:
    if "data_nonce" not in st.session_state:
        st.session_state.data_nonce = 0
        st.session_state.force_refresh = False

    st.title("US Sector Rotation")
    st.caption(
        "Estimated rotation from SPDR sector ETFs, S&P 500 constituents, and equal-weight industry baskets. "
        "Industry groups (semis, memory, software, etc.) are not cap-weighted, so NVDA cannot hide Micron. "
        "Score is absolute pressure vs SPY (50 = in line) — the least-bad sector does not automatically look strong. "
        "Inflow/Outflow require beating or lagging SPY plus confirming tests. "
        "Money-flow figures are volume/price proxies, not ETF creations or 13F flows."
    )

    with st.sidebar:
        st.header("Controls")
        if st.button("Refresh data", width="stretch"):
            st.session_state.data_nonce += 1
            st.session_state.force_refresh = True
            load_payload.clear()
        st.caption("Uses cached Parquet if it is less than 18 hours old. Refresh downloads from Yahoo Finance.")
        trail_days = st.slider(
            "RRG trail (trading days)",
            min_value=5,
            max_value=63,
            value=21,
            help="Each dot is one session. Older dots are smaller and more faded. Click a name in the legend to hide it.",
        )
        st.caption("Default 21 sessions is about one month. Daily RRG still uses a 21-day smoother, so the path is the trend, not raw noise.")
        st.subheader("Score weights")
        st.json(SCORE_WEIGHTS)
        st.caption(
            f"Score is absolute vs SPY (50 = in line), not a ranking. "
            f"Inflow: 1-month RS > 0, majority of names beating SPY, not narrow, "
            f"≥{CONFIRM_NEEDED}/6 confirms, score ≥ {INFLOW_THRESHOLD:.0f}. "
            f"Outflow: 1-month RS < 0, minority beating SPY, ≥{CONFIRM_NEEDED}/6 confirms, "
            f"score ≤ {OUTFLOW_THRESHOLD:.0f}."
        )

    payload = load_payload(st.session_state.data_nonce, st.session_state.force_refresh, 6)
    scored = payload["scored"]
    rs = payload["relative_strength"]
    as_of = payload["as_of"]
    regime = payload["regime"]
    industries = payload.get("industries") or {}
    industry_scored = industries.get("scored", pd.DataFrame())
    industry_rs = industries.get("relative_strength", pd.DataFrame())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("As of", as_of.strftime("%Y-%m-%d"))
    k2.metric("SPY 1D", _fmt_pct(payload["spy_1d"]))
    k3.metric("SPY 1M", _fmt_pct(payload["spy_1m"]))
    k4.metric("Regime", str(regime["label"]), _fmt_pct(regime["spread"]))
    k5.metric("QQQ / IWM 1M", f"{_fmt_pct(payload['qqq_1m'])} / {_fmt_pct(payload['iwm_1m'])}")

    tab_sectors, tab_industries = st.tabs(["Sectors", "Industries (semis, memory, software)"])

    with tab_sectors:
        st.info(payload["narrative"])
        st.plotly_chart(
            heatmap_figure(rs, "Sector relative strength vs SPY (percentage points)"),
            width="stretch",
        )
        left, right = st.columns((1.15, 1))
        with left:
            sector_trails = slice_trails(payload["rrg_trails"], trail_days)
            st.plotly_chart(
                rrg_figure(
                    payload["rrg"],
                    sector_trails,
                    title=f"Sector RRG — daily path, last {trail_days} sessions (100 = SPY)",
                ),
                width="stretch",
            )
            st.caption("Each dot is one trading day. The path moving clockwise through Improving → Leading → Weakening → Lagging is the classic rotation.")
        with right:
            st.subheader("Estimated sector flow ranking")
            st.dataframe(ranking_table(scored, "Sector"), width="stretch", hide_index=True, height=420)
            st.caption(
                "Score is pressure vs SPY (0–100, 50 = in line), not a forced ranking. "
                "Inflow requires: 1-month RS vs SPY > 0, a majority of names beating SPY, leadership not narrow, "
                f"at least {CONFIRM_NEEDED} of 6 confirming tests (1w RS, RRG Improving/Leading *and* still beating SPY over 1 month, "
                "CMF > 0, 5-day volume expanding on an up week, majority of names beating SPY, persistence), "
                f"and score ≥ {INFLOW_THRESHOLD:.0f}. Outflow is the mirror. Otherwise Neutral — "
                "the tape can have no confirmed rotation. Persist = share of the last 21 sessions where 1-month RS beat SPY. "
                "Signed $ vol is 5-day relative dollar volume times the sign of 1-week RS (high volume on a down week is distribution). "
                "Sector CMF/volume use S&P names when available, not just the SPDR print. "
                "% beat SPY is participation; % > 50DMA is trend health."
            )

        st.plotly_chart(
            relative_path_figure(
                payload["etf_prices"],
                list(SECTOR_ETFS.values()),
                SECTOR_LABELS,
                trail_days,
                f"Day-by-day lag vs SPY — last {trail_days} sessions (100 = start of window)",
            ),
            width="stretch",
        )
        st.caption("Above 100 means the sector beat SPY since the start of the window. The slope is the daily lag or lead.")

        st.subheader("Sector drill-down")
        options = [f"{SECTOR_LABELS[etf]} ({etf})" for etf in SECTOR_ETFS.values()]
        default_etf = scored.iloc[0]["etf"]
        default_label = f"{SECTOR_LABELS[default_etf]} ({default_etf})"
        choice = st.selectbox("Sector", options, index=options.index(default_label))
        etf = choice.split("(")[-1].rstrip(")")

        row = scored[scored["etf"] == etf].iloc[0]
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Flow", str(row["flow_bucket"]), str(row.get("conviction", "")))
        b2.metric("RRG quadrant", str(row["quadrant"]))
        b3.metric("% beat SPY", _fmt_pct(row.get("pct_beat_spy"), 0))
        b4.metric("Persist vs SPY", _fmt_pct(row.get("persist_1m"), 0))
        b5.metric("20d highs / lows", f"{int(row.get('new_highs_20', 0))} / {int(row.get('new_lows_20', 0))}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pressure score", _fmt_num(row["score"], 0))
        m2.metric("CMF (20d)", _fmt_num(row.get("cmf_used", row.get("cmf")), 2))
        m3.metric("Signed $ vol", _fmt_num(row.get("signed_rel_dvol"), 2) + "x")
        stock_dvol = row["stock_dollar_volume"] if "stock_dollar_volume" in scored.columns else None
        m4.metric("Constituent $ volume", _fmt_billions(stock_dvol))

        child = pd.DataFrame()
        if not industry_scored.empty and "parent_etf" in industry_scored.columns:
            child = industry_scored[industry_scored["parent_etf"] == etf]
        if not child.empty:
            st.markdown("**Industry groups inside this sector**")
            st.dataframe(ranking_table(child, "Industry"), width="stretch", hide_index=True)
            industry_leaders = industries.get("leaders") or {}
            for _, industry_row in child.sort_values("score", ascending=False).iterrows():
                group_id = industry_row["etf"]
                group_leaders = industry_leaders.get(group_id, pd.DataFrame())
                if group_leaders is None or group_leaders.empty:
                    continue
                st.markdown(
                    f"**{industry_row['sector']}** — score {_fmt_num(industry_row['score'], 0)}, "
                    f"{industry_row['quadrant']}, {str(industry_row['leadership']).lower()} leadership"
                )
                g1, g2 = st.columns(2)
                with g1:
                    st.dataframe(leaders_table(group_leaders, "Leader"), width="stretch", hide_index=True)
                with g2:
                    st.dataframe(leaders_table(group_leaders, "Laggard"), width="stretch", hide_index=True)

        leaders = payload["leaders"].get(etf, pd.DataFrame())
        if leaders is None or leaders.empty:
            st.warning("No constituent leadership data for this sector.")
        else:
            st.markdown("**Sector leaders and laggards (1M RS vs SPY)**")
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(leaders_table(leaders, "Leader"), width="stretch", hide_index=True)
            with c2:
                st.dataframe(leaders_table(leaders, "Laggard"), width="stretch", hide_index=True)

    with tab_industries:
        st.caption(
            "Equal-weight S&P 500 baskets so subgroups like memory are not swallowed by XLK. "
            "Scores use the same absolute vs-SPY scale as sectors (50 = in line). "
            "Industry RS is equal-weight; CMF is still dollar-volume weighted, so mega-caps can dominate the flow layer."
        )
        if industry_scored.empty:
            st.warning("Industry baskets did not load. Click Refresh data in the sidebar.")
        else:
            if industries.get("narrative"):
                st.info(industries["narrative"])
            st.plotly_chart(
                heatmap_figure(
                    industry_rs,
                    "Industry relative strength vs SPY (equal-weight baskets, percentage points)",
                    height=380,
                ),
                width="stretch",
            )
            i_left, i_right = st.columns((1.15, 1))
            with i_left:
                industry_trails = slice_trails(industries.get("rrg_trails"), trail_days)
                st.plotly_chart(
                    rrg_figure(
                        industries["rrg"],
                        industry_trails,
                        title=f"Industry RRG — daily path, last {trail_days} sessions (100 = SPY)",
                    ),
                    width="stretch",
                )
            with i_right:
                st.subheader("Estimated industry flow ranking")
                st.dataframe(
                    ranking_table(industry_scored, "Industry"),
                    width="stretch",
                    hide_index=True,
                    height=420,
                )
            st.plotly_chart(
                rrg_ratio_history_figure(
                    industry_trails,
                    f"Industry RS-Ratio vs SPY — last {trail_days} sessions",
                ),
                width="stretch",
            )
            industry_leaders = industries.get("leaders") or {}
            st.subheader("Industry drill-down")
            industry_options = [
                f"{row.sector} ({row.etf})"
                for row in industry_scored.itertuples(index=False)
            ]
            industry_choice = st.selectbox("Industry group", industry_options)
            industry_id = industry_choice.split("(")[-1].rstrip(")")
            irow = industry_scored[industry_scored["etf"] == industry_id].iloc[0]
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Flow", str(irow["flow_bucket"]), str(irow.get("conviction", "")))
            n2.metric("RRG quadrant", str(irow["quadrant"]))
            n3.metric("% beat SPY", _fmt_pct(irow.get("pct_beat_spy"), 0))
            n4.metric("Persist vs SPY", _fmt_pct(irow.get("persist_1m"), 0))
            group_leaders = industry_leaders.get(industry_id, pd.DataFrame())
            if group_leaders is not None and not group_leaders.empty:
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**Leaders**")
                    st.dataframe(leaders_table(group_leaders, "Leader"), width="stretch", hide_index=True)
                with d2:
                    st.markdown("**Laggards**")
                    st.dataframe(leaders_table(group_leaders, "Laggard"), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
