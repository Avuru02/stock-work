"""Composite 0-100 rotation score, regime, and narrative."""

from __future__ import annotations

import pandas as pd

from config import (
    CYCLICAL_ETFS,
    DEFENSIVE_ETFS,
    INFLOW_THRESHOLD,
    OUTFLOW_THRESHOLD,
    SCORE_WEIGHTS,
)


def _pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average") * 100


def composite_score(
    rs: pd.DataFrame,
    rrg: pd.DataFrame,
    etf_flow: pd.DataFrame,
    breadth: pd.DataFrame,
) -> pd.DataFrame:
    frame = rs.merge(rrg[["etf", "rs_ratio", "rs_momentum", "quadrant"]], on="etf", how="left")
    flow_cols = [c for c in ["etf", "dollar_volume", "rel_dollar_volume", "cmf", "flow_proxy"] if c in etf_flow.columns]
    frame = frame.merge(etf_flow[flow_cols], on="etf", how="left")
    breadth_cols = [
        c
        for c in [
            "etf",
            "pct_above_50dma",
            "pct_up_1m",
            "new_highs_20",
            "new_lows_20",
            "leadership",
            "n_stocks",
        ]
        if c in breadth.columns
    ]
    frame = frame.merge(breadth[breadth_cols], on="etf", how="left")

    components = pd.DataFrame(index=frame.index)
    components["rs_1m"] = _pct_rank(frame["rs_1m"])
    components["rs_momentum"] = _pct_rank(frame["rs_momentum"])
    components["volume_expansion"] = _pct_rank(frame["rel_dollar_volume"])
    components["cmf"] = _pct_rank(frame["cmf"])
    components["breadth"] = _pct_rank(frame["pct_above_50dma"])

    score = pd.Series(0.0, index=frame.index)
    for name, weight in SCORE_WEIGHTS.items():
        score = score + components[name].fillna(50.0) * weight
    frame["score"] = score
    frame["rs_1m_pctile"] = components["rs_1m"]
    frame["rs_momentum_pctile"] = components["rs_momentum"]
    frame["volume_pctile"] = components["volume_expansion"]
    frame["cmf_pctile"] = components["cmf"]
    frame["breadth_pctile"] = components["breadth"]

    def bucket(value: float) -> str:
        if value >= INFLOW_THRESHOLD:
            return "Inflow"
        if value <= OUTFLOW_THRESHOLD:
            return "Outflow"
        return "Neutral"

    frame["flow_bucket"] = frame["score"].map(bucket)
    return frame.sort_values("score", ascending=False).reset_index(drop=True)


def risk_regime(rs: pd.DataFrame) -> dict[str, float | str]:
    cyclicals = rs[rs["etf"].isin(CYCLICAL_ETFS)]["rs_1m"].mean()
    defensives = rs[rs["etf"].isin(DEFENSIVE_ETFS)]["rs_1m"].mean()
    spread = float(cyclicals - defensives)
    if spread > 0.005:
        label = "Risk-On"
    elif spread < -0.005:
        label = "Risk-Off"
    else:
        label = "Mixed"
    return {
        "label": label,
        "cyclical_rs_1m": float(cyclicals),
        "defensive_rs_1m": float(defensives),
        "spread": spread,
    }


def rotation_narrative(score_df: pd.DataFrame) -> str:
    inflow = score_df[score_df["flow_bucket"] == "Inflow"]
    outflow = score_df[score_df["flow_bucket"] == "Outflow"]
    into = ", ".join(inflow["sector"].head(3).tolist()) or "no sector"
    out_of = ", ".join(outflow["sector"].head(3).tolist()) or "no sector"
    leader = score_df.iloc[0]
    lead_note = ""
    if "leadership" in leader and pd.notna(leader["leadership"]):
        lead_note = f" Leadership in {leader['sector']} is {str(leader['leadership']).lower()}."
    return (
        f"Estimated money is rotating into {into} and out of {out_of}. "
        f"{leader['sector']} ranks first with a rotation score of {leader['score']:.0f} "
        f"({leader['quadrant']})."
        f"{lead_note}"
    )
