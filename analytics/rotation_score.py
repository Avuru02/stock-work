"""Composite pressure score with confirmation gates, not rank-only labels.

The 0-100 score is absolute vs SPY (50 = in line). Being the least-bad name
in a weak tape does not inflate the score. Inflow/Outflow still require the
1-month relative-strength gate plus confirming tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BREADTH_CONFIRM,
    CONFIRM_NEEDED,
    CYCLICAL_ETFS,
    DEFENSIVE_ETFS,
    INFLOW_THRESHOLD,
    NARROW_PENALTY,
    OUTFLOW_THRESHOLD,
    PERSIST_CONFIRM,
    RRG_MOM_SCORE_SCALE,
    RS_1M_SCORE_SCALE,
    RS_1W_SCORE_SCALE,
    SCORE_WEIGHTS,
    VOL_SCORE_SCALE,
)


def _unit_score(series: pd.Series, scale: float) -> pd.Series:
    """Map a signed metric onto 0-100. 0 -> 50. About ±scale -> 12 / 88."""
    values = pd.to_numeric(series, errors="coerce")
    mapped = 50.0 + 50.0 * np.tanh(values.to_numpy(dtype=float) / scale)
    return pd.Series(mapped, index=series.index)


def _breadth_share(row: pd.Series) -> float:
    if "pct_beat_spy" in row.index and pd.notna(row.get("pct_beat_spy")):
        return float(row["pct_beat_spy"])
    value = row.get("pct_above_50dma", np.nan)
    return float(value) if pd.notna(value) else np.nan


def _bull_flags(row: pd.Series) -> dict[str, bool]:
    rel_dvol = row.get("rel_dollar_volume", np.nan)
    rs_1w = row.get("rs_1w", np.nan)
    rs_1m = row.get("rs_1m", np.nan)
    breadth = _breadth_share(row)
    rrg_ok = row.get("quadrant") in ("Leading", "Improving")
    rs_up = bool(pd.notna(rs_1m) and rs_1m > 0)
    return {
        "1w RS": bool(pd.notna(rs_1w) and rs_1w > 0),
        "RRG turn": bool(rrg_ok and rs_up),
        "CMF": bool(pd.notna(row.get("cmf_used")) and row.get("cmf_used") > 0),
        "Volume w/ strength": bool(
            pd.notna(rel_dvol) and pd.notna(rs_1w) and rel_dvol >= 1.0 and rs_1w > 0
        ),
        "Breadth": bool(pd.notna(breadth) and breadth >= BREADTH_CONFIRM),
        "Persistence": bool(pd.notna(row.get("persist_1m")) and row.get("persist_1m") >= PERSIST_CONFIRM),
    }


def _bear_flags(row: pd.Series) -> dict[str, bool]:
    rel_dvol = row.get("rel_dollar_volume", np.nan)
    rs_1w = row.get("rs_1w", np.nan)
    rs_1m = row.get("rs_1m", np.nan)
    breadth = _breadth_share(row)
    rrg_ok = row.get("quadrant") in ("Lagging", "Weakening")
    rs_down = bool(pd.notna(rs_1m) and rs_1m < 0)
    return {
        "1w RS": bool(pd.notna(rs_1w) and rs_1w < 0),
        "RRG turn": bool(rrg_ok and rs_down),
        "CMF": bool(pd.notna(row.get("cmf_used")) and row.get("cmf_used") < 0),
        "Volume w/ weakness": bool(
            pd.notna(rel_dvol) and pd.notna(rs_1w) and rel_dvol >= 1.0 and rs_1w < 0
        ),
        "Breadth": bool(pd.notna(breadth) and breadth < BREADTH_CONFIRM),
        "Persistence": bool(pd.notna(row.get("persist_1m")) and row.get("persist_1m") <= (1 - PERSIST_CONFIRM)),
    }


def _classify(row: pd.Series) -> str:
    leadership = str(row.get("leadership", ""))
    rs_1m = row.get("rs_1m", np.nan)
    score = row.get("score", np.nan)
    bull_n = int(row.get("bull_confirms", 0))
    bear_n = int(row.get("bear_confirms", 0))
    narrow = leadership == "Narrow"
    beat = _breadth_share(row)
    if (
        pd.notna(rs_1m)
        and rs_1m > 0
        and not narrow
        and pd.notna(beat)
        and beat >= BREADTH_CONFIRM
        and bull_n >= CONFIRM_NEEDED
        and pd.notna(score)
        and score >= INFLOW_THRESHOLD
    ):
        return "Inflow"
    if (
        pd.notna(rs_1m)
        and rs_1m < 0
        and pd.notna(beat)
        and beat < BREADTH_CONFIRM
        and bear_n >= CONFIRM_NEEDED
        and pd.notna(score)
        and score <= OUTFLOW_THRESHOLD
    ):
        return "Outflow"
    return "Neutral"


def _flag_note(flags: dict[str, bool]) -> str:
    passed = [name for name, ok in flags.items() if ok]
    return ", ".join(passed) if passed else "none"


def composite_score(
    rs: pd.DataFrame,
    rrg: pd.DataFrame,
    etf_flow: pd.DataFrame,
    breadth: pd.DataFrame,
) -> pd.DataFrame:
    frame = rs.merge(rrg[["etf", "rs_ratio", "rs_momentum", "quadrant"]], on="etf", how="left")
    flow_cols = [
        c
        for c in [
            "etf",
            "dollar_volume",
            "rel_dollar_volume",
            "rel_dollar_volume_1d",
            "cmf",
            "flow_proxy",
            "stock_cmf",
            "stock_rel_dollar_volume",
            "stock_dollar_volume",
            "stock_flow_proxy",
        ]
        if c in etf_flow.columns
    ]
    frame = frame.merge(etf_flow[flow_cols], on="etf", how="left")
    breadth_cols = [
        c
        for c in [
            "etf",
            "pct_above_50dma",
            "pct_up_1m",
            "pct_beat_spy",
            "new_highs_20",
            "new_lows_20",
            "leadership",
            "n_stocks",
        ]
        if c in breadth.columns
    ]
    frame = frame.merge(breadth[breadth_cols], on="etf", how="left")

    rs_1w = frame["rs_1w"] if "rs_1w" in frame.columns else pd.Series(np.nan, index=frame.index)
    vol = frame["rel_dollar_volume"] if "rel_dollar_volume" in frame.columns else pd.Series(np.nan, index=frame.index)
    if "stock_rel_dollar_volume" in frame.columns:
        vol = frame["stock_rel_dollar_volume"].fillna(vol)
    frame["rel_dollar_volume"] = vol
    frame["signed_rel_dvol"] = vol * np.sign(rs_1w.fillna(0.0))

    cmf = frame["cmf"] if "cmf" in frame.columns else pd.Series(np.nan, index=frame.index)
    if "stock_cmf" in frame.columns:
        cmf = frame["stock_cmf"].fillna(cmf)
    frame["cmf_used"] = cmf

    beat = (
        frame["pct_beat_spy"]
        if "pct_beat_spy" in frame.columns
        else frame.get("pct_above_50dma", pd.Series(np.nan, index=frame.index))
    )
    persist = frame["persist_1m"] if "persist_1m" in frame.columns else pd.Series(np.nan, index=frame.index)

    components = pd.DataFrame(index=frame.index)
    components["rs_1m"] = _unit_score(frame["rs_1m"], RS_1M_SCORE_SCALE)
    components["rs_1w"] = _unit_score(rs_1w, RS_1W_SCORE_SCALE)
    components["rs_momentum"] = _unit_score(frame["rs_momentum"] - 100.0, RRG_MOM_SCORE_SCALE)
    components["volume_signed"] = _unit_score(frame["signed_rel_dvol"], VOL_SCORE_SCALE)
    components["cmf"] = (cmf.clip(-1, 1) + 1) * 50
    components["breadth"] = beat * 100
    components["persist_1m"] = persist * 100

    score = pd.Series(0.0, index=frame.index)
    for name, weight in SCORE_WEIGHTS.items():
        score = score + components[name].fillna(50.0) * weight
    frame["score"] = score
    narrow = frame.get("leadership", pd.Series("", index=frame.index)).eq("Narrow")
    frame.loc[narrow, "score"] = frame.loc[narrow, "score"] - NARROW_PENALTY
    frame["rs_1m_abs"] = components["rs_1m"]
    frame["rs_1w_abs"] = components["rs_1w"]
    frame["rs_momentum_abs"] = components["rs_momentum"]
    frame["volume_abs"] = components["volume_signed"]
    frame["cmf_abs"] = components["cmf"]
    frame["breadth_abs"] = components["breadth"]
    frame["persist_abs"] = components["persist_1m"]

    bull_counts = []
    bear_counts = []
    bull_notes = []
    bear_notes = []
    for _, row in frame.iterrows():
        bull = _bull_flags(row)
        bear = _bear_flags(row)
        bull_counts.append(int(sum(bull.values())))
        bear_counts.append(int(sum(bear.values())))
        bull_notes.append(_flag_note(bull))
        bear_notes.append(_flag_note(bear))
    frame["bull_confirms"] = bull_counts
    frame["bear_confirms"] = bear_counts
    frame["confirm_total"] = len(_bull_flags(frame.iloc[0])) if len(frame) else 0
    frame["flow_bucket"] = frame.apply(_classify, axis=1)
    show_bear = [
        bucket == "Outflow" or (bucket == "Neutral" and bear_counts[i] > bull_counts[i])
        for i, bucket in enumerate(frame["flow_bucket"].tolist())
    ]
    frame["confirm_flags"] = [
        bear_notes[i] if use_bear else bull_notes[i] for i, use_bear in enumerate(show_bear)
    ]
    total = int(frame["confirm_total"].iloc[0]) if len(frame) else 0
    frame["conviction"] = [
        f"{bear_counts[i]}/{total} bear" if use_bear else f"{bull_counts[i]}/{total} bull"
        for i, use_bear in enumerate(show_bear)
    ]
    order = {"Inflow": 0, "Neutral": 1, "Outflow": 2}
    frame["flow_order"] = frame["flow_bucket"].map(order)
    frame = frame.sort_values(["flow_order", "score"], ascending=[True, False]).drop(columns=["flow_order"])
    return frame.reset_index(drop=True)


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


def rotation_narrative(score_df: pd.DataFrame, noun: str = "sector") -> str:
    if score_df.empty:
        return f"No {noun} data."
    inflow = score_df[score_df["flow_bucket"] == "Inflow"]
    outflow = score_df[score_df["flow_bucket"] == "Outflow"]
    if inflow.empty and outflow.empty:
        top = score_df.sort_values("score", ascending=False).iloc[0]
        return (
            f"No confirmed rotation. {noun.capitalize()} signals disagree. "
            f"Highest pressure is {top['sector']} ({top['score']:.0f}, {top['quadrant']}, "
            f"{top.get('conviction', '?')}) but it does not clear the gate "
            f"(must beat SPY over 1 month, have a majority of names beating SPY, "
            f"not be narrow, and pass {CONFIRM_NEEDED} confirming tests)."
        )
    into = ", ".join(inflow["sector"].head(3).tolist()) if not inflow.empty else "nothing confirmed"
    out_of = ", ".join(outflow["sector"].head(3).tolist()) if not outflow.empty else "nothing confirmed"
    almost = score_df[
        (score_df["flow_bucket"] == "Neutral")
        & (score_df["rs_1m"] > 0)
        & (score_df["score"] >= INFLOW_THRESHOLD)
    ].sort_values("score", ascending=False)
    almost_txt = ""
    if not almost.empty:
        row = almost.iloc[0]
        beat = row.get("pct_beat_spy", float("nan"))
        beat_txt = f", {beat:.0%} of names beating SPY" if pd.notna(beat) else ""
        almost_txt = (
            f" {row['sector']} has high pressure ({row['score']:.0f}{beat_txt}) "
            f"but is not a confirmed inflow."
        )
    if not inflow.empty:
        leader = inflow.iloc[0]
        lead_note = ""
        if "leadership" in leader and pd.notna(leader["leadership"]):
            lead_note = f" Leadership in {leader['sector']} is {str(leader['leadership']).lower()}."
        persist = leader.get("persist_1m", float("nan"))
        persist_txt = f" It beat SPY on {persist:.0%} of the last 21 sessions." if pd.notna(persist) else ""
        return (
            f"Confirmed rotation into {into}; confirmed outflow from {out_of}. "
            f"{leader['sector']} leads among confirmed inflows "
            f"({leader['score']:.0f}, {leader['quadrant']}, {leader.get('conviction', '?')})."
            f"{persist_txt}{lead_note}{almost_txt}"
        )
    worst = outflow.iloc[0]
    return (
        f"No confirmed inflows. Confirmed outflow from {out_of}. "
        f"{worst['sector']} is the weakest ({worst['score']:.0f}, {worst['quadrant']})."
        f"{almost_txt}"
    )
