"""Fleet Overview, Battery Deep-Dive and the Live Fleet Monitor."""

from __future__ import annotations

from datetime import datetime

import altair as alt
import api_client
import pandas as pd
import streamlit as st
from api_client import ApiError
from components import (
    provenance_badge,
    require_agent,
    require_file,
    risk_pill,
    show_api_error,
)
from fleet_data import BATTERY_FEATURES, fleet_with_battery_health
from theme import RISK_BANDS

BAND_SCALE = alt.Scale(
    domain=list(RISK_BANDS), range=[b.accent for b in RISK_BANDS.values()])


def _guard(title: str) -> bool:
    return (require_file(BATTERY_FEATURES, title, "python train/train_battery_model.py")
            and require_agent("battery", title))


@st.cache_data(ttl=300, show_spinner=False)
def _model_metrics() -> dict:
    """Held-out metrics for the model the BACKEND actually loaded.

    This page used to state "MAE is 21.2 ± 12.4 cycles ... over the 6 NASA
    cells" as a literal caption. The shipped model scores 15.9 ± 15.7 over 5,
    and nothing could have caught the difference: the number had no source the
    UI could check it against. It now comes from /battery/metadata, which reads
    it out of the artifact -- the same pattern that fixed the supply-chain
    WATCH threshold this dashboard used to re-invent.
    """
    try:
        return api_client.get("/battery/metadata").get("metrics") or {}
    except ApiError:
        return {}


def _accuracy_caption() -> str:
    m = _model_metrics()
    if not {"mae_mean", "mae_sd", "n_folds"} <= set(m):
        return ("Model accuracy is reported by `/battery/metadata`, which this backend did "
                "not return — retrain with `python train/train_battery_model.py` to "
                "regenerate an artifact that carries its own metrics.")
    strategy = m.get("training_strategy", "unknown")
    return (f"Model MAE is {m['mae_mean']:.1f} ± {m['mae_sd']:.1f} cycles "
            f"(median {m.get('mae_median', float('nan')):.1f}), leave-one-battery-out over the "
            f"{m['n_folds']} NASA cells with a genuine end-of-life event — so treat differences "
            f"smaller than that as noise. Censored-row policy: `{strategy}`. "
            f"Beats a fade-extrapolation baseline {m.get('beats_baseline_folds', 'n/a')} folds.")


# ------------------------------------------------------------------ overview

def render_overview() -> None:
    st.title("Fleet Overview")
    st.caption("Battery health across the fleet, ranked by urgency.")
    if not _guard("Fleet Overview"):
        return
    try:
        df = pd.DataFrame(fleet_with_battery_health()).sort_values("predicted_rul_cycles")
    except ApiError as e:
        show_api_error(e, "Fleet Overview")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Fleet size", len(df))
    c1.metric("Critical assets", int((df.risk_band == "critical").sum()))
    c2.metric("Watch-list assets", int((df.risk_band == "watch").sum()))
    c2.metric("Healthy assets", int((df.risk_band == "healthy").sum()))
    c3.metric("Avg. state of health", f"{df.state_of_health_pct.mean():.1f}%")
    c3.metric("Min. RUL in fleet", f"{df.predicted_rul_cycles.min():.0f} cycles")

    st.divider()
    st.subheader("Remaining Useful Life, by asset")
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("predicted_rul_cycles:Q", title="Predicted RUL (cycles)"),
            y=alt.Y("asset_id:N", sort="x", title=None),
            color=alt.Color("risk_band:N", scale=BAND_SCALE, title="Risk band"),
            tooltip=["asset_id", "depot", "risk_band",
                     alt.Tooltip("predicted_rul_cycles:Q", title="RUL", format=".0f"),
                     alt.Tooltip("state_of_health_pct:Q", title="SOH %", format=".1f")],
        )
        .properties(height=30 * len(df))
    )
    st.altair_chart(chart, width="stretch")
    st.caption(_accuracy_caption())

    st.subheader("Fleet detail")
    for _, r in df.iterrows():
        with st.container(border=True):
            cols = st.columns([2, 2, 2, 2, 2])
            cols[0].markdown(f"**{r.asset_id}**  \n{r.depot}")
            cols[1].markdown(f"SOH: **{r.state_of_health_pct}%**")
            cols[2].markdown(f"RUL: **{r.predicted_rul_cycles:.0f} cycles**")
            cols[3].markdown(risk_pill(r.risk_band), unsafe_allow_html=True)
            cols[4].markdown(f"Capacity: {r.current_capacity_ah:.3f} Ah")


# ------------------------------------------------------------------ deep dive

def render_deep_dive() -> None:
    st.title("Battery Deep-Dive")
    if not _guard("Battery Deep-Dive"):
        return
    try:
        fleet = fleet_with_battery_health()
    except ApiError as e:
        show_api_error(e, "Battery Deep-Dive")
        return

    chosen = st.selectbox("Select asset", [a["asset_id"] for a in fleet])
    asset = next(a for a in fleet if a["asset_id"] == chosen)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("State of health", f"{asset['state_of_health_pct']}%")
    mae = _model_metrics().get("mae_mean")
    c2.metric("Predicted RUL", f"{asset['predicted_rul_cycles']:.0f} cycles",
              help=(f"± ~{mae:.0f} cycles (model MAE, from /battery/metadata)"
                    if mae is not None else "model MAE unavailable — see /battery/metadata"))
    c3.metric("Current capacity", f"{asset['current_capacity_ah']:.3f} Ah")
    c4.markdown("Risk band")
    c4.markdown(risk_pill(asset["risk_band"]), unsafe_allow_html=True)

    if asset.get("business_impact"):
        bi = asset["business_impact"]
        st.subheader("Business impact")
        b1, b2, b3 = st.columns(3)
        b1.metric("Planned replacement cost", f"₹{bi['planned_replacement_cost_inr']:,}")
        b2.metric("Cost if it fails unplanned", f"₹{bi['unplanned_failure_cost_inr']:,}")
        b3.metric("Avoidable by acting early", f"₹{bi['avoidable_cost_inr']:,}")
        st.info(bi["urgency"])
        st.caption(bi["assumption"])
        st.caption("Note: the avoidable figure is 0.5 × pack_kWh × ₹18,000 — it depends on pack "
                   "size only, not on this asset's predicted RUL.")

    st.subheader("Recent capacity trend")
    # The x axis is the cell's REAL discharge-cycle number, not a 1-based
    # position in the window. It used to be range(1, 11) under the title
    # "Discharge cycle", so a truck sitting at cycle 140 of a 168-cycle cell was
    # drawn as cycles 1-10 -- the axis contradicted the "Cycle n/N" figure shown
    # a few lines above it, and read as a nearly-new cell. Encoded :O so the
    # ticks stay whole cycles instead of interpolating 1.5, 2.5, ...
    n = len(asset["capacity_history"])
    last_cycle = int(asset["current_cycle_number"])
    history = pd.DataFrame({
        "cycle": range(last_cycle - n + 1, last_cycle + 1),
        "capacity_ah": asset["capacity_history"],
    })
    line = (
        alt.Chart(history)
        .mark_line(point=True, color=RISK_BANDS["healthy"].accent)
        .encode(x=alt.X("cycle:O", title=f"Discharge cycle (most recent {n})"),
                y=alt.Y("capacity_ah:Q", title="Capacity (Ah)",
                        scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("cycle:O", title="Discharge cycle"),
                         alt.Tooltip("capacity_ah:Q", format=".3f")])
    )
    eol = (
        alt.Chart(pd.DataFrame({"y": [asset["eol_threshold_ah"]]}))
        .mark_rule(color=RISK_BANDS["critical"].accent, strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    st.altair_chart(line + eol, width="stretch")
    st.caption(f"Real NASA PCoE cycling data. Dashed line is this cell's own end-of-life "
               f"threshold ({asset['eol_threshold_ah']:.3f} Ah = 70% of its early-life capacity).")
    provenance_badge("real")


# ------------------------------------------------------------------ live feed

def render_live_monitor() -> None:
    st.title("Live Fleet Monitor")
    st.caption("Simulated live BMS telemetry. Each reading is a **real** NASA discharge cycle for "
               "that truck's cell, replayed one cycle per tick — not fabricated sensor noise. "
               "Once a truck's recorded history runs out it holds at its last real reading and "
               "says so, rather than looping.")
    if not _guard("Live Fleet Monitor"):
        return
    provenance_badge("real")

    if st.button("Reset simulation to start"):
        try:
            api_client.post("/battery/live-fleet-reset")
            st.session_state.pop("live_rul_history", None)
            st.rerun()
        except ApiError as e:
            show_api_error(e, "Reset")

    @st.fragment(run_every="3s")
    def live_fragment() -> None:
        try:
            # POST: advancing mutates shared state. GET /live-fleet-status is
            # read-only and does not advance.
            readings = api_client.post("/battery/live-fleet-tick")["readings"]
        except ApiError as e:
            # inside a 3s fragment an unguarded failure re-rendered a traceback
            # every 3 seconds with no way back
            show_api_error(e, "Live feed")
            return

        df = pd.DataFrame(readings)
        st.caption(f"Last refreshed {datetime.now().strftime('%H:%M:%S')} — auto-updates every 3s")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trucks live", len(df))
        c2.metric("Critical now", int((df.risk_band == "critical").sum()))
        c3.metric("Avg SOH", f"{df.state_of_health_pct.mean():.1f}%")
        c4.metric("Fully replayed", int(df.historical_data_exhausted.sum()))

        st.divider()
        for _, r in df.sort_values("predicted_rul_cycles").iterrows():
            with st.container(border=True):
                cols = st.columns([1.6, 1.7, 1.2, 1.1, 1.4, 1.4])
                cols[0].markdown(f"**{r.asset_id}**  \n{r.depot}")
                note = " 🔁 replay exhausted" if r.historical_data_exhausted else ""
                cols[1].markdown(f"Cycle {r.cycle_number}/{r.total_cycles_recorded}{note}")
                cols[2].markdown(f"{r.voltage_v} V  \n{r.current_a} A")
                cols[3].markdown(f"{r.temp_battery_c} °C")
                cols[4].markdown(f"RUL: **{r.predicted_rul_cycles:.0f}**  \n"
                                  f"SOH: {r.state_of_health_pct}%")
                cols[5].markdown(risk_pill(r.risk_band), unsafe_allow_html=True)

        history = st.session_state.setdefault("live_rul_history", {})
        for _, r in df.iterrows():
            history.setdefault(r.asset_id, []).append(r.predicted_rul_cycles)
            history[r.asset_id] = history[r.asset_id][-50:]  # bounded

        st.divider()
        chosen = st.selectbox("Live RUL trend for", sorted(history), key="live_trend_asset")
        series = history.get(chosen, [])
        if len(series) > 1:
            trend = pd.DataFrame({"tick": range(1, len(series) + 1), "rul": series})
            st.altair_chart(
                alt.Chart(trend).mark_line(color=RISK_BANDS["healthy"].accent).encode(
                    x=alt.X("tick:Q", title="Refresh",
                            axis=alt.Axis(tickMinStep=1, format="d")),
                    y=alt.Y("rul:Q", title="Predicted RUL (cycles)",
                            scale=alt.Scale(zero=False))),
                width="stretch")
        else:
            st.caption("The trend builds as more refreshes arrive this session.")

    live_fragment()
