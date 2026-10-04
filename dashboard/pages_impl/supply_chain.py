"""Supply Chain Risk — anomaly detection over supplier shipment records."""

from __future__ import annotations

import altair as alt
import api_client
import pandas as pd
import streamlit as st
from api_client import ApiError
from components import provenance_badge, require_agent, require_file, risk_pill, show_api_error
from fleet_data import DATA_DIR
from theme import RISK_BANDS

TITLE = "Supply Chain Risk & Traceability"
SHIPMENTS_CSV = DATA_DIR / "processed" / "supply_chain_shipments.csv"


def _band_for(row) -> str:
    """The band the BACKEND assigned, not one this UI invents.

    This page used to render `"critical" if r.risk_score > 0.6 else "watch"`.
    The backend's real WATCH threshold is 0.1776 and CRITICAL is the model's own
    binary flag, so 54 shipments the backend called CRITICAL were painted WATCH
    here -- and the adjacent tab, which rendered the backend's band, disagreed
    with this one about the same shipment.
    """
    if row.flagged == 1:
        return "critical"
    return "watch" if row.risk_score >= _watch_threshold() else "healthy"


@st.cache_data(show_spinner=False)
def _watch_threshold() -> float:
    """The threshold lives in the model artifact. Read it, do not guess it."""
    try:
        return float(api_client.get("/risk/metadata")["watch_threshold"])
    except (ApiError, KeyError):
        # 0.75 quantile of the shipped score distribution -- the same definition
        # the training script uses, computed from the data we already have here
        df = pd.read_csv(SHIPMENTS_CSV)
        return float(df["risk_score"].quantile(0.75))


def render() -> None:
    st.title(TITLE)
    st.caption("Isolation Forest anomaly detection over supplier shipment records.")
    provenance_badge("synthetic")
    st.caption("No real multi-tier battery-material shipment dataset is public at this scale. "
               "Anomalies are injected as 5 named archetypes so evaluation means something: "
               "held-out AUC-PR **0.807 ± 0.055** across 20 stratified splits, against a "
               "prevalence baseline of 0.079.")

    if not require_file(SHIPMENTS_CSV, TITLE, "python train/train_risk_model.py"):
        return

    df = pd.read_csv(SHIPMENTS_CSV)
    flagged = df[df.flagged == 1].sort_values("risk_score", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Shipments monitored", len(df))
    c2.metric("Flagged as anomalous", len(flagged))
    c3.metric("Materials tracked", df.material.nunique())

    tab_flagged, tab_score = st.tabs(["Flagged shipments", "Score a new shipment"])

    with tab_flagged:
        st.subheader("Highest-risk shipments")
        for _, r in flagged.head(10).iterrows():
            with st.container(border=True):
                cols = st.columns([2, 2, 2, 3])
                cols[0].markdown(f"**{r.shipment_id}**  \n{r.material}")
                cols[1].markdown(f"Supplier: {r.supplier_id}  \n{r.region}")
                cols[2].markdown(risk_pill(_band_for(r)), unsafe_allow_html=True)
                cols[2].markdown(f"Score: **{r.risk_score:.2f}**")
                cols[3].markdown(f"_{r.anomaly_type.replace('_', ' ')}_")

        st.divider()
        st.subheader("Supplier concentration risk by material")
        conc = (df.groupby("material")["supplier_concentration_pct"].max()
                  .sort_values(ascending=False).reset_index())
        chart = (
            alt.Chart(conc)
            .mark_bar(cornerRadiusEnd=3, color=RISK_BANDS["critical"].accent)
            .encode(
                x=alt.X("supplier_concentration_pct:Q",
                        title="Largest single-supplier share (%)"),
                y=alt.Y("material:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("material:N", title="Material"),
                         alt.Tooltip("supplier_concentration_pct:Q",
                                     title="Max share (%)", format=".1f")],
            )
            .properties(height=28 * len(conc))
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Materials where one supplier holds a large share of total volume — a single "
                   "disruption there hits the whole material category.")

    with tab_score:
        _render_scoring_form()


def _render_scoring_form() -> None:
    st.subheader("Score a hypothetical shipment")
    if not require_agent("risk", "Shipment scoring"):
        return

    col1, col2 = st.columns(2)
    lead_dev = col1.slider("Lead time deviation (%)", -20, 250, 5)
    price_dev = col1.slider("Price deviation (%)", -20, 100, 2)
    reject_rate = col1.slider("Reject rate (%)", 0.0, 35.0, 1.8)
    reject_trend = col1.slider(
        "Reject rate trend (change per shipment, last 3)", -2.0, 3.0, 0.0,
        help="0 = flat. Positive = climbing, which is what triggers an early WATCH even when "
             "the absolute reject rate is not extreme yet.")
    audit_days = col1.slider("Days since last audit", 0, 365, 40)
    single_sourced = col2.checkbox("Single-sourced (no backup supplier)")
    geo_risk = col2.slider("Geopolitical risk (0-1)", 0.0, 1.0, 0.2)
    concentration = col2.slider("Supplier concentration (%)", 0, 100, 25)
    volume_dev = col2.slider("Order volume deviation (%)", -50, 300, 5)

    if not st.button("Score this shipment", type="primary"):
        return

    try:
        result = api_client.post("/risk/score-shipment", {
            "shipment_id": "MANUAL-TEST-001",
            "lead_time_deviation_pct": lead_dev, "price_deviation_pct": price_dev,
            "reject_rate_pct": reject_rate, "days_since_last_audit": audit_days,
            "single_sourced": int(single_sourced), "geopolitical_risk": geo_risk,
            "supplier_concentration_pct": concentration, "volume_deviation_pct": volume_dev,
            "reject_rate_trend": reject_trend, "price_trend": 0.0,
        })
    except ApiError as e:
        show_api_error(e, "Scoring")
        return

    st.markdown(risk_pill(result["risk_band"]), unsafe_allow_html=True)
    st.metric("Risk score", f"{result['risk_score']:.2f}")
    st.write(result["reason"])
    st.caption(_archetype_recall_caption())


@st.cache_data(ttl=300, show_spinner=False)
def _archetype_recall_caption() -> str:
    """Per-archetype recall, read from the model artifact via /risk/metadata.

    This was a hardcoded caption reading "concentration 0.39". The dataset it
    claims to be measured on gives 0.46 -- one shipment's worth of drift in a
    figure no test could check, because it had no source. Same fix as the
    battery MAE caption and the WATCH threshold before it: ask the artifact.
    """
    try:
        recall = api_client.get("/risk/metadata").get("archetype_recall") or {}
    except ApiError:
        recall = {}
    if not recall:
        return ("Per-archetype recall of the CRITICAL flag is reported by `/risk/metadata`, "
                "which this backend did not return — re-run `python train/train_risk_model.py` "
                "to regenerate an artifact that carries it.")
    ranked = sorted(recall.items(), key=lambda kv: -kv[1]["recall"])
    parts = ", ".join(f"{name.replace('_', ' ')} {e['recall']:.2f} (n={e['n']})"
                      for name, e in ranked)
    worst = ranked[-1][0].replace("_", " ")
    return (f"Per-archetype recall of the binary CRITICAL flag, on the shipped dataset: "
            f"{parts}. The WATCH band exists because the weakest of these — {worst} — is "
            f"caught poorly by the flag alone.")
