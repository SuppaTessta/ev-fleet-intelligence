"""Fleet Command Center — the cross-agent correlation view, and the landing page."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from api_client import ApiError
from components import (
    empty_state,
    priority_icon,
    provenance_badge,
    require_agent,
    require_file,
    risk_pill,
    show_api_error,
)
from fleet_data import BATTERY_FEATURES, fleet_with_compound_risk

TITLE = "Fleet Command Center"


def render() -> None:
    st.title(TITLE)
    st.caption("Where battery health, supplier risk and quality signals line up on the *same* "
               "truck. Individually none of these might look urgent.")

    if not require_file(BATTERY_FEATURES, TITLE,
                        "python train/train_battery_model.py"):
        return
    if not require_agent("battery", TITLE) or not require_agent("compound_risk", TITLE):
        return

    try:
        rows = fleet_with_compound_risk()
    except ApiError as e:
        show_api_error(e, TITLE)
        return

    if not rows:
        empty_state("No vehicles in the demo fleet.")
        return

    df = pd.DataFrame(rows).sort_values("compound_score", ascending=False)
    n_compound = int((df.priority == "compound priority").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compound-priority trucks", n_compound)
    c2.metric("Single-signal priority", int((df.priority == "single-signal priority").sum()))
    c3.metric("Routine", int((df.priority == "routine").sum()))
    c4.metric("Insufficient data", int((df.priority == "insufficient data").sum()),
              help="Assets with no bill-of-materials linkage. Counted separately rather than "
                   "escalated -- missing data is not a risk signal.")

    if n_compound:
        exposure = int(df[df.priority == "compound priority"].avoidable_cost_inr.sum())
        st.warning(
            f"**{n_compound} truck(s) show problems across multiple systems at once.** These would "
            f"not necessarily surface as top priority from any single agent alone. Estimated "
            f"avoidable cost from battery risk on these, if acted on now: **₹{exposure:,}**.")
        st.caption(
            "That figure is a per-pack cost model, not a measured saving: it is "
            "0.5 × pack_kWh × ₹18,000, and every demo truck shares the same illustrative "
            "21.3 kWh pack. Treat it as an order of magnitude.")

    st.divider()
    for _, r in df.iterrows():
        with st.container(border=True):
            cols = st.columns([2, 2, 2.4, 2, 2])
            cols[0].markdown(f"{priority_icon(r.priority)} **{r.asset_id}**  \n{r.depot}")
            cols[1].markdown(f"Battery: {risk_pill(r.battery_risk_band)}", unsafe_allow_html=True)

            if r.asset_known_to_bom:
                cols[2].markdown(
                    f"Supplier ({r.cell_supplier_id}): {risk_pill(r.supplier_risk_level)}"
                    f"  \n{r.supplier_flagged_pct}% of shipments flagged",
                    unsafe_allow_html=True)
                # "unknown" is its own band, not silently folded into healthy --
                # a row could otherwise read HEALTHY while reporting 2/3 signals active
                quality_band = {"defective": "critical", "ok": "healthy"}.get(
                    r.quality_verdict, "unknown")
                cols[3].markdown(f"Last quality check: {risk_pill(quality_band)}",
                                  unsafe_allow_html=True)
            else:
                cols[2].markdown("Supplier: —  \n_not in the BOM linkage_")
                cols[3].markdown("Last quality check: —")

            signals = f"{r.active_signal_count}/3 signals active"
            if r.unknown_signal_count:
                signals += f"  \n{r.unknown_signal_count} unknown"
            cols[4].markdown(f"**{r.priority.upper()}**  \n{signals}")

    st.divider()
    provenance_badge("illustrative")
    st.caption(
        "Each truck's supplier and last-quality-check linkage is a fixed mapping standing in for "
        "a real bill-of-materials feed, which this project has no access to. The battery, "
        "supplier-risk and quality models it correlates are all separately trained — only the "
        "cross-linking is a stand-in.")
