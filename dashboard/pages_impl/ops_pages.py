"""Quality Inspection, Electrification Readiness, Carbon Tracker, Maintenance Optimiser."""

from __future__ import annotations

import base64
import random

import altair as alt
import api_client
import pandas as pd
import streamlit as st
from api_client import ApiError
from components import provenance_badge, require_agent, require_file, show_api_error
from fleet_data import DATA_DIR, fleet_with_battery_health, fleet_with_compound_risk
from theme import RISK_BANDS

READINESS_CSV = DATA_DIR / "processed" / "fleet_readiness.csv"

# Eight real trucks never saturate a 38.4 h/day workshop, so the schedulers all
# score identically and the comparison table proves nothing. This is the padding
# that makes capacity actually bind.
SIMULATED_BACKLOG_JOBS = 25


# ------------------------------------------------------------------ quality

def render_quality() -> None:
    st.title("Manufacturing Quality Inspection")
    st.caption("Two independent classifiers — cast EV components (defective/OK) and steel "
               "surfaces (which of 6 defect types).")
    if not require_agent("quality", "Quality inspection"):
        return

    tab_cast, tab_surface = st.tabs(["Cast component", "Steel surface (NEU-DET)"])

    with tab_cast:
        provenance_badge("real")
        st.caption("**97.3% accuracy, 0.955 defect recall** on a leakage-free part-level split. "
                   "The vendor's own split shared 97.5% of its test parts with training and "
                   "reported 99.44% — a model trained on it scores 88.08% on genuinely unseen "
                   "castings and rejects 28% of good parts.")
        _inspect_tab("/quality/inspect", "cast_uploader", "Component image", _render_cast_result)

    with tab_surface:
        provenance_badge("real")
        st.caption("6-way defect-type classifier, 99.7% on the held-out NEU-DET split. Every "
                   "image is assumed to already contain a defect — this classifies which type, "
                   "it is not a defective/OK gate.")
        _inspect_tab("/quality/inspect-surface", "surface_uploader", "Surface image",
                     _render_surface_result)


def _inspect_tab(endpoint: str, key: str, label: str, render_result) -> None:
    uploaded = st.file_uploader(label, type=["jpg", "jpeg", "png"], key=key)
    if not uploaded:
        st.caption("Upload an image to run an inspection.")
        return

    col1, col2 = st.columns(2)
    col1.image(uploaded, caption="Uploaded image", width="stretch")
    with st.spinner("Inspecting…"):
        try:
            result = api_client.post_file(endpoint, uploaded.name, uploaded.getvalue(),
                                          uploaded.type or "image/jpeg")
        except ApiError as e:
            show_api_error(e, "Inspection")
            return

    col2.image(base64.b64decode(result["gradcam_overlay_png_base64"]),
               caption="Grad-CAM — where the model looked", width="stretch")
    col2.caption("Coarse by construction: at 128×128 input the final conv layer is 4×4, so this "
                 "is attention, not defect segmentation.")
    render_result(result)


def _render_cast_result(result: dict) -> None:
    verdict = result["verdict"]
    tokens = RISK_BANDS["critical" if verdict == "defective" else "healthy"]
    st.markdown(f'<h3 style="color:{tokens.text}">Verdict: {verdict.upper()} '
                f'({result["confidence"] * 100:.1f}% confidence)</h3>',
                unsafe_allow_html=True)
    if verdict == "defective":
        st.warning(
            "**1-10-100 rule of quality cost:** a defect caught at incoming inspection costs "
            "roughly 10× less to address than the same defect caught after assembly, and 100× "
            "less than a field failure. Reject this batch before it reaches the line.")


def _render_surface_result(result: dict) -> None:
    verdict = result["verdict"].replace("_", " ").replace("-", " ")
    st.markdown(f'<h3 style="color:{RISK_BANDS["watch"].text}">Defect type: {verdict.upper()} '
                f'({result["confidence"] * 100:.1f}% confidence)</h3>',
                unsafe_allow_html=True)
    probs = (pd.DataFrame({"defect_type": list(result["class_probabilities"]),
                           "probability": list(result["class_probabilities"].values())})
             .sort_values("probability", ascending=False))
    st.altair_chart(
        alt.Chart(probs).mark_bar(cornerRadiusEnd=3, color=RISK_BANDS["watch"].accent).encode(
            x=alt.X("probability:Q", title="Softmax probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("defect_type:N", sort="-x", title=None),
            tooltip=["defect_type", alt.Tooltip("probability:Q", format=".3f")]),
        width="stretch")


# ------------------------------------------------------------------ readiness

def render_readiness() -> None:
    st.title("Fleet Electrification Readiness")
    st.caption("Rule-based and inspectable — every score traces to real km/kg/hour constraints "
               "against 3 real Indian commercial EVs, not a black box.")
    if not require_file(READINESS_CSV, "Electrification Readiness",
                        "python train/train_fleet_readiness.py"):
        return

    df = pd.read_csv(READINESS_CSV)
    ready_pct = (df.readiness == "ready").mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("ICE vehicles assessed", len(df))
    c2.metric("Ready to electrify today", f"{ready_pct:.0f}%")
    c3.metric("Not yet viable", f"{100 - ready_pct:.0f}%")

    st.info("**Read this number carefully.** The fleet is synthetic and was labelled by calling "
            f"the very scorer being evaluated, so {ready_pct:.0f}% is a property of the route mix "
            "and the 1.25× range safety margin — not an independent validation. There is no "
            "expert baseline on the other side of the comparison.")

    st.subheader("Readiness by route type")
    by_route = (df.groupby(["route_type", "readiness"]).size()
                  .reset_index(name="count"))
    st.altair_chart(
        alt.Chart(by_route).mark_bar().encode(
            x=alt.X("count:Q", stack="normalize", title="Share of vehicles",
                    axis=alt.Axis(format="%")),
            y=alt.Y("route_type:N", title=None),
            color=alt.Color("readiness:N", title=None, scale=alt.Scale(
                domain=["ready", "not_yet_viable"],
                range=[RISK_BANDS["healthy"].accent, RISK_BANDS["critical"].accent])),
            tooltip=["route_type", "readiness", "count"]),
        width="stretch")
    st.caption("Highway-feeder routes lag because no EV in the catalog covers that daily range — "
               "a real limit of today's commercial options, not a modelling gap.")

    st.subheader("Fleet detail")
    show = df.sort_values("confidence_pct", ascending=False).head(12)
    st.dataframe(
        show[["vehicle_id", "route_type", "daily_distance_km", "avg_payload_kg",
              "readiness", "recommended_model", "price_inr_lakh", "confidence_pct"]],
        width="stretch", hide_index=True)
    st.caption("`confidence_pct` is the margin by which the **cheapest feasible** option clears "
               "its gates, so it is not monotonic in duty-cycle difficulty: a harder route can "
               "score higher because it disqualifies the marginal cheap option.")

    _render_tco()


def _render_tco() -> None:
    """Feasibility vs economics -- the second, external criterion.

    Readiness alone cannot validate itself: the synthetic fleet is labelled by
    the scorer being evaluated. "Is the EV cheaper than the diesel it replaces
    over 5 years?" is a question the project does not get to answer for itself,
    and the two answers genuinely diverge.
    """
    st.divider()
    st.subheader("Is it actually cheaper? — 5-year total cost of ownership")
    if not require_agent("fleet_readiness", "TCO analysis"):
        return

    c1, c2, c3 = st.columns(3)
    distance = c1.slider("Daily distance (km)", 10, 200, 40)
    payload = c2.slider("Average payload (kg)", 50, 900, 580)
    dwell = c3.slider("Depot dwell time (h)", 1.0, 14.0, 3.0)

    # `expected_cycles_remaining` is documented as "the one place the RUL model
    # changes a rupee figure that matters" -- and nothing in the platform passed
    # it. The endpoint accepted it, tco.py implemented the battery-replacement
    # term behind it, and the advertised battery -> TCO link was dead in the UI.
    # This wires it: pick a real truck and its predicted RUL decides whether a
    # pack replacement lands inside the five-year horizon.
    payload_extra: dict = {}
    replacement_note = None
    try:
        fleet = fleet_with_battery_health()
    except ApiError:
        fleet = []
    if fleet:
        options = {"New pack (no replacement modelled)": None}
        options.update({
            f"{a['asset_id']} — {a['predicted_rul_cycles']:.0f} cycles left "
            f"({a['state_of_health_pct']}% SOH)": a["predicted_rul_cycles"]
            for a in sorted(fleet, key=lambda a: a["predicted_rul_cycles"])})
        chosen = st.selectbox(
            "Battery age to assume", list(options), key="tco_battery_asset",
            help="Uses that truck's predicted RUL from the Battery APM agent. If the pack "
                 "will not last the 5-year horizon, a replacement is added to the EV's TCO.")
        cycles = options[chosen]
        if cycles is not None:
            payload_extra["expected_cycles_remaining"] = float(cycles)
            replacement_note = chosen

    try:
        r = api_client.post("/fleet-readiness/score-with-economics", {
            "vehicle_id": "WHAT-IF", "daily_distance_km": float(distance),
            "avg_payload_kg": float(payload), "dwell_time_hours": float(dwell),
            **payload_extra})
    except ApiError as e:
        show_api_error(e, "TCO analysis")
        return

    if r["economics"] is None:
        st.warning(f"No EV in the catalog fits this duty cycle. {r['reason']}")
        return

    e = r["economics"]
    verdict_band = "healthy" if r["verdict"] == "ready_and_economic" else "watch"
    st.markdown(
        f'<h4 style="color:{RISK_BANDS[verdict_band].text}">'
        f'{r["recommended_model"]} — {r["verdict"].replace("_", " ").upper()}</h4>',
        unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("EV, 5 years", f"₹{e['ev']['total_inr']:,}", f"₹{e['cost_per_km_ev']}/km")
    m2.metric("Diesel equivalent", f"₹{e['diesel']['total_inr']:,}",
              f"₹{e['cost_per_km_diesel']}/km")
    m3.metric("EV saving", f"₹{e['ev_saving_inr']:,}",
              delta=f"{'cheaper' if e['ev_cheaper'] else 'MORE EXPENSIVE'}",
              delta_color="normal" if e["ev_cheaper"] else "inverse")

    if replacement_note and e["ev"]["battery_replacement_inr"]:
        st.info(f"Assuming **{replacement_note}**, the pack does not last the five-year "
                f"horizon, so a replacement of **Rs {e['ev']['battery_replacement_inr']:,}** is "
                f"included in the EV column above. This is the Battery APM agent changing a "
                f"rupee figure.")
    elif replacement_note:
        st.caption(f"Assuming {replacement_note}, the pack outlasts the five-year horizon, so no "
                   f"replacement is added.")

    parts = ["purchase", "energy", "maintenance", "battery_replacement", "residual_value"]
    breakdown = pd.DataFrame([
        {"component": p.replace("_", " "), "vehicle": v.upper(),
         "inr": e[v][f"{p}_inr"]}
        for p in parts for v in ("ev", "diesel")])
    st.altair_chart(
        alt.Chart(breakdown).mark_bar().encode(
            x=alt.X("inr:Q", title="₹ over 5 years"),
            y=alt.Y("component:N", sort=parts, title=None),
            color=alt.Color("vehicle:N", title=None, scale=alt.Scale(
                domain=["EV", "DIESEL"],
                range=[RISK_BANDS["healthy"].accent, RISK_BANDS["critical"].accent])),
            yOffset="vehicle:N",
            tooltip=["component", "vehicle", alt.Tooltip("inr:Q", format=",.0f")]),
        width="stretch")

    be = e["breakeven"]
    # Both break-evens are legitimately None when the EV never crosses over
    # inside the searched range. Interpolating a None into the string printed
    # "Break-even: None km/day", which reads as a bug rather than an answer.
    km_be = be["daily_km_above_which_ev_wins"]
    price_be = be["diesel_price_inr_per_litre_above_which_ev_wins"]
    km_text = (f"**Break-even: {km_be} km/day**" if km_be is not None
               else "**No distance break-even in the 1–500 km/day range**")
    price_text = (f", or diesel above ₹{price_be}/L" if price_be is not None else "")
    # Every figure in this sentence is read from the response being displayed.
    # It used to hardcode "the Tata Ace EV", "40 km/day" and "loses ₹150,505"
    # from one observed run, so selecting a battery age printed that stale loss
    # directly beneath a metric reading -₹533,905, and any distance above the
    # break-even printed "loses" beneath a green "cheaper" badge.
    lakh = e["ev"]["purchase_inr"] / 100_000
    saving = e["ev_saving_inr"]
    economics = (
        f"already saves ₹{saving:,} against its diesel twin over five years, past the "
        f"distance at which its ₹{lakh:.2f}L purchase price pays back"
        if e["ev_cheaper"] else
        f"loses ₹{abs(saving):,} against its diesel twin over five years, because a "
        f"₹{lakh:.2f}L purchase price needs distance to recover")
    st.info(
        f"{km_text} for the {r['recommended_model']}{price_text}. "
        "Feasibility and economics are separate gates, and they diverge: the "
        f"{r['recommended_model']} is physically capable on a {distance:g} km/day route and "
        f"{economics}. Ranking by sticker price cannot express that.")
    # A break-even beyond the vehicle's own range gate is not actionable advice --
    # it recommends a route this same page reports as not_yet_viable. The API
    # labels it; surfacing that label is the whole reason it exists.
    if be.get("breakeven_is_reachable") is False:
        st.warning(f"⚠️ {be['distance_breakeven_note']}. On this duty cycle the "
                   f"{r['recommended_model']} does not pay back within five years at any "
                   f"distance it is actually rated for.")
    with st.expander("Assumptions — every one is a mid-range figure, not a citation"):
        st.json(e["assumptions"])


# ------------------------------------------------------------------ carbon

def render_carbon() -> None:
    st.title("Net Zero Carbon Intelligence Tracker")
    st.caption("EV vs. diesel emissions on official Government of India grid factors.")
    if not require_agent("carbon", "Carbon tracker"):
        return
    provenance_badge("real")

    try:
        meta = api_client.get("/carbon/metadata")
        results = {m: api_client.post("/carbon/compute-savings",
                                       {"vehicle_model": m, "daily_distance_km": 60})
                   for m in meta["vehicle_models"]}
    except ApiError as e:
        show_api_error(e, "Carbon tracker")
        return

    st.subheader("India's grid carbon intensity")
    trend = pd.DataFrame({"year": list(meta["grid_emission_factor_trend"]),
                          "factor": list(meta["grid_emission_factor_trend"].values())})
    st.altair_chart(
        alt.Chart(trend).mark_line(point=True, color=RISK_BANDS["watch"].accent).encode(
            x=alt.X("year:N", title=None),
            y=alt.Y("factor:Q", title="tCO2/MWh", scale=alt.Scale(zero=False)),
            tooltip=["year", alt.Tooltip("factor:Q", format=".4f")]),
        width="stretch")
    st.caption("Weighted average grid emission factor, FY2020-21 to FY2024-25. Source: CEA CO2 "
               "Baseline Database v21.0. **Read it honestly: it rose for four of these five years** "
               "(0.6975 → 0.7273) before easing to 0.7097, still 1.75% above the FY2020-21 "
               "baseline. EV emissions improve automatically only once this line falls durably.")

    st.divider()
    st.subheader("EV vs. diesel, per catalog vehicle")
    cols = st.columns(len(results))
    for col, (model, r) in zip(cols, results.items(), strict=True):
        with col:
            st.markdown(f"**{model}**")
            st.metric("EV emissions", f"{r['ev_g_co2_per_km']} g CO2/km")
            st.metric("Diesel equivalent", f"{r['diesel_equivalent_g_co2_per_km']} g CO2/km")
            st.metric("Savings", f"{r['savings_pct']}%")
            st.caption(f"Break-even grid factor **{r['breakeven_grid_factor_tco2_per_mwh']:.3f}** "
                       f"(India is at {r['grid_factor_used']}). Includes "
                       f"{int(r['charger_efficiency_used'] * 100)}% charger efficiency.")
            st.caption(r["diesel_comparison_source"])

    # Range read from the cards above rather than restated, so a grid-factor
    # revision cannot leave the summary contradicting the figures it summarises.
    savings = [r["savings_pct"] for r in results.values()]
    st.info(
        f"Savings are real but modest — roughly **{min(savings)}-{max(savings)}%**, not 90%+ — "
        "because ~70% of India's grid is still fossil-generated. These are Scope 2 figures on "
        "*purchased* electricity, so they include the ~10% lost between meter and battery; "
        "omitting that overstated the Tata Ace by roughly nine points. Grid transmission losses "
        "are Scope 3, out of boundary here, and would erase the Tata Ace advantage entirely — "
        "the API returns that as a labelled `scope3_sensitivity` rather than burying it.")


# ------------------------------------------------------------------ maintenance

def render_maintenance() -> None:
    st.title("Maintenance Operations Optimiser")
    st.caption("Scheduling, not prediction. Combines maintenance urgency from the Battery and "
               "Compound-Risk agents with workshop capacity and charging uptime.")
    if not require_agent("maintenance", "Maintenance Optimiser"):
        return

    st.subheader("Workshop configuration")
    c1, c2, c3, c4 = st.columns(4)
    n_bays = c1.number_input("Service bays", 1, 20, 3)
    technicians = c2.number_input("Technicians", 1, 30, 4)
    hours_per_shift = c3.number_input("Hours/shift", 4.0, 12.0, 8.0)
    shifts_per_day = c4.number_input("Shifts/day", 1, 3, 2)
    st.caption("Benchmarks: 0.7 technicians/bay (PartsTech 2025). Charging uptime assumed 80% — "
               "independently measured functional uptime (UC Berkeley/SLAC, ChargerHelp 2025), "
               "not the higher self-reported figures networks publish.")

    try:
        fleet_rows = fleet_with_compound_risk()
    except ApiError as e:
        show_api_error(e, "Maintenance queue")
        return

    priority_map = {"compound priority": "critical", "single-signal priority": "watch",
                    "routine": "routine", "insufficient data": "routine"}
    jobs = [{"asset_id": r["asset_id"],
             "priority": priority_map.get(r["priority"], "routine"),
             "reason": r["priority"]} for r in fleet_rows]

    st.subheader(f"Maintenance queue — {len(jobs)} vehicles, live from Fleet Command Center")
    st.dataframe(pd.DataFrame(jobs), width="stretch", hide_index=True)

    if st.checkbox("Stress-test with a larger simulated backlog",
                   help="8 real trucks rarely generate enough contention to separate the "
                        "schedulers. Clearly a what-if, not a claim about the real fleet."):
        real_jobs = len(jobs)
        rng = random.Random(7)
        jobs += [{"asset_id": f"SIM-{i + 1:03d}",
                  "priority": rng.choices(["critical", "watch", "routine"],
                                           weights=[0.3, 0.3, 0.4])[0],
                  "reason": "simulated for stress test"} for i in range(SIMULATED_BACKLOG_JOBS)]
        st.caption(f"Testing with {len(jobs)} jobs ({real_jobs} real + "
                   f"{SIMULATED_BACKLOG_JOBS} simulated).")

    if not st.button("Build optimised schedule", type="primary"):
        return

    try:
        result = api_client.post("/maintenance/schedule", {
            "jobs": jobs, "n_bays": n_bays, "technicians": technicians,
            "hours_per_shift": hours_per_shift, "shifts_per_day": shifts_per_day,
        }, timeout=60)
    except ApiError as e:
        show_api_error(e, "Scheduling")
        return

    _render_schedule(result)


def _render_schedule(result: dict) -> None:
    st.divider()
    budget = result["daily_capacity_hours"]
    c1, c2 = st.columns(2)
    c1.metric("Daily capacity", f"{budget} h/day")
    c2.metric("Equivalent job slots", f"{result['daily_capacity_jobs']} jobs/day")

    st.subheader("Every scheduler, including the ones that lose")
    rows = []
    for key in ("greedy_original_overflow", "greedy_jobcount", "optimal_jobcount",
                "greedy", "optimal"):
        s = result[key]
        rows.append({
            "Scheduler": s["method"],
            "Missed deadlines": s["jobs_missing_deadline"],
            "Peak day (h)": s["peak_day_hours"],
            "Executable?": "yes" if s["peak_day_hours"] <= budget + 1e-6 else "NO — overbooked",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # The tie is the usual outcome, not a guaranteed one: on a mixed backlog
    # large enough to make hours bind unevenly the LP does pull ahead, and
    # asserting "the two tie" unconditionally contradicted the count beside it.
    saved = result["improvement"]
    verdict = ("the two tie — which is what Moore-Hodgson predicts when jobs are unit-size and "
               "capacity is a flat count" if saved == 0 else
               "the LP keeps a narrow edge, which is where Moore-Hodgson stops applying: these "
               "jobs are not unit-size once hours differ by priority")
    st.info(
        f"**The like-for-like comparison is the last two rows** — both respect the {budget} h/day "
        f"budget. The LP saves **{saved}** deadline{'' if saved == 1 else 's'} over an "
        "hours-aware greedy. The platform previously advertised a 37.5% improvement; that came "
        "from a bug in its own greedy baseline, which sent already-late jobs to the *earliest* "
        "free slot and so cannibalised capacity from jobs that could still make their deadline. "
        f"With that repaired {verdict}. The job-count rows are kept visible because a negative "
        "result you can check is worth more than one you have to take on trust.")

    st.subheader("Optimised schedule (by day)")
    st.dataframe(
        pd.DataFrame(result["optimal"]["assignments"]).sort_values("scheduled_day"),
        width="stretch", hide_index=True)
    st.caption(result["assumptions"])
    if result["optimal"].get("solver_notes"):
        st.caption(f"Solver notes: {result['optimal']['solver_notes']}")
