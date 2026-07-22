"""
EV Fleet Intelligence — unified dashboard.

Calls the FastAPI backend (must be running separately on :8000) so the
architecture stays properly decoupled: this file is purely presentation.

Run:  streamlit run dashboard/app.py   (from the project root)
"""

import base64
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

RISK_COLORS = {"healthy": "#00A676", "watch": "#F2A93B", "critical": "#E5484D"}
# Solid brand colors above read at only ~2-3.9:1 contrast as white-on-pill text
# (WCAG AA needs 4.5:1) -- "watch" in particular is close to unreadable. These
# darker variants are for text/accents on light backgrounds instead; the
# original bright colors remain as small non-text dot indicators, which only
# need to meet the lower 3:1 UI-component bar.
RISK_TEXT_COLORS = {"healthy": "#00875A", "watch": "#B45309", "critical": "#C53030"}
RISK_BG_TINTS = {"healthy": "rgba(0,166,118,0.12)", "watch": "rgba(242,169,59,0.16)", "critical": "rgba(229,72,77,0.12)"}

st.set_page_config(page_title="EV Fleet Intelligence", page_icon="🔋", layout="wide")

# ---- minimal theming: avoid Streamlit's bare default look ----
st.markdown("""
<style>
    .stMetric {
        background: #F7F9F8; border-radius: 10px; padding: 14px 18px;
        border: 1px solid #E7ECEA;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-weight: 500; color: #5C6F6A; letter-spacing: 0.01em; }
    .risk-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 12px 4px 9px; border-radius: 999px;
        font-weight: 600; font-size: 0.8rem; letter-spacing: 0.02em;
    }
    .risk-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    h1 { border-bottom: 3px solid #00A676; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


def api_alive() -> bool:
    try:
        requests.get(f"{API_BASE}/", timeout=2)
        return True
    except requests.exceptions.ConnectionError:
        return False


def risk_pill(risk_band: str) -> str:
    dot = RISK_COLORS.get(risk_band, "#888")
    text = RISK_TEXT_COLORS.get(risk_band, "#5C6F6A")
    bg = RISK_BG_TINTS.get(risk_band, "#F0F0F0")
    return (f'<span class="risk-pill" style="background:{bg};color:{text}">'
            f'<span class="risk-dot" style="background:{dot}"></span>{risk_band.upper()}</span>')


@st.cache_data
def build_fleet_snapshots():
    """Builds a simulated fleet from real NASA battery capacity trajectories —
    each 'truck' is a window of real cycling data cut off at a different
    point in life, so the fleet spans healthy -> critical realistically."""
    df = pd.read_csv(DATA_DIR / "processed" / "battery_features.csv")
    cutoffs = [
        ("B0005", 30, "EV-TRUCK-01", "Chakan Depot"),
        ("B0005", 140, "EV-TRUCK-02", "Chakan Depot"),
        ("B0006", 50, "EV-TRUCK-03", "Pune Ring Road"),
        ("B0006", 120, "EV-TRUCK-04", "Pune Ring Road"),
        ("B0006", 165, "EV-TRUCK-05", "Pune Ring Road"),
        ("B0007", 40, "EV-TRUCK-06", "Hinjewadi Route"),
        ("B0007", 100, "EV-TRUCK-07", "Hinjewadi Route"),
        ("B0007", 160, "EV-TRUCK-08", "Hinjewadi Route"),
    ]
    fleet = []
    for battery, cutoff, asset_id, depot in cutoffs:
        full_history = df[df.battery_id == battery].sort_values("discharge_cycle_num")
        rated_capacity = full_history["capacity"].iloc[:5].max()  # true early-life peak, not window-limited
        window = full_history[full_history.discharge_cycle_num <= cutoff].tail(10)
        fleet.append({
            "asset_id": asset_id, "depot": depot,
            "capacity_history": window["capacity"].round(3).tolist(),
            "current_cycle_number": int(cutoff),
            "rated_capacity_ah": round(float(rated_capacity), 3),
            "pack_kwh": 21.3,  # illustrative — matches the Tata Ace EV class used in Fleet Readiness
        })
    return fleet


def call_battery_agent(asset_id, capacity_history, current_cycle_number=None,
                        rated_capacity_ah=None, pack_kwh=None):
    payload = {"asset_id": asset_id, "capacity_history_ah": capacity_history}
    if current_cycle_number is not None:
        payload["current_cycle_number"] = current_cycle_number
    if rated_capacity_ah is not None:
        payload["rated_capacity_ah"] = rated_capacity_ah
    if pack_kwh is not None:
        payload["pack_kwh"] = pack_kwh
    resp = requests.post(f"{API_BASE}/battery/predict-rul", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🔋 EV Fleet Intelligence")
st.sidebar.caption("ET AI Hackathon 2.0 — Problem Statement #3")
section = st.sidebar.radio("Navigate", [
    "Fleet Command Center", "Fleet Overview", "Battery Deep-Dive", "Quality Inspection",
    "Supply Chain Risk", "Electrification Readiness", "Net Zero Carbon Tracker",
    "Maintenance Optimiser",
])
st.sidebar.divider()
if api_alive():
    st.sidebar.success("Backend connected (:8000)")
else:
    st.sidebar.error("Backend not reachable")
    st.sidebar.code("cd backend\nuvicorn app.main:app --reload --port 8000", language="bash")

# ---------------------------------------------------------------- Fleet Overview
if section == "Fleet Command Center":
    st.title("Fleet Command Center")
    st.caption("Cross-agent correlation — where battery health, supplier risk, and quality signals "
               "line up on the *same* truck. Individually, none of these might look urgent.")

    if not api_alive():
        st.warning("Start the backend (see sidebar) to load the command center.")
        st.stop()

    fleet = build_fleet_snapshots()
    rows = []
    for asset in fleet:
        battery_result = call_battery_agent(asset["asset_id"], asset["capacity_history"],
                                             asset["current_cycle_number"], asset["rated_capacity_ah"],
                                             asset["pack_kwh"])
        resp = requests.post(f"{API_BASE}/fleet/compound-risk", json={
            "asset_id": asset["asset_id"], "battery_risk_band": battery_result["risk_band"],
        }, timeout=10)
        combined = resp.json()
        combined["depot"] = asset["depot"]
        combined["avoidable_cost_inr"] = (battery_result.get("business_impact") or {}).get("avoidable_cost_inr", 0)
        rows.append(combined)

    df = pd.DataFrame(rows).sort_values("compound_score", ascending=False)

    n_compound = (df.priority == "compound priority").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Compound-priority trucks", n_compound)
    c2.metric("Single-signal priority", (df.priority == "single-signal priority").sum())
    c3.metric("Routine", (df.priority == "routine").sum())

    if n_compound > 0:
        total_exposure = df[df.priority == "compound priority"].avoidable_cost_inr.sum()
        st.warning(f"{n_compound} truck(s) show problems across multiple systems at once — "
                   "these would NOT necessarily surface as top priority from any single agent alone. "
                   f"Estimated avoidable cost by acting on these now, battery risk alone: "
                   f"**₹{total_exposure:,}**.")

    st.divider()
    for _, r in df.iterrows():
        border_color = {"compound priority": "🔴", "single-signal priority": "🟡", "routine": "🟢"}[r.priority]
        with st.container(border=True):
            cols = st.columns([2, 2, 2, 2, 2])
            cols[0].markdown(f"{border_color} **{r.asset_id}**  \n{r.depot}")
            cols[1].markdown(f"Battery: {risk_pill(r.battery_risk_band)}", unsafe_allow_html=True)
            cols[2].markdown(f"Supplier ({r.cell_supplier_id}): {risk_pill(r.supplier_risk_level)}"
                              f"  \n{r.supplier_flagged_pct}% shipments flagged", unsafe_allow_html=True)
            q_band = "critical" if r.quality_verdict == "defective" else "healthy"
            cols[3].markdown(f"Last quality check: {risk_pill(q_band)}", unsafe_allow_html=True)
            cols[4].markdown(f"**{r.priority.upper()}**  \nsignals active: {r.active_signal_count}/3")

    st.caption("Each truck's supplier and quality linkage is an illustrative mapping standing in for a "
               "real bill-of-materials system — the battery, supplier-risk, and quality models feeding "
               "it are all real, independently trained models.")

elif section == "Fleet Overview":
    st.title("Fleet Overview")
    st.caption("Battery health across the fleet, ranked by urgency.")

    if not api_alive():
        st.warning("Start the backend (see sidebar) to load live predictions.")
        st.stop()

    fleet = build_fleet_snapshots()
    rows = []
    for asset in fleet:
        result = call_battery_agent(asset["asset_id"], asset["capacity_history"], asset["current_cycle_number"], asset["rated_capacity_ah"], asset["pack_kwh"])
        rows.append({**asset, **result})

    df = pd.DataFrame(rows).sort_values("predicted_rul_cycles")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fleet size", len(df))
    c1.metric("Critical assets", int((df.risk_band == "critical").sum()))
    c2.metric("Watch-list assets", int((df.risk_band == "watch").sum()))
    c2.metric("Healthy assets", int((df.risk_band == "healthy").sum()))
    c3.metric("Avg. state of health", f"{df.state_of_health_pct.mean():.1f}%")
    c3.metric("Min. RUL in fleet", f"{df.predicted_rul_cycles.min():.0f} cycles")

    st.divider()
    st.subheader("Remaining Useful Life, by asset")
    st.bar_chart(df.set_index("asset_id")["predicted_rul_cycles"])

    st.subheader("Fleet detail")
    for _, r in df.iterrows():
        with st.container(border=True):
            cols = st.columns([2, 2, 2, 2, 2])
            cols[0].markdown(f"**{r.asset_id}**  \n{r.depot}")
            cols[1].markdown(f"SOH: **{r.state_of_health_pct}%**")
            cols[2].markdown(f"RUL: **{r.predicted_rul_cycles:.0f} cycles**")
            cols[3].markdown(risk_pill(r.risk_band), unsafe_allow_html=True)
            cols[4].markdown(f"Capacity: {r.current_capacity_ah} Ah")

# ---------------------------------------------------------------- Battery Deep-Dive
elif section == "Battery Deep-Dive":
    st.title("Battery Deep-Dive")

    if not api_alive():
        st.warning("Start the backend (see sidebar) to load live predictions.")
        st.stop()

    fleet = build_fleet_snapshots()
    asset_ids = [a["asset_id"] for a in fleet]
    chosen = st.selectbox("Select asset", asset_ids)
    asset = next(a for a in fleet if a["asset_id"] == chosen)
    result = call_battery_agent(asset["asset_id"], asset["capacity_history"], asset["current_cycle_number"], asset["rated_capacity_ah"], asset["pack_kwh"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("State of health", f"{result['state_of_health_pct']}%")
    c2.metric("Predicted RUL", f"{result['predicted_rul_cycles']:.0f} cycles")
    c3.metric("Current capacity", f"{result['current_capacity_ah']} Ah")
    c4.markdown("Risk band")
    c4.markdown(risk_pill(result["risk_band"]), unsafe_allow_html=True)

    if result.get("business_impact"):
        bi = result["business_impact"]
        st.subheader("Business impact")
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Planned replacement cost", f"₹{bi['planned_replacement_cost_inr']:,}")
        bc2.metric("Cost if it fails unplanned", f"₹{bi['unplanned_failure_cost_inr']:,}")
        bc3.metric("Avoidable by acting early", f"₹{bi['avoidable_cost_inr']:,}")
        st.info(bi["urgency"])
        st.caption(bi["assumption"])

    st.subheader("Recent capacity trend")
    st.line_chart(pd.Series(asset["capacity_history"], name="Capacity (Ah)"))
    st.caption("Real NASA PCoE cycling data, most recent 10 discharge cycles shown.")

# ---------------------------------------------------------------- Quality Inspection
elif section == "Quality Inspection":
    st.title("Manufacturing Quality Inspection")
    st.caption("Two independent defect classifiers — cast EV components (defective/OK) "
               "and sheet-metal / structural steel surfaces (which of 6 known defect types).")

    if not api_alive():
        st.warning("Start the backend (see sidebar) to run inspections.")
        st.stop()

    tab_cast, tab_surface = st.tabs(["Cast Component", "Steel Surface (NEU-DET)"])

    with tab_cast:
        st.caption("Upload a cast-component image for a defect verdict, with Grad-CAM explainability.")
        uploaded = st.file_uploader("Component image", type=["jpg", "jpeg", "png"], key="cast_uploader")
        if uploaded:
            col1, col2 = st.columns(2)
            col1.image(uploaded, caption="Uploaded image", use_container_width=True)

            with st.spinner("Inspecting..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                resp = requests.post(f"{API_BASE}/quality/inspect", files=files, timeout=30)

            if resp.status_code == 200:
                result = resp.json()
                gradcam_bytes = base64.b64decode(result["gradcam_overlay_png_base64"])
                col2.image(gradcam_bytes, caption="Grad-CAM — where the model looked", use_container_width=True)

                verdict = result["verdict"]
                color = "#C53030" if verdict == "defective" else "#00875A"
                st.markdown(
                    f'<h3 style="color:{color}">Verdict: {verdict.upper()} '
                    f'({result["confidence"]*100:.1f}% confidence)</h3>',
                    unsafe_allow_html=True,
                )
                if verdict == "defective":
                    st.warning(
                        "**Business impact — 1-10-100 rule of quality cost:** a defect caught here, "
                        "at incoming inspection, costs roughly **10x less** to address than the same "
                        "defect caught after assembly, and roughly **100x less** than a field failure "
                        "or recall. Reject this batch before it reaches the line."
                    )
            else:
                st.error(f"Inspection failed: {resp.text}")

    with tab_surface:
        st.caption("Upload a steel-surface image to classify which of 6 known defect types is present "
                    "(NEU-DET) — crazing, inclusion, patches, pitted surface, rolled-in scale, or scratches. "
                    "Every image here is assumed to already contain a defect; this classifies the type, "
                    "it isn't a defective/OK gate like the cast-component tab.")
        uploaded_s = st.file_uploader("Surface image", type=["jpg", "jpeg", "png"], key="surface_uploader")
        if uploaded_s:
            col1, col2 = st.columns(2)
            col1.image(uploaded_s, caption="Uploaded image", use_container_width=True)

            with st.spinner("Inspecting..."):
                files = {"file": (uploaded_s.name, uploaded_s.getvalue(), uploaded_s.type)}
                resp = requests.post(f"{API_BASE}/quality/inspect-surface", files=files, timeout=30)

            if resp.status_code == 200:
                result = resp.json()
                gradcam_bytes = base64.b64decode(result["gradcam_overlay_png_base64"])
                col2.image(gradcam_bytes, caption="Grad-CAM — where the model looked", use_container_width=True)

                verdict = result["verdict"].replace("_", " ").replace("-", " ")
                st.markdown(
                    f'<h3 style="color:#B45309">Defect type: {verdict.upper()} '
                    f'({result["confidence"]*100:.1f}% confidence)</h3>',
                    unsafe_allow_html=True,
                )
                st.caption("Confidence across all 6 classes:")
                probs_df = pd.DataFrame(
                    {"probability": result["class_probabilities"]}
                ).sort_values("probability", ascending=False)
                st.bar_chart(probs_df)
            else:
                st.error(f"Inspection failed: {resp.text}")

# ---------------------------------------------------------------- Supply Chain Risk
elif section == "Electrification Readiness":
    st.title("Fleet Electrification Readiness")
    st.caption("Rule-based expert baseline — every score traces back to real km/kg/hour constraints "
               "against 3 real Indian commercial EV models, not a black box.")

    readiness_csv = DATA_DIR / "processed" / "fleet_readiness.csv"
    if not readiness_csv.exists():
        st.warning("Run `python train/train_fleet_readiness.py` first to generate the ICE fleet baseline.")
        st.stop()

    df = pd.read_csv(readiness_csv)
    ready_pct = (df.readiness == "ready").mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("ICE vehicles assessed", len(df))
    c2.metric("Ready for electrification today", f"{ready_pct:.0f}%")
    c3.metric("Not yet viable", f"{100-ready_pct:.0f}%")

    st.subheader("Readiness by route type")
    by_route = df.groupby("route_type").readiness.value_counts(normalize=True).unstack().fillna(0) * 100
    st.bar_chart(by_route)
    st.caption("Highway feeder routes lag because no current EV in the catalog covers that daily range — "
               "a real, honest limit of today's commercial EV options, not a modelling gap.")

    tab1, tab2 = st.tabs(["Fleet detail", "Score a vehicle"])
    with tab1:
        for _, r in df.sort_values("confidence_pct", ascending=False).head(12).iterrows():
            with st.container(border=True):
                cols = st.columns([2, 2, 2, 3])
                cols[0].markdown(f"**{r.vehicle_id}**  \n{r.route_type}")
                cols[1].markdown(f"{r.daily_distance_km:.0f} km/day, {r.avg_payload_kg:.0f} kg")
                band = "healthy" if r.readiness == "ready" else "critical"
                cols[2].markdown(risk_pill(band if r.readiness == "ready" else "watch"), unsafe_allow_html=True)
                if r.readiness == "ready":
                    cols[3].markdown(f"→ **{r.recommended_model}** (₹{r.price_inr_lakh}L, {r.confidence_pct:.0f}% confidence)")
                else:
                    cols[3].markdown(f"_{r.reason}_")

    with tab2:
        if not api_alive():
            st.warning("Start the backend (see sidebar) to score a vehicle.")
        else:
            dist = st.slider("Daily distance (km)", 10, 250, 50)
            payload = st.slider("Average payload (kg)", 50, 1000, 350)
            dwell = st.slider("Depot dwell time (hours)", 0.0, 16.0, 10.0)
            if st.button("Score this vehicle", type="primary"):
                resp = requests.post(f"{API_BASE}/fleet-readiness/score", json={
                    "vehicle_id": "MANUAL-TEST", "daily_distance_km": dist,
                    "avg_payload_kg": payload, "dwell_time_hours": dwell,
                }, timeout=10)
                if resp.status_code == 200:
                    r = resp.json()
                    if r["readiness"] == "ready":
                        st.success(f"Ready → **{r['recommended_model']}** "
                                   f"(₹{r['price_inr_lakh']}L, {r['confidence_pct']:.0f}% confidence)")
                    else:
                        st.error(f"Not yet viable — {r['reason']}")
                else:
                    st.error(resp.text)

elif section == "Net Zero Carbon Tracker":
    st.title("Net Zero Progress & Carbon Intelligence")
    st.caption("Grid emission factor: Central Electricity Authority official database (real, cited). "
               "Diesel comparison: real Tata Ace Diesel mileage where available.")

    if not api_alive():
        st.warning("Start the backend (see sidebar) to load the carbon tracker.")
        st.stop()

    meta = requests.get(f"{API_BASE}/carbon/metadata", timeout=10).json()

    st.subheader("India's grid is getting cleaner — real CEA data, not a projection")
    trend_df = pd.Series(meta["grid_emission_factor_trend"], name="tCO2/MWh")
    st.line_chart(trend_df)
    st.caption("Weighted average grid emission factor, FY2020-21 to FY2024-25 (incl. renewables, "
               "captive generation, cross-border transfers). Source: CEA CO2 Baseline Database v21.0. "
               "Every EV on this platform gets cleaner automatically as this number drops — no fleet "
               "action required.")

    st.divider()
    st.subheader("EV vs. diesel, per catalog vehicle")
    cols = st.columns(3)
    for i, model in enumerate(meta["vehicle_models"]):
        resp = requests.post(f"{API_BASE}/carbon/compute-savings", json={
            "vehicle_model": model, "daily_distance_km": 60,
        }, timeout=10)
        r = resp.json()
        with cols[i]:
            st.markdown(f"**{model}**")
            st.metric("EV emissions", f"{r['ev_g_co2_per_km']} g CO2/km")
            st.metric("Diesel equivalent", f"{r['diesel_equivalent_g_co2_per_km']} g CO2/km")
            st.metric("Savings", f"{r['savings_pct']}%")
            st.caption(r["diesel_comparison_source"])

    st.info("Honest framing: savings are real but moderate today (20-32%, not 90%+) because "
            "~70% of India's grid is still fossil-fuel generation. As the grid decarbonizes, "
            "every EV on this platform automatically gets cleaner — no fleet action required.")

    st.divider()
    st.subheader("Fleet-wide annual impact calculator")
    n_vehicles = st.slider("Number of vehicles electrified", 1, 200, 20)
    model_choice = st.selectbox("Vehicle model", meta["vehicle_models"])
    daily_km = st.slider("Typical daily distance (km)", 10, 250, 60)
    resp = requests.post(f"{API_BASE}/carbon/compute-savings", json={
        "vehicle_model": model_choice, "daily_distance_km": daily_km,
    }, timeout=10)
    r = resp.json()
    annual_fleet_tonnes = round(r["annual_co2_savings_kg"] * n_vehicles / 1000, 1)
    st.metric(f"Estimated annual CO2 savings, {n_vehicles} vehicles", f"{annual_fleet_tonnes} tonnes")
    st.caption("Assumes 300 operating days/year per vehicle.")

elif section == "Maintenance Optimiser":
    st.title("Maintenance Operations Optimiser")
    st.caption("Scheduling optimization — not prediction. Combines maintenance urgency (from the "
               "Battery/Compound-Risk agents), workshop capacity, and charging-infrastructure uptime "
               "into an actual optimized plan, compared honestly against a standard scheduling heuristic.")

    if not api_alive():
        st.warning("Start the backend (see sidebar) to build a schedule.")
        st.stop()

    st.subheader("Workshop configuration")
    c1, c2, c3, c4 = st.columns(4)
    n_bays = c1.number_input("Service bays", 1, 20, 3)
    technicians = c2.number_input("Technicians", 1, 30, 4)
    hours_per_shift = c3.number_input("Hours/shift", 4.0, 12.0, 8.0)
    shifts_per_day = c4.number_input("Shifts/day", 1, 3, 2)

    st.caption("Capacity benchmarks: 0.7 technicians/bay, 1.5-2 bays/technician for heavy-commercial "
               "vehicles (PartsTech 2025 survey, Heavy Duty Journal). Charging uptime assumed at 80% — "
               "independently-measured functional uptime (UC Berkeley/SLAC, ChargerHelp 2025), not the "
               "higher self-reported figures charging networks often publish.")

    # pull real priority signals from the fleet already built elsewhere on this platform
    fleet = build_fleet_snapshots()
    jobs_payload = []
    for asset in fleet:
        battery_result = call_battery_agent(asset["asset_id"], asset["capacity_history"],
                                             asset["current_cycle_number"], asset["rated_capacity_ah"])
        cr_resp = requests.post(f"{API_BASE}/fleet/compound-risk", json={
            "asset_id": asset["asset_id"], "battery_risk_band": battery_result["risk_band"],
        }, timeout=10).json()
        priority = ({"compound priority": "critical", "single-signal priority": "watch",
                     "routine": "routine"}).get(cr_resp["priority"], "routine")
        jobs_payload.append({"asset_id": asset["asset_id"], "priority": priority,
                              "reason": cr_resp["priority"]})

    st.subheader(f"Maintenance queue — {len(jobs_payload)} vehicles, pulled live from Fleet Command Center")
    st.dataframe(pd.DataFrame(jobs_payload), use_container_width=True, hide_index=True)

    stress_test = st.checkbox(
        "Stress-test with a larger simulated backlog (for demo purposes)",
        help="8 real trucks often isn't enough volume for the two scheduling methods to diverge "
             "(see note below once you build a schedule). This adds simulated jobs on top of the "
             "real 8 so the optimizer's advantage is reliably visible — clearly a what-if, not a "
             "claim about the real fleet.")
    if stress_test:
        import random as _random
        _rng = _random.Random(7)
        extra = [{"asset_id": f"SIM-{i+1:03d}",
                  "priority": _rng.choices(["critical", "watch", "routine"], weights=[0.3, 0.3, 0.4])[0],
                  "reason": "simulated for stress test"} for i in range(25)]
        jobs_payload = jobs_payload + extra
        st.caption(f"Testing with {len(jobs_payload)} jobs (8 real + 25 simulated).")

    if st.button("Build optimized schedule", type="primary"):
        resp = requests.post(f"{API_BASE}/maintenance/schedule", json={
            "jobs": jobs_payload, "n_bays": n_bays, "technicians": technicians,
            "hours_per_shift": hours_per_shift, "shifts_per_day": shifts_per_day,
        }, timeout=30)
        result = resp.json()

        st.divider()
        st.metric("Daily capacity", f"{result['daily_capacity_jobs']} jobs/day")
        gc1, gc2 = st.columns(2)
        gc1.metric("Greedy (EDF) — deadlines missed", result["greedy"]["jobs_missing_deadline"])
        gc2.metric("LP-Optimal — deadlines missed", result["optimal"]["jobs_missing_deadline"],
                   delta=-(result["greedy"]["jobs_missing_deadline"] - result["optimal"]["jobs_missing_deadline"]),
                   delta_color="inverse")
        g_crit = sum(1 for a in result["greedy"]["assignments"] if a["missed_deadline"] and a["priority"] == "critical")
        o_crit = sum(1 for a in result["optimal"]["assignments"] if a["missed_deadline"] and a["priority"] == "critical")
        if result["improvement"] > 0:
            st.success(f"Optimization beats the standard scheduling heuristic by "
                       f"{result['improvement']} fewer missed deadlines on this queue — "
                       f"and never at the cost of a critical job (both methods miss {o_crit} critical here).")
        else:
            st.info(f"Both methods tied on this queue ({result['optimal']['jobs_missing_deadline']} missed each, "
                    f"{o_crit} critical). With only 8 real trucks there often isn't enough job volume for the "
                    f"two methods to diverge (real divergence shows up in about 1 of 4 realistic queues at this "
                    f"size). To see it reliably: tick the stress-test box above, set bays/technicians low "
                    f"(e.g. 1 and 1), and rebuild. What's guaranteed either way, tied or not: the optimizer "
                    f"never misses MORE critical jobs than greedy — it only ever matches or improves on the "
                    f"highest-priority tier, by design.")

        st.subheader("Optimized schedule (by day)")
        opt_df = pd.DataFrame(result["optimal"]["assignments"]).sort_values("scheduled_day")
        st.dataframe(opt_df, use_container_width=True, hide_index=True)
        st.caption(result["assumptions"])

elif section == "Supply Chain Risk":
    st.title("Supply Chain Risk & Traceability")
    st.caption("Isolation Forest anomaly detection over supplier shipment records.")

    risk_csv = DATA_DIR / "processed" / "supply_chain_shipments.csv"
    if not risk_csv.exists():
        st.warning("Run `python train/train_risk_model.py` first to generate shipment data.")
        st.stop()

    df = pd.read_csv(risk_csv)
    flagged = df[df.flagged == 1].sort_values("risk_score", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Shipments monitored", len(df))
    c2.metric("Flagged as anomalous", len(flagged))
    c3.metric("Materials tracked", df.material.nunique())

    tab1, tab2 = st.tabs(["Flagged shipments", "Score a new shipment"])

    with tab1:
        st.subheader("Highest-risk shipments right now")
        for _, r in flagged.head(10).iterrows():
            with st.container(border=True):
                cols = st.columns([2, 2, 2, 3])
                cols[0].markdown(f"**{r.shipment_id}**  \n{r.material}")
                cols[1].markdown(f"Supplier: {r.supplier_id}  \n{r.region}")
                cols[2].markdown(risk_pill("critical" if r.risk_score > 0.6 else "watch"), unsafe_allow_html=True)
                cols[2].markdown(f"Score: **{r.risk_score:.2f}**")
                cols[3].markdown(f"_{r.anomaly_type.replace('_', ' ')}_")

        st.divider()
        st.subheader("Supplier concentration risk by material")
        conc = df.groupby("material")["supplier_concentration_pct"].max().sort_values()
        st.bar_chart(conc)
        st.caption("Materials where one supplier holds a large share of total volume — "
                   "a single disruption there hits the whole material category.")

    with tab2:
        st.subheader("Score a hypothetical shipment")
        if not api_alive():
            st.warning("Start the backend (see sidebar) to score shipments.")
        else:
            col1, col2 = st.columns(2)
            lead_dev = col1.slider("Lead time deviation (%)", -20, 250, 5)
            price_dev = col1.slider("Price deviation (%)", -20, 100, 2)
            reject_rate = col1.slider("Reject rate (%)", 0.0, 35.0, 1.8)
            reject_trend = col1.slider(
                "Reject rate trend (change per shipment, last 3)", -2.0, 3.0, 0.0,
                help="0 = flat/stable at the level set above. Positive = climbing — this is what "
                     "triggers an early WATCH flag even when the absolute reject rate isn't extreme yet.")
            audit_days = col1.slider("Days since last audit", 0, 365, 40)
            single_sourced = col2.checkbox("Single-sourced (no backup supplier)")
            geo_risk = col2.slider("Geopolitical risk (0-1)", 0.0, 1.0, 0.2)
            concentration = col2.slider("Supplier concentration (%)", 0, 100, 25)
            volume_dev = col2.slider("Order volume deviation (%)", -50, 300, 5)

            if st.button("Score this shipment", type="primary"):
                payload = {
                    "shipment_id": "MANUAL-TEST-001",
                    "lead_time_deviation_pct": lead_dev, "price_deviation_pct": price_dev,
                    "reject_rate_pct": reject_rate, "days_since_last_audit": audit_days,
                    "single_sourced": int(single_sourced), "geopolitical_risk": geo_risk,
                    "supplier_concentration_pct": concentration, "volume_deviation_pct": volume_dev,
                    "reject_rate_trend": reject_trend, "price_trend": 0.0,
                }
                resp = requests.post(f"{API_BASE}/risk/score-shipment", json=payload, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()
                    st.markdown(risk_pill(result["risk_band"]), unsafe_allow_html=True)
                    st.metric("Risk score", f"{result['risk_score']:.2f}")
                    st.write(result["reason"])
                else:
                    st.error(resp.text)
