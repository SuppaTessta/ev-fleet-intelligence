"""Demo-fleet construction and the cross-agent calls several pages share.

The Command Center, Fleet Overview, Battery Deep-Dive and Maintenance pages all
need the same battery and compound-risk results. Fetched once here and cached,
rather than each page issuing its own round trips on every rerun.
"""

from __future__ import annotations

from pathlib import Path

import api_client
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BATTERY_FEATURES = DATA_DIR / "processed" / "battery_features.csv"

# Each "truck" is a window of one real NASA cell's cycling data, cut off at a
# different point in life so the fleet spans healthy -> critical.
#
# Duplicated in backend/app/agents/telemetry_simulator.py because the dashboard
# talks to the backend over HTTP and does not import its internals. Sharing it
# properly needs an endpoint that serves the demo fleet; until then
# tests/test_no_drift.py fails the build if the two definitions diverge.
FLEET_CUTOFFS = [
    ("B0005", 30, "EV-TRUCK-01", "Chakan Depot"),
    ("B0005", 140, "EV-TRUCK-02", "Chakan Depot"),
    ("B0006", 50, "EV-TRUCK-03", "Pune Ring Road"),
    ("B0006", 120, "EV-TRUCK-04", "Pune Ring Road"),
    ("B0006", 165, "EV-TRUCK-05", "Pune Ring Road"),
    ("B0007", 40, "EV-TRUCK-06", "Hinjewadi Route"),
    ("B0007", 100, "EV-TRUCK-07", "Hinjewadi Route"),
    ("B0007", 160, "EV-TRUCK-08", "Hinjewadi Route"),
]

DEMO_PACK_KWH = 21.3  # illustrative -- the Tata Ace EV class used in Fleet Readiness


@st.cache_data(show_spinner=False)
def build_fleet_snapshots() -> list[dict]:
    df = pd.read_csv(BATTERY_FEATURES)
    fleet = []
    for battery, cutoff, asset_id, depot in FLEET_CUTOFFS:
        history = df[df.battery_id == battery].sort_values("discharge_cycle_num")
        # the cell's true early-life peak, not the max of the sent window --
        # SOH computed off a window that misses the peak understates degradation
        rated_capacity = history["capacity"].iloc[:5].max()
        window = history[history.discharge_cycle_num <= cutoff].tail(10)
        latest = window.iloc[-1]
        fleet.append({
            "asset_id": asset_id, "depot": depot,
            "capacity_history": window["capacity"].round(3).tolist(),
            "current_cycle_number": int(cutoff),
            "rated_capacity_ah": round(float(rated_capacity), 3),
            "pack_kwh": DEMO_PACK_KWH,
            # The REAL operating conditions from the same CSV row, not the
            # request schema's defaults. See predict_battery.
            "temp_battery_c": float(latest["temp_battery"]),
            "ambient_temp_c": float(latest["ambient_temp"]),
            "discharge_current_a": float(latest["discharge_current"]),
        })
    return fleet


def predict_battery(asset: dict) -> dict:
    """Score one demo truck.

    temp_battery_c / ambient_temp_c / discharge_current_a are sent explicitly
    because they are real model features. Omitting them hands the model the
    request schema's defaults -- 25.0 degC against a recorded 32.1-41.1 degC
    across the three demo cells -- which scores the fleet out of distribution
    and makes this page disagree with the Live Fleet Monitor about the same
    truck at the same cycle.
    """
    payload = {
        "asset_id": asset["asset_id"],
        "capacity_history_ah": asset["capacity_history"],
        "current_cycle_number": asset["current_cycle_number"],
        "rated_capacity_ah": asset["rated_capacity_ah"],
        "pack_kwh": asset["pack_kwh"],
        "temp_battery_c": asset["temp_battery_c"],
        "ambient_temp_c": asset["ambient_temp_c"],
        "discharge_current_a": asset["discharge_current_a"],
    }
    return api_client.post("/battery/predict-rul", payload)


@st.cache_data(ttl=60, show_spinner="Scoring the fleet…")
def fleet_with_battery_health() -> list[dict]:
    """Battery prediction for every demo truck. One cached pass."""
    return [{**asset, **predict_battery(asset)} for asset in build_fleet_snapshots()]


@st.cache_data(ttl=60, show_spinner="Correlating signals across agents…")
def fleet_with_compound_risk() -> list[dict]:
    """Battery + compound risk for every truck, cached as one unit.

    Still N round trips, but once per minute instead of on every widget
    interaction. Collapsing it into a single batch endpoint is the right fix and
    is a tracked follow-up; caching removes the user-visible cost now.
    """
    rows = []
    for asset in fleet_with_battery_health():
        combined = api_client.post("/fleet/compound-risk", {
            "asset_id": asset["asset_id"],
            "battery_risk_band": asset["risk_band"],
        })
        combined["depot"] = asset["depot"]
        combined["predicted_rul_cycles"] = asset["predicted_rul_cycles"]
        combined["state_of_health_pct"] = asset["state_of_health_pct"]
        combined["avoidable_cost_inr"] = (
            (asset.get("business_impact") or {}).get("avoidable_cost_inr", 0))
        rows.append(combined)
    return rows
