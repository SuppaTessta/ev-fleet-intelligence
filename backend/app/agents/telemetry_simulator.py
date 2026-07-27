"""
Live BMS telemetry simulator for the Battery APM agent.

Addresses PS3's "IoT / Telematics / BMS Data Integration" suggested
technology, which the batch endpoint alone doesn't cover: /battery/predict-rul
takes a capacity history you already have in hand and returns one prediction.
This module simulates what a live BMS integration would actually feed that
agent continuously.

This does NOT fabricate synthetic sensor curves. Each of the 8 demo trucks
(same asset_id / battery_id / depot mapping as dashboard.build_fleet_snapshots)
starts at its existing fixed cutoff and, on every tick, advances one real NASA
PCoE discharge cycle further into that battery's ACTUAL recorded history --
so "live" readings are genuine historical measurements played forward in
simulated time, not invented values. Once a truck's real recorded history is
exhausted, it holds at the last real reading and reports that fact explicitly
(historical_data_exhausted: true) rather than looping or fabricating cycles
that were never measured -- the same "don't inflate it" standard the rest of
this codebase holds to.

State lives server-side (module-level, one dict per backend process) so it
advances consistently regardless of which dashboard client is polling --
multiple viewers watching the live monitor see the same simulated fleet, the
way a real shared BMS feed would work.
"""
from pathlib import Path
from threading import Lock
import pandas as pd
from app.agents.battery_agent import get_battery_agent

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "battery_features.csv"
ROLL_WINDOW_HISTORY = 10  # matches dashboard.build_fleet_snapshots' window size, for a like-for-like feed

# Identical fleet mapping to dashboard.build_fleet_snapshots -- this simulator
# makes the existing static snapshot dynamic, it doesn't invent a new fleet.
FLEET = [
    ("B0005", 30, "EV-TRUCK-01", "Chakan Depot"),
    ("B0005", 140, "EV-TRUCK-02", "Chakan Depot"),
    ("B0006", 50, "EV-TRUCK-03", "Pune Ring Road"),
    ("B0006", 120, "EV-TRUCK-04", "Pune Ring Road"),
    ("B0006", 165, "EV-TRUCK-05", "Pune Ring Road"),
    ("B0007", 40, "EV-TRUCK-06", "Hinjewadi Route"),
    ("B0007", 100, "EV-TRUCK-07", "Hinjewadi Route"),
    ("B0007", 160, "EV-TRUCK-08", "Hinjewadi Route"),
]

_lock = Lock()
_state = None  # {asset_id: {battery_id, position, depot, rated_capacity_ah, pack_kwh}}
_df = None


def _load_df():
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
    return _df


def _init_state_locked():
    global _state
    df = _load_df()
    _state = {}
    for battery_id, cutoff, asset_id, depot in FLEET:
        rated = df[df.battery_id == battery_id]["capacity"].iloc[:5].max()
        _state[asset_id] = {
            "battery_id": battery_id,
            "position": cutoff,
            "depot": depot,
            "rated_capacity_ah": round(float(rated), 3),
            "pack_kwh": 21.3,  # illustrative -- matches the Tata Ace EV class used in Fleet Readiness
        }


def reset_fleet():
    """Puts every truck back at its original fixed cutoff (fresh demo run)."""
    with _lock:
        _init_state_locked()


def advance_fleet_tick():
    """Advances every truck by one real historical cycle (or holds, if that
    truck's real recorded history is exhausted) and returns fresh live
    readings + a live RUL re-prediction for each -- using the same trained
    BatteryAgent and the same predict() code path as the batch endpoint, not
    a separate/simplified model."""
    with _lock:
        if _state is None:
            _init_state_locked()
        df = _load_df()
        agent = get_battery_agent()
        readings = []
        for asset_id, s in _state.items():
            hist = df[df.battery_id == s["battery_id"]].sort_values("discharge_cycle_num")
            max_cycle = int(hist["discharge_cycle_num"].max())
            exhausted = s["position"] >= max_cycle
            if not exhausted:
                s["position"] += 1

            window = hist[hist.discharge_cycle_num <= s["position"]].tail(ROLL_WINDOW_HISTORY)
            latest = window.iloc[-1]

            result = agent.predict(
                capacity_history=window["capacity"].round(3).tolist(),
                temp_battery=float(latest["temp_battery"]),
                time_s=float(latest["time"]),
                ambient_temp=float(latest["ambient_temp"]),
                discharge_current=float(latest["discharge_current"]),
                current_cycle_number=int(s["position"]),
                rated_capacity_ah=s["rated_capacity_ah"],
                pack_kwh=s["pack_kwh"],
            )
            readings.append({
                "asset_id": asset_id,
                "depot": s["depot"],
                "battery_id": s["battery_id"],
                "cycle_number": int(s["position"]),
                "total_cycles_recorded": max_cycle,
                "historical_data_exhausted": bool(exhausted),
                "timestamp": str(latest["date_time"]),
                "voltage_v": round(float(latest["voltage_battery"]), 3),
                "current_a": round(float(latest["current_battery"]), 4),
                "temp_battery_c": round(float(latest["temp_battery"]), 2),
                **result,
            })
        return readings
