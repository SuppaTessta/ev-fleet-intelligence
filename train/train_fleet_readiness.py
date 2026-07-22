"""
Fleet Electrification Readiness & Procurement Intelligence Agent.

Deliberately rule-based rather than a trained ML model — the problem
statement asks for scoring "versus an expert baseline," which means the
expert baseline has to exist and be inspectable, not be a black box we
also claim is the expert. Every score here can be traced back to which
constraint it passed or failed, in real units (km, kg, hours) — that
traceability is the point for a procurement decision this consequential.

Catalog is 3 real Indian commercial EVs (Mahindra Treo Zor, Euler HiLoad EV,
Tata Ace EV) with real specs, not invented ones.
"""

from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

EV_CATALOG = [
    {"model": "Mahindra Treo Zor", "category": "Light 3-wheeler cargo",
     "range_km": 80, "payload_kg": 550, "price_inr_lakh": 3.3, "charge_time_hours": 5.0},
    {"model": "Euler HiLoad EV", "category": "Medium 3-wheeler cargo",
     "range_km": 135, "payload_kg": 688, "price_inr_lakh": 4.2, "charge_time_hours": 5.0},
    {"model": "Tata Ace EV", "category": "Mini-truck (4-wheeler)",
     "range_km": 154, "payload_kg": 600, "price_inr_lakh": 10.51, "charge_time_hours": 1.75},
]
RANGE_SAFETY_MARGIN = 1.25  # required range = daily distance x this, to cover traffic/weather/battery aging

ROUTE_TYPES = ["Urban short-haul", "Urban long-haul", "Mixed depot-to-depot", "Highway feeder"]


def score_vehicle(daily_distance_km: float, avg_payload_kg: float, dwell_time_hours: float) -> dict:
    """The expert baseline: transparent, inspectable, unit-based gates."""
    required_range = daily_distance_km * RANGE_SAFETY_MARGIN
    candidates = []

    for ev in EV_CATALOG:
        range_ok = ev["range_km"] >= required_range
        payload_ok = ev["payload_kg"] >= avg_payload_kg
        charge_ok = dwell_time_hours >= ev["charge_time_hours"]
        feasible = range_ok and payload_ok and charge_ok

        # margin-based confidence: how comfortably it clears each gate, not just pass/fail
        range_margin = min(1.5, ev["range_km"] / required_range) / 1.5 if required_range > 0 else 1.0
        payload_margin = min(1.5, ev["payload_kg"] / max(avg_payload_kg, 1)) / 1.5
        charge_margin = min(1.5, dwell_time_hours / ev["charge_time_hours"]) / 1.5 if ev["charge_time_hours"] > 0 else 1.0
        confidence = round(float(100 * np.mean([range_margin, payload_margin, charge_margin])), 1) if feasible else 0.0

        candidates.append({**ev, "feasible": feasible, "confidence_pct": confidence,
                            "range_ok": range_ok, "payload_ok": payload_ok, "charge_ok": charge_ok})

    feasible_options = [c for c in candidates if c["feasible"]]
    if feasible_options:
        best = min(feasible_options, key=lambda c: c["price_inr_lakh"])  # cheapest that actually fits
        return {
            "readiness": "ready", "recommended_model": best["model"],
            "confidence_pct": best["confidence_pct"], "price_inr_lakh": best["price_inr_lakh"],
            "reason": f"Fits within {best['model']}'s {best['range_km']}km range, "
                      f"{best['payload_kg']}kg payload, and {best['charge_time_hours']}h charge window.",
            "all_options": candidates,
        }
    else:
        blockers = []
        best_attempt = max(candidates, key=lambda c: c["confidence_pct"] if c["feasible"] else
                            np.mean([c["range_ok"], c["payload_ok"], c["charge_ok"]]))
        if not best_attempt["range_ok"]:
            blockers.append(f"no catalog option covers the required {required_range:.0f}km range")
        if not best_attempt["payload_ok"]:
            blockers.append(f"payload of {avg_payload_kg:.0f}kg exceeds all current EV options")
        if not best_attempt["charge_ok"]:
            blockers.append(f"only {dwell_time_hours:.1f}h dwell time isn't enough to recharge")
        return {
            "readiness": "not_yet_viable", "recommended_model": None, "confidence_pct": 0.0,
            "price_inr_lakh": None, "reason": "; ".join(blockers), "all_options": candidates,
        }


def generate_fleet(n=60) -> pd.DataFrame:
    rows = []
    for i in range(n):
        route = RNG.choice(ROUTE_TYPES, p=[0.35, 0.25, 0.25, 0.15])
        if route == "Urban short-haul":
            dist, payload, dwell = RNG.uniform(20, 60), RNG.uniform(150, 500), RNG.uniform(8, 14)
        elif route == "Urban long-haul":
            dist, payload, dwell = RNG.uniform(60, 110), RNG.uniform(300, 650), RNG.uniform(4, 9)
        elif route == "Mixed depot-to-depot":
            dist, payload, dwell = RNG.uniform(50, 100), RNG.uniform(400, 700), RNG.uniform(6, 12)
        else:  # Highway feeder
            dist, payload, dwell = RNG.uniform(100, 220), RNG.uniform(500, 900), RNG.uniform(2, 6)

        result = score_vehicle(dist, payload, dwell)
        rows.append({
            "vehicle_id": f"ICE-{i+1:03d}", "route_type": route,
            "daily_distance_km": round(dist, 1), "avg_payload_kg": round(payload, 0),
            "dwell_time_hours": round(dwell, 1), "readiness": result["readiness"],
            "recommended_model": result["recommended_model"],
            "confidence_pct": result["confidence_pct"], "price_inr_lakh": result["price_inr_lakh"],
            "reason": result["reason"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scoring against {len(EV_CATALOG)} real EV models: {[e['model'] for e in EV_CATALOG]}\n")
    fleet = generate_fleet(60)
    fleet.to_csv(PROC_DIR / "fleet_readiness.csv", index=False)

    ready_pct = (fleet.readiness == "ready").mean() * 100
    print(f"{len(fleet)} vehicles scored. {ready_pct:.0f}% ready for electrification today.\n")
    print("By route type:")
    print(fleet.groupby("route_type").readiness.value_counts(normalize=True).round(2))
    print(f"\nSaved to {PROC_DIR / 'fleet_readiness.csv'}")
