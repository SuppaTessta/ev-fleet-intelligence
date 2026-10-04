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

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The scoring engine lives in the backend, which is the thing that has to work
# in production. This script is dev tooling and imports FROM it -- the arrow used
# to point the other way, with the serving module inserting the project root into
# sys.path so it could reach into train/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.agents.fleet_readiness_agent import (  # noqa: E402
    EV_CATALOG,
    ROUTE_TYPES,
    score_vehicle,
)

RNG = np.random.default_rng(7)


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

        # Score the values that get WRITTEN, not the raw draws. Scoring
        # `dist` while publishing `round(dist, 1)` made the CSV's own
        # confidence_pct unreproducible from its own input columns on 8 of 60
        # rows (ICE-001 82.1 published vs 82.3 recomputed, ICE-047 78.4 vs
        # 78.6). The dashboard shows the inputs and the confidence side by side,
        # so a reader can and does check that arithmetic.
        dist, payload, dwell = round(dist, 1), round(payload, 0), round(dwell, 1)
        result = score_vehicle(dist, payload, dwell)
        rows.append({
            "vehicle_id": f"ICE-{i+1:03d}", "route_type": route,
            "daily_distance_km": dist, "avg_payload_kg": payload,
            "dwell_time_hours": dwell, "readiness": result["readiness"],
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
