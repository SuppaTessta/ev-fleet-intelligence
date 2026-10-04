"""Fleet Electrification Readiness: can an EV do this job, and should you buy it?

Rule-based rather than trained. A procurement decision this consequential has to
be inspectable, and every score here traces to which constraint it passed or
failed, in real units. The catalog is 3 real Indian commercial EVs with
published specs.

The scoring logic lives here rather than in train/ because production owns it;
train/train_fleet_readiness.py imports from this module, not the reverse.
"""

import numpy as np

EV_CATALOG = [
    {"model": "Mahindra Treo Zor", "category": "Light 3-wheeler cargo",
     "range_km": 80, "payload_kg": 550, "price_inr_lakh": 3.3, "charge_time_hours": 5.0},
    {"model": "Euler HiLoad EV", "category": "Medium 3-wheeler cargo",
     "range_km": 135, "payload_kg": 688, "price_inr_lakh": 4.2, "charge_time_hours": 5.0},
    {"model": "Tata Ace EV", "category": "Mini-truck (4-wheeler)",
     "range_km": 154, "payload_kg": 600, "price_inr_lakh": 10.51, "charge_time_hours": 1.75},
]

# required range = daily distance x this, to cover traffic, weather, battery aging
RANGE_SAFETY_MARGIN = 1.25

ROUTE_TYPES = ["Urban short-haul", "Urban long-haul", "Mixed depot-to-depot", "Highway feeder"]


def score_vehicle(daily_distance_km: float, avg_payload_kg: float,
                   dwell_time_hours: float) -> dict:
    """Score one duty cycle against the catalog on physical gates alone.

    Two limitations worth knowing:

    The recommendation optimises PURCHASE PRICE among feasible options, not
    total cost of ownership -- see score_vehicle_with_economics for the second
    gate. `confidence_pct` therefore describes the cheapest feasible option
    rather than the best-fitting one, and is not monotonic in duty-cycle
    difficulty: a harder route can score higher because it disqualifies the
    cheap-but-marginal option.

    charge_time_hours mixes modalities. The Tata Ace's 1.75h is DC fast
    charging; the two 3-wheelers' 5.0h are AC. Treating them as interchangeable
    assumes a DC charger on site.
    """
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
        charge_margin = (min(1.5, dwell_time_hours / ev["charge_time_hours"]) / 1.5
                         if ev["charge_time_hours"] > 0 else 1.0)
        confidence = (round(float(100 * np.mean([range_margin, payload_margin, charge_margin])), 1)
                      if feasible else 0.0)

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

    # Blockers are reported PER VEHICLE, and catalog-wide phrasing is reserved
    # for gates every option fails. Reading them off a single best-effort
    # candidate while wording them as statements about the whole catalog names
    # the wrong constraint to someone making a purchasing decision -- claiming
    # no option has the range when one does, while never mentioning the payload
    # that actually excluded it.
    def _fails(c) -> list[str]:
        out = []
        if not c["range_ok"]:
            out.append(f"{c['range_km']}km range < {required_range:.1f}km needed")
        if not c["payload_ok"]:
            out.append(f"{c['payload_kg']}kg payload < {avg_payload_kg:.1f}kg load")
        if not c["charge_ok"]:
            out.append(f"needs {c['charge_time_hours']}h to charge, dwell is only "
                       f"{dwell_time_hours:.1f}h")
        return out

    universal = []
    if all(not c["range_ok"] for c in candidates):
        universal.append(f"no catalog option covers the required {required_range:.1f}km range")
    if all(not c["payload_ok"] for c in candidates):
        universal.append(f"payload of {avg_payload_kg:.1f}kg exceeds every catalog option")
    if all(not c["charge_ok"] for c in candidates):
        universal.append(f"{dwell_time_hours:.1f}h dwell is too short for any catalog option")

    per_vehicle = "; ".join(f"{c['model']} ({', '.join(_fails(c))})" for c in candidates)
    reason = (f"{'; '.join(universal)}. Per option: {per_vehicle}" if universal
              else f"No option clears every gate. {per_vehicle}")
    return {
        "readiness": "not_yet_viable", "recommended_model": None, "confidence_pct": 0.0,
        "price_inr_lakh": None, "reason": reason, "all_options": candidates,
    }


def score_vehicle_with_economics(daily_distance_km: float, avg_payload_kg: float,
                                  dwell_time_hours: float,
                                  expected_cycles_remaining: float | None = None) -> dict:
    """score_vehicle plus a five-year TCO verdict on the recommendation.

    Feasibility and economics are different questions and the honest answers
    differ. Keeping the gate function free of prices, tariffs and a horizon
    means a disagreement about the diesel price cannot change whether a vehicle
    is reported as capable.

    It also supplies the external criterion readiness scoring never had: the
    synthetic fleet is labelled by calling this very scorer, so "77% ready"
    restates the route mix. "Is the EV cheaper than the diesel it replaces over
    five years?" is a question the project does not get to answer for itself,
    and the two answers diverge -- a Tata Ace EV is feasible on many routes and
    only cheaper than its diesel twin above ~68 km/day.
    """
    from app.agents.tco import compare  # local: keeps the pure gate import-light

    result = score_vehicle(daily_distance_km, avg_payload_kg, dwell_time_hours)
    if result["recommended_model"] is None:
        result["economics"] = None
        result["verdict"] = "not_yet_viable"
        return result

    economics = compare(result["recommended_model"], daily_distance_km,
                        result["price_inr_lakh"], expected_cycles_remaining)
    result["economics"] = economics

    # Two independent gates, reported separately rather than collapsed into one
    # score, because "capable but not yet worth it" is a real and common answer.
    result["verdict"] = "ready_and_economic" if economics["ev_cheaper"] else "ready_but_uneconomic"
    return result
