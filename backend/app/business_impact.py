"""Business-impact translation.

The agents speak in RUL cycles, risk scores and confidence percentages. A fleet
manager cannot act on "RUL = 42 cycles"; they can act on "replace this pack
within three weeks or risk a roadside failure costing 1.5x the planned swap".

Every function returns its assumptions alongside the number, so a reader can
disagree with the inputs rather than having to trust the output.

Sourcing:
- Battery replacement Rs 15,000-25,000/kWh for Indian commercial EVs in 2026,
  consistent across OEM service data and industry press. Midpoint used.
- Local delivery trucking Rs 10-25/km for mini/light commercial trucks on local
  routes, 2025-26 Indian logistics pricing. Midpoint used.
- Unplanned repairs cost more than scheduled ones (emergency labour, expedited
  parts, towing). 1.5x is used rather than the 2-3x some sources cite, to avoid
  overstating the case.
"""

from app.constants import BATTERY_COST_INR_PER_KWH

# re-exported under the old name so call sites and the assumption strings
# below read unchanged; the single definition lives in app.constants
BATTERY_COST_PER_KWH_INR = BATTERY_COST_INR_PER_KWH
UNPLANNED_REPAIR_MULTIPLIER = 1.5  # conservative vs. industry-cited 2-3x
DEFAULT_TRUCK_RATE_PER_KM_INR = 15  # midpoint of Rs 10-25/km, mini/light commercial trucks, local routes

# route_type -> typical daily distance (km), matching Fleet Readiness agent's own assumptions
ROUTE_DAILY_DISTANCE_KM = {
    "Urban short-haul": 40, "Urban long-haul": 85,
    "Mixed depot-to-depot": 75, "Highway feeder": 160,
}


def battery_business_impact(risk_band: str, predicted_rul_cycles: float, pack_kwh: float) -> dict:
    """Translates a battery RUL prediction into replacement cost and urgency framing."""
    replacement_cost = round(pack_kwh * BATTERY_COST_PER_KWH_INR)
    unplanned_cost = round(replacement_cost * UNPLANNED_REPAIR_MULTIPLIER)
    avoidable_cost = unplanned_cost - replacement_cost

    if risk_band == "critical":
        urgency = "Schedule replacement now — waiting risks an unplanned failure mid-route."
    elif risk_band == "watch":
        urgency = f"Plan replacement within the next ~{max(1, int(predicted_rul_cycles / 4))} weeks " \
                  f"(assuming ~4 cycles/week for a single-shift local delivery vehicle)."
    else:
        urgency = "No action needed yet — keep monitoring."

    return {
        "planned_replacement_cost_inr": replacement_cost,
        "unplanned_failure_cost_inr": unplanned_cost,
        "avoidable_cost_inr": avoidable_cost,
        "urgency": urgency,
        "assumption": f"Rs {BATTERY_COST_PER_KWH_INR:,}/kWh (Indian commercial EV 2026 midpoint); "
                      f"unplanned repairs assumed {UNPLANNED_REPAIR_MULTIPLIER}x planned cost.",
    }


def downtime_exposure(route_type: str, days_down: int = 1,
                       rate_per_km: float = DEFAULT_TRUCK_RATE_PER_KM_INR) -> dict:
    """Revenue at risk if a vehicle on this route type is out of service."""
    daily_km = ROUTE_DAILY_DISTANCE_KM.get(route_type, 60)
    daily_revenue = round(daily_km * rate_per_km)
    return {
        "daily_revenue_at_risk_inr": daily_revenue,
        "total_exposure_inr": daily_revenue * days_down,
        "assumption": f"Rs {rate_per_km}/km (local delivery mini-truck midpoint) x "
                      f"{daily_km} km/day typical for {route_type}.",
    }


def quality_business_impact(verdict: str) -> dict:
    """Cost-of-quality framing for a catch/miss at incoming inspection —
    the "1-10-100 rule" is a standard, widely-cited quality-management
    principle (Crosby-derived): a defect costs roughly 10x more to fix at
    the next production stage, 100x more once it reaches the customer."""
    if verdict == "defective":
        return {
            "stage_caught": "incoming inspection",
            "principle": "1-10-100 rule of quality cost: a defect caught at incoming inspection "
                         "costs roughly 10x less to address than the same defect caught after "
                         "assembly, and roughly 100x less than a field failure or recall.",
            "action": "Reject this batch / component before it reaches assembly.",
        }
    return {"stage_caught": None, "principle": None, "action": "No action needed."}
