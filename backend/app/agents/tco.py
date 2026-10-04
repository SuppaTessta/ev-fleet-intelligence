"""Five-year total cost of ownership, EV vs diesel.

The readiness agent recommends the cheapest feasible EV by sticker price, in a
problem space whose premise is TCO parity. A Tata Ace EV costs 3.2x a Treo Zor
to buy and can be cheaper to run; purchase price alone cannot express that.

This also gives readiness scoring something external to be measured against.
The synthetic fleet is labelled by the same scorer being evaluated, so "77%
ready" restates the route mix. "Is the EV actually cheaper than continuing to
run diesel?" has an answer this project does not get to choose.

Deliberately transparent arithmetic rather than a fitted model: with no real
fleet cost data to fit against, a regression would be false precision. Where a
figure is a defensible mid-range rather than a citation, it says so, and
compare() returns break-even points for the two assumptions that dominate the
result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.agents.carbon_agent import CHARGER_EFFICIENCY, VEHICLE_ENERGY_PROFILES
from app.constants import BATTERY_COST_INR_PER_KWH, OPERATING_DAYS_PER_YEAR

# ---------------------------------------------------------------- constants

HORIZON_YEARS = 5

# Rs/kWh. Commercial/industrial LT tariffs across Indian states span roughly
# 7-11 Rs/kWh in 2025-26; public DC fast charging runs 18-24. A depot-charged
# fleet is on the former. Mid-range of the commercial band.
ELECTRICITY_TARIFF_INR_PER_KWH = 9.0

# Rs/litre. Indian retail diesel sat in the 87-96 band through 2025-26 depending
# on state VAT. Mid-range.
DIESEL_PRICE_INR_PER_LITRE = 91.0

# Rs/km. EVs have far fewer wearing parts -- no oil changes, no clutch, no
# exhaust aftertreatment, and regenerative braking extends pad life. Operator
# reports cluster around a 40-60% reduction; 50% is used, and the absolute
# diesel figure is the mid-range of published Indian LCV service costs.
DIESEL_MAINTENANCE_INR_PER_KM = 1.60
EV_MAINTENANCE_INR_PER_KM = 0.80

# Residual value at 5 years, as a fraction of purchase price. EV residuals in
# India are genuinely uncertain -- a thin used market and battery-health anxiety
# push them below diesel equivalents. Deliberately pessimistic for the EV, since
# assuming otherwise would flatter the case this model exists to test.
EV_RESIDUAL_FRACTION = 0.30
DIESEL_RESIDUAL_FRACTION = 0.40

LAKH = 100_000

# Diesel counterparts, from the carbon agent's own comparison basis so the two
# agents cannot disagree about the same vehicle.
DIESEL_EQUIVALENTS = {
    "Mahindra Treo Zor": {"price_inr_lakh": 2.6, "kmpl": 28,
                           "note": "industry-typical diesel/CNG 3-wheeler cargo (no direct twin)"},
    "Euler HiLoad EV": {"price_inr_lakh": 3.1, "kmpl": 28,
                         "note": "industry-typical diesel/CNG 3-wheeler cargo (no direct twin)"},
    "Tata Ace EV": {"price_inr_lakh": 6.2, "kmpl": 22,
                     "note": "real Tata Ace Diesel -- same body, same duty cycle"},
}

PACK_KWH = {"Mahindra Treo Zor": 7.37, "Euler HiLoad EV": 12.5, "Tata Ace EV": 21.3}


@dataclass(frozen=True)
class TcoBreakdown:
    purchase: float
    energy: float
    maintenance: float
    battery_replacement: float
    residual_value: float          # negative: recovered at end of life

    @property
    def total(self) -> float:
        return (self.purchase + self.energy + self.maintenance
                + self.battery_replacement + self.residual_value)

    def as_dict(self) -> dict:
        """Components and total, guaranteed to add up as printed.

        The total is summed from the ROUNDED components. Rounding each component
        and the unrounded total independently makes them disagree by a rupee,
        and the dashboard renders the components as a bar chart beside the
        total.
        """
        parts = {
            "purchase_inr": round(self.purchase),
            "energy_inr": round(self.energy),
            "maintenance_inr": round(self.maintenance),
            "battery_replacement_inr": round(self.battery_replacement),
            "residual_value_inr": round(self.residual_value),
        }
        return {**parts, "total_inr": sum(parts.values())}


def _lifetime_km(daily_distance_km: float) -> float:
    return daily_distance_km * OPERATING_DAYS_PER_YEAR * HORIZON_YEARS


def ev_tco(model: str, daily_distance_km: float, price_inr_lakh: float,
           expected_cycles_remaining: float | None = None) -> TcoBreakdown:
    """Five-year EV cost.

    `expected_cycles_remaining` wires in the Battery APM agent: if the pack is
    predicted to reach end of life inside the horizon, a replacement lands in
    the TCO. That is the one place the RUL model changes a rupee figure.
    """
    km = _lifetime_km(daily_distance_km)
    kwh_per_km_battery = VEHICLE_ENERGY_PROFILES[model][0]
    # metered energy, not battery energy -- the fleet pays for what the meter
    # records, including the ~10% lost in charging (see carbon_agent)
    kwh_per_km_metered = kwh_per_km_battery / CHARGER_EFFICIENCY

    purchase = price_inr_lakh * LAKH
    energy = km * kwh_per_km_metered * ELECTRICITY_TARIFF_INR_PER_KWH
    maintenance = km * EV_MAINTENANCE_INR_PER_KM

    battery_replacement = 0.0
    if expected_cycles_remaining is not None:
        # one "cycle" ~ one full-range day for a depot-charged delivery vehicle
        range_km = PACK_KWH[model] / kwh_per_km_battery
        cycles_needed = km / max(range_km, 1e-6)
        if cycles_needed > expected_cycles_remaining:
            battery_replacement = PACK_KWH[model] * BATTERY_COST_INR_PER_KWH

    residual = -purchase * EV_RESIDUAL_FRACTION
    return TcoBreakdown(purchase, energy, maintenance, battery_replacement, residual)


def diesel_tco(model: str, daily_distance_km: float) -> TcoBreakdown:
    """Five-year cost of the diesel vehicle this EV would replace."""
    spec = DIESEL_EQUIVALENTS[model]
    km = _lifetime_km(daily_distance_km)

    purchase = spec["price_inr_lakh"] * LAKH
    energy = (km / spec["kmpl"]) * DIESEL_PRICE_INR_PER_LITRE
    maintenance = km * DIESEL_MAINTENANCE_INR_PER_KM
    residual = -purchase * DIESEL_RESIDUAL_FRACTION
    return TcoBreakdown(purchase, energy, maintenance, 0.0, residual)


def compare(model: str, daily_distance_km: float, price_inr_lakh: float,
            expected_cycles_remaining: float | None = None) -> dict:
    """EV vs. diesel over five years, with the sensitivities that decide it."""
    if model not in DIESEL_EQUIVALENTS:
        raise ValueError(f"No diesel comparison basis for {model!r}")
    if daily_distance_km <= 0:
        # A vehicle that covers no distance has no cost-per-km and no break-even
        # distance -- both are 0/0. The request schema permits 0 (a parked or
        # seasonal vehicle is a real thing to record), so this must be a clean
        # 422 rather than the ZeroDivisionError -> 500 it produced before.
        raise ValueError(
            "daily_distance_km must be greater than 0 to compare running costs; "
            "a vehicle covering no distance has no cost per km.")

    ev = ev_tco(model, daily_distance_km, price_inr_lakh, expected_cycles_remaining)
    diesel = diesel_tco(model, daily_distance_km)
    km = _lifetime_km(daily_distance_km)

    # Every rupee figure below derives from the two totals actually RETURNED, so
    # the response reconciles with itself. The dashboard shows the saving and
    # both totals as adjacent metrics.
    ev_d, diesel_d = ev.as_dict(), diesel.as_dict()
    saving = diesel_d["total_inr"] - ev_d["total_inr"]

    return {
        "vehicle_model": model,
        "horizon_years": HORIZON_YEARS,
        "lifetime_km": round(km),
        "ev": ev_d,
        "diesel": diesel_d,
        "ev_saving_inr": saving,
        "ev_cheaper": bool(saving > 0),
        "cost_per_km_ev": round(ev_d["total_inr"] / km, 2),
        "cost_per_km_diesel": round(diesel_d["total_inr"] / km, 2),
        "breakeven": _breakeven(model, daily_distance_km, price_inr_lakh,
                                 expected_cycles_remaining),
        "assumptions": {
            "electricity_inr_per_kwh": ELECTRICITY_TARIFF_INR_PER_KWH,
            "diesel_inr_per_litre": DIESEL_PRICE_INR_PER_LITRE,
            "operating_days_per_year": OPERATING_DAYS_PER_YEAR,
            "ev_maintenance_inr_per_km": EV_MAINTENANCE_INR_PER_KM,
            "diesel_maintenance_inr_per_km": DIESEL_MAINTENANCE_INR_PER_KM,
            "ev_residual_fraction": EV_RESIDUAL_FRACTION,
            "diesel_residual_fraction": DIESEL_RESIDUAL_FRACTION,
            "charger_efficiency": CHARGER_EFFICIENCY,
            "diesel_basis": DIESEL_EQUIVALENTS[model]["note"],
        },
    }


def _max_feasible_daily_km(model: str) -> float | None:
    """The longest daily route this vehicle passes the readiness range gate on.

    Imported lazily: fleet_readiness_agent imports THIS module (inside a
    function) for the economics, so a module-scope import here would close the
    loop.
    """
    from app.agents.fleet_readiness_agent import EV_CATALOG, RANGE_SAFETY_MARGIN

    spec = next((e for e in EV_CATALOG if e["model"] == model), None)
    return None if spec is None else spec["range_km"] / RANGE_SAFETY_MARGIN


def _breakeven(model: str, daily_distance_km: float, price_inr_lakh: float,
               expected_cycles_remaining: float | None) -> dict:
    """The two numbers that actually decide this, solved rather than guessed.

    A single TCO figure invites false confidence. Daily distance and the
    diesel/electricity price ratio move the answer far more than any modelling
    refinement would, so they are reported as break-even points.
    """
    def saving_at(distance: float) -> float:
        return (diesel_tco(model, distance).total
                - ev_tco(model, distance, price_inr_lakh,
                          expected_cycles_remaining).total)

    # Scan, do not bisect. `saving_at` is not monotonic once
    # expected_cycles_remaining is supplied: the pack replacement fires the
    # moment lifetime_km / range_km exceeds the remaining cycles, so the saving
    # steps sharply DOWN at that distance and there can be three sign changes.
    # A bisection needs a single root, and given two it converges on the upper
    # one -- reporting that the vehicle never pays back in a response that also
    # says it is already saving money.
    #
    # A coarse scan finds every sign change; bisection then refines each
    # bracket. Only needed when a replacement can fire: with
    # expected_cycles_remaining None every term is linear in distance plus a
    # constant, so two grid points bracket the single root exactly and the
    # common path stays ~0.2 ms instead of ~2.5 ms.
    LO, HI = 1.0, 500.0
    STEP = 0.5 if expected_cycles_remaining is not None else (HI - LO)
    grid = [LO + i * STEP for i in range(int(round((HI - LO) / STEP)) + 1)]
    wins = [saving_at(d) > 0 for d in grid]

    def _refine(lo_d: float, hi_d: float) -> float:
        """Smallest 0.1 step in (lo_d, hi_d] at which the EV wins."""
        for _ in range(40):
            mid = (lo_d + hi_d) / 2
            if saving_at(mid) > 0:
                hi_d = mid
            else:
                lo_d = mid
        candidate = math.ceil(hi_d * 10) / 10
        for _ in range(10):     # verify, never assume -- see the step above
            if saving_at(candidate) > 0:
                return candidate
            candidate = round(candidate + 0.1, 1)
        return candidate

    # "above which the EV wins" means it keeps winning. Walk back from the top
    # of the range through the last unbroken run of wins.
    breakeven_km = None
    if wins[-1]:
        i = len(wins) - 1
        while i > 0 and wins[i - 1]:
            i -= 1
        breakeven_km = LO if i == 0 else _refine(grid[i - 1], grid[i])

    # Any bounded stretch below that where the EV also wins is a real answer to
    # a question a reader will ask, and silently dropping it is what made the
    # note contradict the headline. Reported separately rather than folded in.
    also_wins = []
    for i, won in enumerate(wins):
        if won and not wins[i - 1] if i else won:
            start = LO if i == 0 else _refine(grid[i - 1], grid[i])
            end = next((grid[j - 1] for j in range(i + 1, len(wins)) if not wins[j]), None)
            if end is not None:
                also_wins.append([start, round(end, 1)])

    km = _lifetime_km(daily_distance_km)
    ev = ev_tco(model, daily_distance_km, price_inr_lakh, expected_cycles_remaining)
    diesel_no_fuel = diesel_tco(model, daily_distance_km).total - diesel_tco(
        model, daily_distance_km).energy
    litres = km / DIESEL_EQUIVALENTS[model]["kmpl"]
    diesel_price_breakeven = ((ev.total - diesel_no_fuel) / litres) if litres else None

    # Is the break-even distance one this vehicle can actually cover? The search
    # brackets 1-500 km/day with no reference to the recommended vehicle's own
    # range gate, so it can land beyond what /fleet-readiness/score calls
    # feasible. Still reported -- it is the honest answer to the question asked
    # -- but labelled, because offering it as procurement guidance would be
    # advice to run a route the same service says the vehicle cannot serve.
    max_feasible_km = _max_feasible_daily_km(model)
    unreachable = (breakeven_km is not None and max_feasible_km is not None
                   and breakeven_km > max_feasible_km)

    # THE NOTE MUST NOT CONTRADICT THE HEADLINE. `wins_here` is the same
    # comparison `compare()` reports as ev_cheaper, so an "it never pays back"
    # warning can no longer sit beside a response saying the EV is already
    # saving money at the distance actually asked about.
    wins_here = saving_at(daily_distance_km) > 0
    if breakeven_km is None:
        note = ("no break-even between 1 and 500 km/day: the EV is on the same side of "
                "the comparison across the whole range")
    elif unreachable and wins_here:
        note = (f"{model} only wins *permanently* above {breakeven_km} km/day, beyond the "
                f"{max_feasible_km:.1f} km/day its own range gate allows -- but it is already "
                f"cheaper at {daily_distance_km:g} km/day. A pack replacement lands inside the "
                f"five-year horizon at longer distances, so the saving is not monotonic in "
                f"distance; see also_wins_between_km")
    elif unreachable:
        note = (f"unreachable: {model} is only feasible up to {max_feasible_km:.1f} km/day on "
                f"its own range gate, so it never reaches the distance at which it would "
                f"permanently pay back")
    else:
        note = None

    # `is not None`, not truthiness: a break-even of exactly 0.0 is a real
    # answer ("free diesel would still lose") and was being reported as
    # "not computable".
    return {
        "daily_km_above_which_ev_wins": breakeven_km,
        "diesel_price_inr_per_litre_above_which_ev_wins": (
            round(diesel_price_breakeven, 2) if diesel_price_breakeven is not None else None),
        # Callers render these directly. Say WHY a value is absent, or why it is
        # not actionable, rather than letting a None print as the word "None"
        # next to a rupee sign.
        "distance_breakeven_note": note,
        "breakeven_is_reachable": None if breakeven_km is None else not unreachable,
        "max_feasible_daily_km": max_feasible_km,
        "ev_cheaper_at_this_distance": wins_here,
        # Bounded stretches below the permanent break-even where the EV also
        # wins. Non-empty only when a battery replacement makes the saving
        # non-monotonic; dropping them is what let the note contradict the
        # headline.
        "also_wins_between_km": also_wins or None,
        "note": "Below the distance break-even the EV's purchase premium is not "
                "recovered from lower running costs within 5 years.",
    }
