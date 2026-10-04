"""
Five-year TCO model, and the feasibility/economics split it enables.

Needs no artifacts, so it runs in CI.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.agents.fleet_readiness_agent import score_vehicle_with_economics  # noqa: E402
from app.agents.tco import DIESEL_EQUIVALENTS, compare, diesel_tco, ev_tco  # noqa: E402

MODELS = list(DIESEL_EQUIVALENTS)
PRICES = {"Mahindra Treo Zor": 3.3, "Euler HiLoad EV": 4.2, "Tata Ace EV": 10.51}


# ---------------------------------------------------------------- arithmetic

@pytest.mark.parametrize("model", MODELS)
def test_total_is_the_sum_of_its_parts(model):
    b = ev_tco(model, 60, PRICES[model])
    assert b.total == pytest.approx(
        b.purchase + b.energy + b.maintenance + b.battery_replacement + b.residual_value)


@pytest.mark.parametrize("model", MODELS)
def test_residual_value_reduces_cost(model):
    """Signed as a negative cost, not a positive one -- getting this backwards
    would inflate every total by twice the residual."""
    assert ev_tco(model, 60, PRICES[model]).residual_value < 0
    assert diesel_tco(model, 60).residual_value < 0


@pytest.mark.parametrize("model", MODELS)
def test_cost_rises_with_distance(model):
    cheap = compare(model, 30, PRICES[model])["ev"]["total_inr"]
    dear = compare(model, 120, PRICES[model])["ev"]["total_inr"]
    assert dear > cheap


@pytest.mark.parametrize("model", MODELS)
def test_ev_energy_uses_metered_not_battery_side_consumption(model):
    """The fleet pays for what the meter records, including charging loss. Using
    battery-side energy would understate the running cost by ~10% -- the same
    boundary error the carbon agent had."""
    from app.agents.carbon_agent import CHARGER_EFFICIENCY, VEHICLE_ENERGY_PROFILES
    from app.agents.tco import ELECTRICITY_TARIFF_INR_PER_KWH, HORIZON_YEARS, OPERATING_DAYS_PER_YEAR

    km = 60 * OPERATING_DAYS_PER_YEAR * HORIZON_YEARS
    battery_side = VEHICLE_ENERGY_PROFILES[model][0] * km * ELECTRICITY_TARIFF_INR_PER_KWH
    actual = ev_tco(model, 60, PRICES[model]).energy
    assert actual > battery_side
    assert actual == pytest.approx(battery_side / CHARGER_EFFICIENCY)


# ---------------------------------------------------------------- the finding

def test_tata_ace_is_uneconomic_at_low_utilisation():
    """The result that justifies this whole model existing. The Tata Ace EV is
    physically capable on a 40 km/day route and loses money against its diesel
    twin, because a Rs 10.51L purchase price against Rs 6.2L needs distance to
    recover. Sticker-price ranking cannot express this."""
    result = compare("Tata Ace EV", 40, PRICES["Tata Ace EV"])
    assert not result["ev_cheaper"]
    assert result["ev_saving_inr"] < 0


def test_tata_ace_becomes_economic_at_high_utilisation():
    assert compare("Tata Ace EV", 120, PRICES["Tata Ace EV"])["ev_cheaper"]


def test_breakeven_distance_actually_separates_the_two_regimes():
    """The reported break-even must be the real crossing point, not decoration."""
    model = "Tata Ace EV"
    breakeven = compare(model, 60, PRICES[model])["breakeven"]["daily_km_above_which_ev_wins"]
    assert breakeven is not None
    assert not compare(model, breakeven - 5, PRICES[model])["ev_cheaper"]
    assert compare(model, breakeven + 5, PRICES[model])["ev_cheaper"]


def test_battery_replacement_is_charged_when_the_pack_will_not_last():
    """Wires the RUL model into a rupee figure: a pack predicted to die inside
    the horizon costs a replacement."""
    healthy = compare("Tata Ace EV", 100, PRICES["Tata Ace EV"],
                      expected_cycles_remaining=100_000)
    worn = compare("Tata Ace EV", 100, PRICES["Tata Ace EV"],
                   expected_cycles_remaining=10)
    assert worn["ev"]["battery_replacement_inr"] > 0
    assert healthy["ev"]["battery_replacement_inr"] == 0
    assert worn["ev"]["total_inr"] > healthy["ev"]["total_inr"]


def test_unknown_model_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        compare("Not A Real EV", 60, 5.0)


# ---------------------------------------------------------------- the two gates

def test_feasible_and_economic_are_reported_separately():
    """A vehicle can be perfectly capable and still not worth buying. Collapsing
    the two into one score would hide exactly the case the TCO model exists to
    surface."""
    # payload > 550 rules out the Treo Zor, dwell < 5h rules out the Euler,
    # so only the Tata Ace survives -- at 40 km/day, where it loses money
    result = score_vehicle_with_economics(40, 580, 3.0)
    assert result["readiness"] == "ready"           # physically capable
    assert result["verdict"] == "ready_but_uneconomic"   # and a bad purchase
    assert result["economics"]["ev_saving_inr"] < 0


def test_verdict_flips_to_economic_above_the_breakeven():
    result = score_vehicle_with_economics(90, 580, 3.0)
    assert result["verdict"] == "ready_and_economic"


def test_infeasible_routes_carry_no_economics():
    result = score_vehicle_with_economics(400, 300, 10.0)
    assert result["verdict"] == "not_yet_viable"
    assert result["economics"] is None


def test_score_vehicle_stays_free_of_economic_assumptions():
    """The pure gate must not gain price or tariff assumptions -- a disagreement
    about the diesel price should not change whether a vehicle is capable."""
    from app.agents.fleet_readiness_agent import score_vehicle

    plain = score_vehicle(40, 580, 3.0)
    assert "economics" not in plain
    assert "verdict" not in plain
    assert plain["readiness"] == "ready"


def test_zero_distance_is_a_clean_error_not_a_crash():
    """The request schema permits daily_distance_km=0 (a parked or seasonal
    vehicle is real), and cost-per-km is then 0/0. This raised
    ZeroDivisionError -> HTTP 500 until guarded."""
    with pytest.raises(ValueError, match="greater than 0"):
        compare("Tata Ace EV", 0, PRICES["Tata Ace EV"])


# ------------------------------------------- the response must reconcile

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("distance", [5, 40, 80.3, 93.3, 150])
def test_every_returned_figure_reconciles_with_the_others(model, distance):
    """The dashboard prints components, totals and the saving side by side.

    They were each rounded independently from unrounded intermediates, so at
    80.3 km/day the diesel components summed to 770,183 against a total_inr of
    770,182, and ev_saving_inr disagreed with diesel.total - ev.total by a rupee
    in 55 of 227 sampled distances.
    """
    r = compare(model, distance, PRICES[model])
    parts = ("purchase_inr", "energy_inr", "maintenance_inr",
             "battery_replacement_inr", "residual_value_inr")
    for block in ("ev", "diesel"):
        assert sum(r[block][p] for p in parts) == r[block]["total_inr"], (
            f"{block} components do not sum to total_inr at {distance} km/day")
    assert r["ev_saving_inr"] == r["diesel"]["total_inr"] - r["ev"]["total_inr"]
    assert r["ev_cheaper"] is (r["ev_saving_inr"] > 0)


@pytest.mark.parametrize("model", MODELS)
def test_the_ev_really_does_win_at_the_reported_breakeven(model):
    """`daily_km_above_which_ev_wins` has to satisfy its own name.

    It was round-to-nearest on the bisection midpoint, so the Tata Ace reported
    68.2 against a true crossover of 68.238 -- posting 68.2 back to the endpoint
    returned ev_cheaper false.
    """
    be = compare(model, 60, PRICES[model])["breakeven"]
    km = be["daily_km_above_which_ev_wins"]
    if km is None:
        assert be["distance_breakeven_note"], "an absent break-even must say why"
        return
    assert compare(model, km, PRICES[model])["ev_cheaper"], (
        f"{model}: reported break-even {km} km/day but the EV is not cheaper there")
    # and just below it, the EV should still be losing -- otherwise the reported
    # value is not the threshold, just some distance above it
    assert not compare(model, round(km - 0.2, 1), PRICES[model])["ev_cheaper"]


@pytest.mark.parametrize("model", MODELS)
def test_an_unreachable_breakeven_is_labelled_as_such(model):
    """A break-even beyond the vehicle's own range gate is advice to run a route
    the same service reports as not_yet_viable. It may be reported, but not
    silently."""
    be = compare(model, 40, PRICES[model], expected_cycles_remaining=390)["breakeven"]
    km, reachable = be["daily_km_above_which_ev_wins"], be["breakeven_is_reachable"]
    if km is None:
        return
    assert reachable is (km <= be["max_feasible_daily_km"])
    if not reachable:
        assert "unreachable" in (be["distance_breakeven_note"] or "")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("cycles", [None, 200, 900, 2000, 5000])
def test_the_breakeven_block_never_contradicts_the_headline(model, cycles):
    """`saving_at` is NOT monotonic once a pack replacement can fire.

    The Rs 383,400 replacement lands the moment lifetime_km / range_km exceeds
    the remaining cycles, so the saving steps sharply DOWN at that distance and
    there can be three sign changes, not one. A bisection needs a single root;
    given two it converged on the upper one, and at 80 km/day with cycles=900
    the response reported ev_cheaper true, a Rs 62,690 saving and verdict
    "ready_and_economic" while the break-even block beside it said the vehicle
    "never reaches the distance at which it would pay back".
    """
    r = compare(model, 80, PRICES[model], cycles)
    be = r["breakeven"]

    # the block must agree with the headline about the distance actually asked about
    assert be["ev_cheaper_at_this_distance"] == r["ev_cheaper"]
    if be["ev_cheaper_at_this_distance"]:
        assert "never reaches" not in (be["distance_breakeven_note"] or ""), (
            "a 'never pays back' note beside a response that says it already has")

    km = be["daily_km_above_which_ev_wins"]
    if km is None:
        assert be["distance_breakeven_note"], "an absent break-even must say why"
        return

    # "above which the EV wins" has to mean it keeps winning
    for d in (km, km + 10, km + 50, 500.0):
        assert compare(model, d, PRICES[model], cycles)["ev_cheaper"], (
            f"{model} (cycles={cycles}): reported break-even {km} but the EV loses at {d}")

    # any bounded winning stretch below it must be real and below it
    for lo, hi in be["also_wins_between_km"] or []:
        assert hi < km
        assert compare(model, lo, PRICES[model], cycles)["ev_cheaper"]
        assert compare(model, (lo + hi) / 2, PRICES[model], cycles)["ev_cheaper"]
