"""
Carbon Tracker tests -- pure lookup/arithmetic against real cited constants
(CEA grid data, cross-checked diesel factors), no trained model or data file
needed, so these always run (including in CI).
"""
import pytest
from app.agents.carbon_agent import GRID_EMISSION_FACTOR_TREND, compute_savings, fleet_wide_progress


def test_known_vehicle_returns_positive_savings():
    result = compute_savings("Tata Ace EV", daily_distance_km=60.0)
    assert result["savings_pct"] > 0
    assert result["annual_co2_savings_kg"] > 0
    # grid is still mostly fossil -- savings should be real but not fantastical
    assert 0 < result["savings_pct"] < 100


def test_unknown_vehicle_raises():
    with pytest.raises(ValueError):
        compute_savings("Tesla Cybertruck", daily_distance_km=60.0)


def test_zero_distance_is_zero_savings():
    result = compute_savings("Tata Ace EV", daily_distance_km=0.0)
    assert result["daily_co2_savings_kg"] == 0
    assert result["annual_co2_savings_kg"] == 0


def test_grid_factor_trend_has_five_years_and_current_value_used_by_default():
    assert len(GRID_EMISSION_FACTOR_TREND) == 5
    assert GRID_EMISSION_FACTOR_TREND["2024-25"] == 0.7097
    result = compute_savings("Tata Ace EV", daily_distance_km=60.0)
    assert result["grid_factor_used"] == GRID_EMISSION_FACTOR_TREND["2024-25"]


def test_fleet_wide_progress_aggregates_correctly():
    assignments = [("Tata Ace EV", 60.0), ("Tata Ace EV", 40.0), ("Mahindra Treo Zor", 50.0)]
    result = fleet_wide_progress(assignments)
    assert result["fleet_size"] == 3
    expected_total = sum(compute_savings(v, d)["annual_co2_savings_kg"] for v, d in assignments)
    assert result["total_annual_co2_savings_kg"] == pytest.approx(expected_total, abs=0.1)


# ---------------------------------------------------------------- Scope 2 boundary

def test_metered_energy_exceeds_battery_energy():
    """Scope 2 bills PURCHASED electricity, so the figure multiplied by the grid
    factor must be the meter-side one. The agent used to multiply battery-side
    energy (pack_kwh / range_km) by the grid factor, understating EV emissions."""
    from app.agents.carbon_agent import CHARGER_EFFICIENCY, compute_savings
    r = compute_savings("Tata Ace EV", 60)
    assert r["kwh_per_km_at_meter"] > r["kwh_per_km_at_battery"]
    assert r["kwh_per_km_at_meter"] == pytest.approx(
        r["kwh_per_km_at_battery"] / CHARGER_EFFICIENCY, rel=1e-3)


def test_tata_ace_saving_is_the_corrected_figure_not_the_old_one():
    """Regression guard on a number that was published wrong. 19.7% came from
    omitting charging losses; inside the declared Scope 2 boundary it is ~10.8%."""
    from app.agents.carbon_agent import compute_savings
    r = compute_savings("Tata Ace EV", 60)
    assert r["savings_pct"] == pytest.approx(10.8, abs=0.3)
    assert r["savings_pct"] < 19.0, "charging losses appear to have been dropped again"


def test_breakeven_grid_factor_is_above_todays_indian_grid():
    """The saving is real but the margin is thin -- worth stating as a number
    rather than a percentage, because it is what actually changes over time."""
    from app.agents.carbon_agent import CURRENT_GRID_FACTOR_KG_PER_KWH, compute_savings
    r = compute_savings("Tata Ace EV", 60)
    assert r["breakeven_grid_factor_tco2_per_mwh"] > CURRENT_GRID_FACTOR_KG_PER_KWH
    assert r["breakeven_grid_factor_tco2_per_mwh"] == pytest.approx(0.795, abs=0.01)


def test_scope3_sensitivity_is_reported_but_is_not_the_headline():
    """Including T&D losses flips the Tata Ace to worse-than-diesel. That is a
    Scope 3 result and must stay labelled as a sensitivity, never the headline."""
    from app.agents.carbon_agent import compute_savings
    r = compute_savings("Tata Ace EV", 60)
    s3 = r["scope3_sensitivity"]
    assert s3["if_grid_td_losses_included_savings_pct"] < r["savings_pct"]
    assert "outside" in s3["note"].lower() or "scope 3" in s3["note"].lower()
