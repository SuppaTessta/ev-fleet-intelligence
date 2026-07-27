"""
Carbon Tracker tests -- pure lookup/arithmetic against real cited constants
(CEA grid data, cross-checked diesel factors), no trained model or data file
needed, so these always run (including in CI).
"""
import pytest
from app.agents.carbon_agent import compute_savings, fleet_wide_progress, GRID_EMISSION_FACTOR_TREND


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
