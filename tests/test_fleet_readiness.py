"""
Fleet Electrification Readiness tests -- pure rule-based scoring against
real OEM spec constants (EV_CATALOG), no trained model or data file needed,
so these always run (including in CI).
"""
from train.train_fleet_readiness import score_vehicle


def test_easy_case_is_ready():
    """Short daily distance, light payload, generous dwell time -- should
    clear every real EV in the catalog comfortably."""
    result = score_vehicle(daily_distance_km=30.0, avg_payload_kg=200.0, dwell_time_hours=12.0)
    assert result["readiness"] == "ready"
    assert result["recommended_model"] is not None
    assert result["confidence_pct"] > 0


def test_impossible_range_is_not_yet_viable():
    """No EV in the catalog covers a 2000km/day duty cycle -- must not be
    silently marked ready."""
    result = score_vehicle(daily_distance_km=2000.0, avg_payload_kg=200.0, dwell_time_hours=12.0)
    assert result["readiness"] == "not_yet_viable"
    assert result["recommended_model"] is None
    assert len(result["reason"]) > 0  # blockers get joined into "reason" for this branch


def test_recommended_model_is_cheapest_feasible_option():
    """When more than one EV would technically fit, the recommendation
    should be the cheapest one that clears every gate, not just any of them."""
    result = score_vehicle(daily_distance_km=30.0, avg_payload_kg=100.0, dwell_time_hours=12.0)
    feasible = [c for c in result["all_options"] if c["feasible"]]
    cheapest = min(feasible, key=lambda c: c["price_inr_lakh"])
    assert result["recommended_model"] == cheapest["model"]
