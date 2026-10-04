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


def test_not_viable_reason_never_claims_a_blocker_that_is_false():
    """The reason is read by someone making a purchase decision. It must not
    name a constraint the catalog actually clears.

    Blockers were read off a single `best_attempt` candidate but worded as
    claims about the whole catalog: at 110 km/day / 650 kg / 6 h dwell it said
    "no catalog option covers the required 138km range" -- the Tata Ace's 154 km
    does -- and never mentioned payload, the gate that actually excludes it.
    """
    from app.agents.fleet_readiness_agent import EV_CATALOG, RANGE_SAFETY_MARGIN

    cases = [(110.0, 650.0, 6.0), (110.0, 650.0, 2.0), (300.0, 100.0, 12.0),
             (30.0, 2000.0, 12.0), (30.0, 200.0, 0.5)]
    for distance, payload, dwell in cases:
        result = score_vehicle(distance, payload, dwell)
        assert result["readiness"] == "not_yet_viable", (distance, payload, dwell)
        reason = result["reason"]
        required_range = distance * RANGE_SAFETY_MARGIN

        # a catalog-wide claim is only allowed when every option really fails it
        if "no catalog option covers the required" in reason:
            assert all(ev["range_km"] < required_range for ev in EV_CATALOG), (
                f"reason claims no option has the range, but one does: {reason}")
        if "exceeds every catalog option" in reason:
            assert all(ev["payload_kg"] < payload for ev in EV_CATALOG), (
                f"reason claims no option has the payload, but one does: {reason}")
        if "too short for any catalog option" in reason:
            assert all(ev["charge_time_hours"] > dwell for ev in EV_CATALOG), (
                f"reason claims no option can charge in time, but one can: {reason}")

        # and every option that failed must be accounted for
        for ev in EV_CATALOG:
            assert ev["model"] in reason, f"{ev['model']} unexplained in: {reason}"


def test_the_shipped_csv_is_reproducible_from_its_own_columns():
    """The dashboard prints the inputs and the score side by side; a reader can
    check that arithmetic, so it has to hold.

    The generator scored the UNROUNDED draws and published rounded inputs, so
    confidence_pct was unreproducible on 8 of 60 rows. The CSV also kept the
    pre-fix blocker wording, including two rows asserting a charge-window
    blocker the catalog clears.
    """
    from pathlib import Path

    import pandas as pd
    import pytest

    csv = Path(__file__).resolve().parents[1] / "data" / "processed" / "fleet_readiness.csv"
    if not csv.exists():
        pytest.skip("fleet_readiness.csv not generated yet -- run train/train_fleet_readiness.py")

    df = pd.read_csv(csv)
    mismatches = []
    for row in df.itertuples(index=False):
        scored = score_vehicle(row.daily_distance_km, row.avg_payload_kg, row.dwell_time_hours)
        if scored["readiness"] != row.readiness:
            mismatches.append((row.vehicle_id, "readiness"))
        if abs(scored["confidence_pct"] - row.confidence_pct) > 1e-9:
            mismatches.append((row.vehicle_id, "confidence_pct",
                               row.confidence_pct, scored["confidence_pct"]))
        if scored["reason"] != row.reason:
            mismatches.append((row.vehicle_id, "reason"))
    assert not mismatches, f"{len(mismatches)} published rows do not reproduce: {mismatches[:4]}"
