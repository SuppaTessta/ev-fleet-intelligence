"""
Live telemetry simulator tests -- needs data/processed/battery_features.csv
(gitignored, produced by data/parse_new_batteries.py or equivalent) and the
trained battery model (the simulator calls the real BatteryAgent for each
tick). Auto-skips with a clear reason if either is missing.
"""
from conftest import skip_if_missing, MODELS_DIR, DATA_PROCESSED_DIR

skip_if_missing(DATA_PROCESSED_DIR / "battery_features.csv", MODELS_DIR / "battery_rul_model.pkl")

from app.agents import telemetry_simulator as ts  # noqa: E402


def setup_function():
    """Reset the shared simulation state before every test so tests don't
    leak position changes into each other -- these tests share module-level
    state on purpose, same as the real simulator does."""
    ts.reset_fleet()


def test_tick_returns_all_eight_trucks():
    readings = ts.advance_fleet_tick()
    assert len(readings) == 8
    assert {r["asset_id"] for r in readings} == {f"EV-TRUCK-{i:02d}" for i in range(1, 9)}


def test_position_advances_by_one_each_tick():
    first = {r["asset_id"]: r for r in ts.advance_fleet_tick()}
    second = {r["asset_id"]: r for r in ts.advance_fleet_tick()}
    for asset_id in first:
        if not second[asset_id]["historical_data_exhausted"]:
            assert second[asset_id]["cycle_number"] == first[asset_id]["cycle_number"] + 1
        else:
            assert second[asset_id]["cycle_number"] == first[asset_id]["cycle_number"]


def test_reset_returns_every_truck_to_its_starting_cutoff():
    starting_cutoffs = {asset_id: cutoff for _, cutoff, asset_id, _ in ts.FLEET}
    ts.advance_fleet_tick()
    ts.advance_fleet_tick()
    ts.advance_fleet_tick()
    ts.reset_fleet()
    first_after_reset = {r["asset_id"]: r["cycle_number"] for r in ts.advance_fleet_tick()}
    for asset_id, cutoff in starting_cutoffs.items():
        assert first_after_reset[asset_id] == cutoff + 1


def test_exhausted_truck_holds_rather_than_looping():
    for _ in range(400):  # comfortably past the longest recorded battery's cycle count
        readings = ts.advance_fleet_tick()
    truck_readings = {r["asset_id"]: r for r in readings}
    for asset_id, r in truck_readings.items():
        assert r["cycle_number"] == r["total_cycles_recorded"]
        assert r["historical_data_exhausted"] is True


def test_reading_includes_a_live_rul_prediction():
    readings = ts.advance_fleet_tick()
    for r in readings:
        assert "predicted_rul_cycles" in r
        assert r["risk_band"] in ("healthy", "watch", "critical")
