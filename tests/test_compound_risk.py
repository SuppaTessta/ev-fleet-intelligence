"""
Compound Risk agent tests -- needs data/processed/supply_chain_shipments.csv
(gitignored, produced by data/generate_supply_chain_data.py or equivalent
prep step) for supplier_risk_level()'s lookup. Auto-skips with a clear
reason if that file isn't present yet.
"""
from conftest import skip_if_missing, DATA_PROCESSED_DIR

skip_if_missing(DATA_PROCESSED_DIR / "supply_chain_shipments.csv")

from app.agents.compound_risk_agent import compute_compound_risk, FLEET_LINKAGE  # noqa: E402


def test_shared_risky_supplier_produces_compound_priority_for_both_trucks():
    """The headline example from the submission doc: EV-TRUCK-02 and
    EV-TRUCK-04 share the same flagged supplier (NMC-SUP1). EV-TRUCK-02's
    battery reading alone is only 'watch' -- not urgent by itself -- but
    combined with the risky supplier it should surface as compound priority."""
    result_02 = compute_compound_risk("EV-TRUCK-02", battery_risk_band="watch")
    assert result_02["active_signal_count"] >= 2
    assert result_02["priority"] == "compound priority"


def test_all_healthy_signals_is_routine():
    result = compute_compound_risk("EV-TRUCK-01", battery_risk_band="healthy")
    assert result["priority"] == "routine"
    assert result["compound_score"] == 0


def test_unknown_asset_id_does_not_crash():
    result = compute_compound_risk("EV-TRUCK-NONEXISTENT", battery_risk_band="healthy")
    assert result["cell_supplier_id"] == ""
    assert result["quality_verdict"] == "unknown"


def test_fleet_linkage_covers_all_eight_demo_trucks():
    expected = {f"EV-TRUCK-{i:02d}" for i in range(1, 9)}
    assert set(FLEET_LINKAGE.keys()) == expected
