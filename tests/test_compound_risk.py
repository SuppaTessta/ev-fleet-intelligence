"""
Compound Risk agent tests -- needs data/processed/supply_chain_shipments.csv
(gitignored, produced by data/generate_supply_chain_data.py or equivalent
prep step) for supplier_risk_level()'s lookup. Auto-skips with a clear
reason if that file isn't present yet.
"""
from conftest import DATA_PROCESSED_DIR, skip_if_missing

skip_if_missing(DATA_PROCESSED_DIR / "supply_chain_shipments.csv")

from app.agents.compound_risk_agent import FLEET_LINKAGE, compute_compound_risk  # noqa: E402


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


def test_unknown_asset_is_not_escalated():
    """A healthy truck missing from the BOM linkage must not be escalated.

    This used to return "compound priority" -- the top of the maintenance queue --
    because unknown supplier and unknown quality each scored 1, the same as
    "watch", so absence of data read as two active risk signals. The earlier
    version of this test was named ..._does_not_crash and asserted only on the
    echoed fields, so it passed throughout.
    """
    result = compute_compound_risk("EV-TRUCK-NONEXISTENT", battery_risk_band="healthy")
    assert result["cell_supplier_id"] == ""
    assert result["quality_verdict"] == "unknown"
    assert result["asset_known_to_bom"] is False
    assert result["active_signal_count"] == 0, "missing data must not count as a risk signal"
    assert result["compound_score"] == 0
    assert result["priority"] != "compound priority"


def test_unknown_everything_reports_insufficient_data():
    """All three inputs unavailable is distinct from all three being healthy."""
    result = compute_compound_risk("EV-TRUCK-NONEXISTENT", battery_risk_band="not-a-band")
    assert result["unknown_signal_count"] == 3
    assert result["priority"] == "insufficient data"


def test_garbage_battery_band_does_not_invent_risk():
    """A malformed band (e.g. wrong case) must not silently become a weak signal."""
    result = compute_compound_risk("EV-TRUCK-01", battery_risk_band="Watch")
    assert result["compound_score"] == 0
    assert result["priority"] == "routine"


def test_fleet_linkage_covers_all_eight_demo_trucks():
    expected = {f"EV-TRUCK-{i:02d}" for i in range(1, 9)}
    assert set(FLEET_LINKAGE.keys()) == expected


def test_quality_vocabulary_cannot_leak_into_the_battery_slot():
    """Each signal has its own vocabulary; an out-of-vocabulary value never escalates.

    All three slots shared one score table, so battery_risk_band="defective" --
    a value /battery/predict-rul cannot emit, and one the field's own OpenAPI
    description rules out -- scored 2 and moved a healthy truck from "routine"
    to "single-signal priority". Through the dashboard's priority_map that is a
    10-day maintenance deadline instead of 30.
    """
    for bogus_band in ("defective", "ok", "Critical", " critical", "zzz", ""):
        r = compute_compound_risk("EV-TRUCK-01", battery_risk_band=bogus_band)
        assert r["compound_score"] == 0, f"{bogus_band!r} invented a battery signal"
        assert r["active_signal_count"] == 0, f"{bogus_band!r} counted as active"
        assert r["unknown_signal_count"] >= 1, f"{bogus_band!r} was treated as known"
        assert r["priority"] == "routine", f"{bogus_band!r} escalated to {r['priority']!r}"


def test_only_real_battery_bands_are_counted_as_evidence():
    """The three bands the battery agent actually emits must score, and must be
    the ONLY things that score in that slot."""
    for band, expected_score in (("healthy", 0), ("watch", 1), ("critical", 2)):
        r = compute_compound_risk("EV-TRUCK-01", battery_risk_band=band)
        # EV-TRUCK-01's supplier is healthy and its quality verdict is ok, so the
        # battery band is the only thing that can contribute.
        assert r["compound_score"] == expected_score
        assert r["unknown_signal_count"] == 0
