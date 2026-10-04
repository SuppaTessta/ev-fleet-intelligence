"""Compound Risk: the cross-agent correlation layer.

A truck's battery health, its cell supplier's risk score and its last quality
inspection are three separate numbers, none necessarily urgent alone. This asks
the question a fleet manager actually has: is the SAME truck showing a weak
signal across more than one of them at once?

Linkage caveat: there is no real bill of materials connecting a given truck's
cells to a given supplier, so FLEET_LINKAGE below is a fixed, documented mapping
standing in for what a traceability system would provide. The battery,
supplier-risk and quality models are all real; only the cross-linking is not.
"""

from functools import lru_cache

import pandas as pd

from app import config

PROC_DIR = config.PROCESSED_DATA_DIR

# asset_id -> (cell_supplier_id, last_quality_verdict)
# Illustrative BOM linkage — see module docstring.
FLEET_LINKAGE = {
    "EV-TRUCK-01": ("LFP-SUP2", "ok"),
    # NMC-SUP1 is flagged on 20.0% of its shipments (5/25), which reaches the
    # "critical" supplier band at >=15% and is what makes the compound-risk demo
    # work. Four suppliers tie at that rate -- COBALT-SUP1, COBALT-SUP4,
    # NMC-SUP1, REM-SUP3 -- behind BMS-SUP2 (100.0%) and COBALT-SUP2 (28.0%).
    "EV-TRUCK-02": ("NMC-SUP1", "ok"),
    "EV-TRUCK-03": ("LFP-SUP1", "ok"),
    "EV-TRUCK-04": ("NMC-SUP1", "defective"),    # same supplier as TRUCK-02, and a defect on top
    "EV-TRUCK-05": ("LFP-SUP3", "ok"),
    "EV-TRUCK-06": ("NMC-SUP4", "ok"),
    "EV-TRUCK-07": ("NMC-SUP2", "defective"),
    "EV-TRUCK-08": ("LFP-SUP4", "ok"),
}


@lru_cache(maxsize=1)
def _shipments() -> pd.DataFrame:
    """Shipment records, read once per process rather than once per request.

    The Fleet Command Center scores every truck on each page load, so a bare
    read_csv here re-parsed a 1,000-row file per truck. Cached deliberately: the
    file is a training artifact that only changes when train_risk_model.py
    reruns, which needs a restart to pick up anyway. A missing file still raises
    FileNotFoundError, which app.errors maps to 503 rather than an empty result.
    """
    return pd.read_csv(PROC_DIR / "supply_chain_shipments.csv")


def supplier_risk_level(supplier_id: str) -> dict:
    shipments = _shipments()
    rows = shipments[shipments.supplier_id == supplier_id]
    if len(rows) == 0:
        return {"flagged_pct": 0.0, "level": "unknown"}
    flagged_pct = round(100 * rows.flagged.mean(), 1)
    level = "critical" if flagged_pct >= 15 else "watch" if flagged_pct >= 5 else "healthy"
    return {"flagged_pct": flagged_pct, "level": level}


def compute_compound_risk(asset_id: str, battery_risk_band: str) -> dict:
    """Correlate battery, supplier and quality signals for one asset.

    Missing data is not a risk signal. An asset absent from FLEET_LINKAGE
    contributes nothing rather than scoring as a weak signal, and the unknowns
    are counted separately so a caller can distinguish "no evidence of risk"
    from "no evidence at all".
    """
    known = asset_id in FLEET_LINKAGE
    supplier_id, quality_verdict = FLEET_LINKAGE.get(asset_id, (None, "unknown"))
    supplier = supplier_risk_level(supplier_id) if supplier_id else {"flagged_pct": 0.0, "level": "unknown"}

    # One vocabulary per slot. A single shared score table lets the quality
    # vocabulary into the battery slot, where an out-of-vocabulary value such as
    # "defective" would score 2 and escalate an otherwise healthy truck -- via
    # the dashboard's priority map, into a 10-day maintenance deadline instead of
    # 30. Unrecognised values score 0 and increment unknown_signal_count.
    BAND_SCORES = {"healthy": 0, "watch": 1, "critical": 2}      # battery, supplier
    QUALITY_SCORES = {"ok": 0, "defective": 2}                   # inspection verdict

    scores, unknown_count = [], 0
    for value, vocabulary in ((battery_risk_band, BAND_SCORES),
                               (supplier["level"], BAND_SCORES),
                               (quality_verdict, QUALITY_SCORES)):
        if value in vocabulary:
            scores.append(vocabulary[value])
        else:
            scores.append(0)
            unknown_count += 1

    active_signals = sum(1 for s in scores if s > 0)
    compound_score = sum(scores)

    if unknown_count == len(scores):
        priority = "insufficient data"
    elif active_signals >= 2:
        priority = "compound priority"
    elif compound_score >= 2:
        priority = "single-signal priority"
    else:
        priority = "routine"

    return {
        "asset_id": asset_id,
        "battery_risk_band": battery_risk_band,
        "cell_supplier_id": (supplier_id or "").strip(),
        "supplier_risk_level": supplier["level"],
        "supplier_flagged_pct": supplier["flagged_pct"],
        "quality_verdict": quality_verdict,
        "active_signal_count": active_signals,
        "unknown_signal_count": unknown_count,
        "asset_known_to_bom": known,
        "compound_score": compound_score,
        "priority": priority,
    }
