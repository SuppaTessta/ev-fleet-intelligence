"""
Compound Risk Agent — the cross-agent correlation layer.

Individually, a truck's battery agent, its cell supplier's risk score, and
its last quality inspection are three separate numbers. None of them alone
might look urgent. This agent asks the actual question a fleet manager
cares about: is the SAME truck showing up as a weak signal across more
than one of these at once? That's a compound-risk case — worth a look even
when no single metric alone crossed a "critical" bar.

Linking note (documented honestly, not hidden): the demo fleet is built
from real NASA battery cycling data, and supplier/quality data comes from
the other two agents' own (synthetic and real-image-based) sources. There's
no real bill-of-materials connecting "this exact truck's cells" to "this
exact supplier" — so each truck is assigned a cell supplier and a last
quality-inspection result via a fixed, documented mapping below, standing
in for what a real BOM/traceability system would provide. The battery,
supplier-risk, and quality models themselves are all real; only the
cross-linking is illustrative.
"""

from pathlib import Path
import pandas as pd

PROC_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

# asset_id -> (cell_supplier_id, last_quality_verdict)
# Illustrative BOM linkage — see module docstring.
FLEET_LINKAGE = {
    "EV-TRUCK-01": ("LFP-SUP2", "ok"),
    "EV-TRUCK-02": ("NMC-SUP1", "ok"),          # NMC-SUP1 has the most flagged shipments
    "EV-TRUCK-03": ("LFP-SUP1", "ok"),
    "EV-TRUCK-04": ("NMC-SUP1", "defective"),    # same risky supplier as TRUCK-02, and a defect on top
    "EV-TRUCK-05": ("LFP-SUP3", "ok"),
    "EV-TRUCK-06": ("NMC-SUP4", "ok"),
    "EV-TRUCK-07": ("NMC-SUP2", "defective"),
    "EV-TRUCK-08": ("LFP-SUP4", "ok"),
}


def supplier_risk_level(supplier_id: str) -> dict:
    df = pd.read_csv(PROC_DIR / "supply_chain_shipments.csv")
    rows = df[df.supplier_id == supplier_id]
    if len(rows) == 0:
        return {"flagged_pct": 0.0, "level": "unknown"}
    flagged_pct = round(100 * rows.flagged.mean(), 1)
    level = "critical" if flagged_pct >= 15 else "watch" if flagged_pct >= 5 else "healthy"
    return {"flagged_pct": flagged_pct, "level": level}


def compute_compound_risk(asset_id: str, battery_risk_band: str) -> dict:
    supplier_id, quality_verdict = FLEET_LINKAGE.get(asset_id, (None, "unknown"))
    supplier = supplier_risk_level(supplier_id) if supplier_id else {"flagged_pct": 0.0, "level": "unknown"}

    level_score = {"healthy": 0, "ok": 0, "watch": 1, "unknown": 1, "critical": 2, "defective": 2}
    battery_score = level_score.get(battery_risk_band, 1)
    supplier_score = level_score.get(supplier["level"], 1)
    quality_score = level_score.get(quality_verdict, 1)

    active_signals = sum(1 for s in [battery_score, supplier_score, quality_score] if s > 0)
    compound_score = battery_score + supplier_score + quality_score

    if active_signals >= 2:
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
        "compound_score": compound_score,
        "priority": priority,
    }
