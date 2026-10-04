"""
Supply Chain Risk agent tests -- needs backend/models/risk_model.pkl
(gitignored, produced by train/train_risk_model.py). Auto-skips with a
clear reason if that file isn't present yet.
"""
from conftest import MODELS_DIR, skip_if_missing

skip_if_missing(MODELS_DIR / "risk_model.pkl")

from app.agents.risk_agent import get_risk_agent  # noqa: E402

HEALTHY_SHIPMENT = {
    "lead_time_deviation_pct": 5.0, "price_deviation_pct": 2.0, "reject_rate_pct": 1.8,
    "days_since_last_audit": 40.0, "single_sourced": 0, "geopolitical_risk": 0.2,
    "supplier_concentration_pct": 25.0, "volume_deviation_pct": 5.0,
}

SUSPICIOUS_SHIPMENT = {
    "lead_time_deviation_pct": 85.0, "price_deviation_pct": 45.0, "reject_rate_pct": 1.0,
    "days_since_last_audit": 200.0, "single_sourced": 1, "geopolitical_risk": 0.8,
    "supplier_concentration_pct": 90.0, "volume_deviation_pct": 10.0,
}


def test_healthy_shipment_scores_low():
    agent = get_risk_agent()
    result = agent.score(HEALTHY_SHIPMENT)
    assert result["risk_band"] == "healthy"
    assert result["flagged"] is False


def test_suspicious_shipment_flags_higher_than_healthy():
    """Directional check (this is anomaly detection, not a classifier
    trained for 100% separation -- see the submission doc's honest
    precision/recall numbers) rather than asserting an exact band."""
    agent = get_risk_agent()
    healthy_score = agent.score(HEALTHY_SHIPMENT)["risk_score"]
    suspicious_score = agent.score(SUSPICIOUS_SHIPMENT)["risk_score"]
    assert suspicious_score > healthy_score


def test_missing_required_field_raises():
    agent = get_risk_agent()
    incomplete = {k: v for k, v in HEALTHY_SHIPMENT.items() if k != "reject_rate_pct"}
    try:
        agent.score(incomplete)
        assert False, "expected ValueError for missing field"
    except ValueError as e:
        assert "reject_rate_pct" in str(e)


def test_reason_is_provided_when_flagged():
    agent = get_risk_agent()
    result = agent.score(SUSPICIOUS_SHIPMENT)
    assert isinstance(result["reason"], str) and len(result["reason"]) > 0


def test_every_shipment_in_the_shipped_dataset_is_acceptable_to_the_request_schema():
    """The request schema must accept the feature distribution the model was fitted on.

    price_trend is an ABSOLUTE currency delta (unit_price.diff(3)/3, on prices
    from 2.71 to 69,461), but carried reject_rate_trend's +/-100 percentage-point
    bound. That rejected 379 of the 1,000 shipments in the project's own scored
    dataset with a 422 -- 35 of the 94 the dashboard renders as CRITICAL, and 27
    of the 79 true injected anomalies. Clipping to the bound to get past
    validation changed the answer: COBALT-SUP1-014 is stored at 0.4196/flagged
    and came back 0.2644/"watch".
    """
    import pandas as pd
    from app.schemas import ShipmentRiskRequest
    from conftest import DATA_PROCESSED_DIR

    df = pd.read_csv(DATA_PROCESSED_DIR / "supply_chain_shipments.csv")
    fields = [f for f in ShipmentRiskRequest.model_fields if f != "shipment_id"]
    rejected = []
    for row in df.itertuples(index=False):
        payload = {"shipment_id": row.shipment_id}
        payload.update({f: getattr(row, f) for f in fields})
        try:
            ShipmentRiskRequest(**payload)
        except Exception as exc:  # noqa: BLE001 - any validation failure is the bug
            rejected.append((row.shipment_id, str(exc).splitlines()[0]))

    assert not rejected, (
        f"{len(rejected)} of {len(df)} shipments in the project's own dataset are "
        f"rejected by its own request schema, e.g. {rejected[:3]}")


def test_the_api_reproduces_the_score_stored_for_every_flagged_shipment():
    """The dashboard's flagged table reads the CSV; the endpoint recomputes. They
    must agree about the same shipment, for all of them, not just the scoreable ones."""
    import pandas as pd
    from app.agents.risk_agent import get_risk_agent
    from conftest import DATA_PROCESSED_DIR

    agent = get_risk_agent()
    df = pd.read_csv(DATA_PROCESSED_DIR / "supply_chain_shipments.csv")
    flagged = df[df.flagged == 1]
    mismatches = []
    for row in flagged.itertuples(index=False):
        out = agent.score({c: getattr(row, c) for c in agent.feature_cols})
        if (round(out["risk_score"], 4) != round(row.risk_score, 4)
                or out["flagged"] != bool(row.flagged)):
            mismatches.append((row.shipment_id, row.risk_score, out["risk_score"]))
    assert not mismatches, f"{len(mismatches)} flagged shipments disagree: {mismatches[:3]}"
