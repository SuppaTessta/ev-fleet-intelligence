"""
Supply Chain Risk agent tests -- needs backend/models/risk_model.pkl
(gitignored, produced by train/train_risk_model.py). Auto-skips with a
clear reason if that file isn't present yet.
"""
from conftest import skip_if_missing, MODELS_DIR

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
