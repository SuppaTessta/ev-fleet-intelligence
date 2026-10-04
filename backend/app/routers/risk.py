from fastapi import APIRouter

from app.agents.risk_agent import get_risk_agent
from app.errors import DomainValidationError, ModelUnavailableError
from app.schemas import ShipmentRiskRequest, ShipmentRiskResponse

router = APIRouter(prefix="/risk", tags=["Supply Chain Risk Agent"])


@router.post("/score-shipment", response_model=ShipmentRiskResponse)
def score_shipment(req: ShipmentRiskRequest):
    # No blanket `except Exception -> 400`. A missing artifact is a 503 and a
    # bug is a 500; both are handled centrally in app/errors.py, which logs the
    # traceback under a request id instead of returning it to the caller.
    try:
        agent = get_risk_agent()
    except (FileNotFoundError, OSError) as e:
        raise ModelUnavailableError(f"risk model could not be loaded: {e}") from e

    try:
        result = agent.score(req.model_dump(exclude={"shipment_id"}))
    except ValueError as e:
        raise DomainValidationError(str(e)) from e

    return ShipmentRiskResponse(shipment_id=req.shipment_id, **result)


@router.get("/metadata", summary="Decision thresholds and held-out metrics")
def risk_metadata():
    """The model's own decision boundary and held-out metrics.

    A UI cannot re-derive a rule the model owns, so it has to be able to ask.
    Without this the dashboard invented its own cut-off and disagreed with the
    backend about which shipments were critical.
    """
    try:
        agent = get_risk_agent()
    except (FileNotFoundError, OSError) as e:
        raise ModelUnavailableError(f"risk model could not be loaded: {e}") from e
    return {
        "watch_threshold": agent.watch_threshold,
        # Per-archetype recall of the CRITICAL flag, read from the artifact so
        # the UI never hardcodes a figure that can drift from the model.
        "archetype_recall": agent.metrics.get("archetype_recall_shipped_dataset", {}),
        "bands": {
            "critical": "the model's own binary anomaly flag",
            "watch": f"risk_score >= {agent.watch_threshold:.4f} (75th percentile of the scored shipment set, computed in train_risk_model.py and stored in the artifact)",
            "healthy": "everything else",
        },
        "metrics": agent.metrics,
    }
