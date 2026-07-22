from fastapi import APIRouter, HTTPException
from app.schemas import ShipmentRiskRequest, ShipmentRiskResponse
from app.agents.risk_agent import get_risk_agent

router = APIRouter(prefix="/risk", tags=["Supply Chain Risk Agent"])


@router.post("/score-shipment", response_model=ShipmentRiskResponse)
def score_shipment(req: ShipmentRiskRequest):
    try:
        agent = get_risk_agent()
        result = agent.score(req.model_dump(exclude={"shipment_id"}))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ShipmentRiskResponse(shipment_id=req.shipment_id, **result)
