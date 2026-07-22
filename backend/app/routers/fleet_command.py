from fastapi import APIRouter
from app.schemas import CompoundRiskRequest, CompoundRiskResponse
from app.agents.compound_risk_agent import compute_compound_risk

router = APIRouter(prefix="/fleet", tags=["Compound Risk Agent"])


@router.post("/compound-risk", response_model=CompoundRiskResponse)
def compound_risk(req: CompoundRiskRequest):
    result = compute_compound_risk(req.asset_id, req.battery_risk_band)
    return CompoundRiskResponse(**result)
