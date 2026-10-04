from fastapi import APIRouter

from app.agents.compound_risk_agent import compute_compound_risk
from app.schemas import CompoundRiskRequest, CompoundRiskResponse

router = APIRouter(prefix="/fleet", tags=["Compound Risk Agent"])


@router.post("/compound-risk", response_model=CompoundRiskResponse)
def compound_risk(req: CompoundRiskRequest):
    result = compute_compound_risk(req.asset_id, req.battery_risk_band)
    return CompoundRiskResponse(**result)
