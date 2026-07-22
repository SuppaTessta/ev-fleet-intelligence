from fastapi import APIRouter
from app.schemas import FleetReadinessRequest, FleetReadinessResponse
from app.agents.fleet_readiness_agent import score_vehicle

router = APIRouter(prefix="/fleet-readiness", tags=["Fleet Electrification Readiness Agent"])


@router.post("/score", response_model=FleetReadinessResponse)
def score(req: FleetReadinessRequest):
    result = score_vehicle(req.daily_distance_km, req.avg_payload_kg, req.dwell_time_hours)
    return FleetReadinessResponse(
        vehicle_id=req.vehicle_id,
        readiness=result["readiness"],
        recommended_model=result["recommended_model"],
        confidence_pct=result["confidence_pct"],
        price_inr_lakh=result["price_inr_lakh"],
        reason=result["reason"],
    )
