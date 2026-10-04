from fastapi import APIRouter

from app.agents.fleet_readiness_agent import score_vehicle, score_vehicle_with_economics
from app.errors import DomainValidationError
from app.schemas import (
    FleetReadinessEconomicsRequest,
    FleetReadinessRequest,
    FleetReadinessResponse,
    ReadinessEconomicsResponse,
)

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


@router.post("/score-with-economics", response_model=ReadinessEconomicsResponse)
def score_with_economics(req: FleetReadinessEconomicsRequest):
    """Feasibility AND five-year total cost of ownership.

    /score answers "can an EV do this job?" from physical gates alone. This adds
    "and is it cheaper than the diesel it replaces?", which is the external
    criterion readiness scoring never had -- the synthetic fleet is labelled by
    calling the scorer itself, so "77% ready" restates the route mix.

    The two answers diverge: a Tata Ace EV is feasible on a 40 km/day route and
    loses Rs 150,505 over five years, breaking even only above ~68 km/day.
    """
    try:
        result = score_vehicle_with_economics(
            req.daily_distance_km, req.avg_payload_kg, req.dwell_time_hours,
            req.expected_cycles_remaining)
    except ValueError as e:
        raise DomainValidationError(str(e)) from e

    return ReadinessEconomicsResponse(vehicle_id=req.vehicle_id, **{
        k: result[k] for k in
        ("readiness", "verdict", "recommended_model", "confidence_pct",
         "price_inr_lakh", "reason", "economics")})
