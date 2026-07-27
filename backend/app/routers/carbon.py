from fastapi import APIRouter, HTTPException
from app.schemas import CarbonRequest, CarbonResponse
from app.agents.carbon_agent import compute_savings, GRID_EMISSION_FACTOR_TREND, VEHICLE_ENERGY_PROFILES

router = APIRouter(prefix="/carbon", tags=["Net Zero Carbon Tracker"])


@router.post("/compute-savings", response_model=CarbonResponse)
def carbon_savings(req: CarbonRequest):
    try:
        result = compute_savings(req.vehicle_model, req.daily_distance_km)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CarbonResponse(**result)


@router.get("/metadata")
def carbon_metadata():
    return {
        "grid_emission_factor_trend": GRID_EMISSION_FACTOR_TREND,
        "vehicle_models": list(VEHICLE_ENERGY_PROFILES.keys()),
    }
