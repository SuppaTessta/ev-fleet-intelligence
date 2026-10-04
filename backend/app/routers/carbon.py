from fastapi import APIRouter

from app.agents.carbon_agent import GRID_EMISSION_FACTOR_TREND, VEHICLE_ENERGY_PROFILES, compute_savings
from app.errors import DomainValidationError
from app.schemas import CarbonRequest, CarbonResponse

router = APIRouter(prefix="/carbon", tags=["Net Zero Carbon Tracker"])


@router.post("/compute-savings", response_model=CarbonResponse)
def carbon_savings(req: CarbonRequest):
    try:
        result = compute_savings(req.vehicle_model, req.daily_distance_km)
    except ValueError as e:
        # 422, not 400: the request is well-formed but names a model we do not
        # have an energy profile for. Handled centrally in app/errors.py.
        raise DomainValidationError(str(e)) from e
    return CarbonResponse(**result)


@router.get("/metadata")
def carbon_metadata():
    return {
        "grid_emission_factor_trend": GRID_EMISSION_FACTOR_TREND,
        "vehicle_models": list(VEHICLE_ENERGY_PROFILES.keys()),
    }
