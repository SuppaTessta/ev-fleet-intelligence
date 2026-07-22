from fastapi import APIRouter, HTTPException
from app.schemas import BatteryRULRequest, BatteryRULResponse
from app.agents.battery_agent import get_battery_agent

router = APIRouter(prefix="/battery", tags=["Battery APM Agent"])


@router.post("/predict-rul", response_model=BatteryRULResponse)
def predict_rul(req: BatteryRULRequest):
    try:
        agent = get_battery_agent()
        result = agent.predict(
            capacity_history=req.capacity_history_ah,
            temp_battery=req.temp_battery_c,
            time_s=req.discharge_time_s,
            ambient_temp=req.ambient_temp_c,
            discharge_current=req.discharge_current_a,
            current_cycle_number=req.current_cycle_number,
            rated_capacity_ah=req.rated_capacity_ah,
            pack_kwh=req.pack_kwh,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return BatteryRULResponse(asset_id=req.asset_id, **result)
