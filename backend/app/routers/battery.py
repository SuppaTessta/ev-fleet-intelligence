from fastapi import APIRouter, HTTPException
from app.schemas import BatteryRULRequest, BatteryRULResponse, LiveFleetStatusResponse
from app.agents.battery_agent import get_battery_agent
from app.agents import telemetry_simulator

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


@router.get("/live-fleet-status", response_model=LiveFleetStatusResponse)
def live_fleet_status():
    """Advances the simulated live BMS feed by one tick and returns fresh
    readings + a live RUL re-prediction for all 8 demo trucks. Each call
    moves the simulation forward -- see telemetry_simulator.py docstring for
    what "live" means here (real historical cycles replayed in simulated
    time, not fabricated sensor values)."""
    try:
        readings = telemetry_simulator.advance_fleet_tick()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return LiveFleetStatusResponse(readings=readings)


@router.post("/live-fleet-reset")
def live_fleet_reset():
    """Resets every truck back to its original starting cycle, for a fresh demo run."""
    telemetry_simulator.reset_fleet()
    return {"status": "reset"}
