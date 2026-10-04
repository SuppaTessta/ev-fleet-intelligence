from fastapi import APIRouter

from app.agents import telemetry_simulator
from app.agents.battery_agent import get_battery_agent
from app.errors import DomainValidationError, ModelUnavailableError
from app.schemas import BatteryRULRequest, BatteryRULResponse, LiveFleetStatusResponse

router = APIRouter(prefix="/battery", tags=["Battery APM Agent"])


@router.post("/predict-rul", response_model=BatteryRULResponse)
def predict_rul(req: BatteryRULRequest):
    # No blanket `except Exception -> 400`. A missing artifact is a 503 and a
    # bug is a 500; app/errors.py handles both centrally and logs the traceback
    # under a request id rather than returning it (str(e) on a FileNotFoundError
    # is an absolute filesystem path).
    try:
        agent = get_battery_agent()
    except (FileNotFoundError, OSError) as e:
        raise ModelUnavailableError(f"battery model could not be loaded: {e}") from e

    try:
        result = agent.predict(
            capacity_history=req.capacity_history_ah,
            temp_battery=req.temp_battery_c,
            ambient_temp=req.ambient_temp_c,
            discharge_current=req.discharge_current_a,
            current_cycle_number=req.current_cycle_number,
            rated_capacity_ah=req.rated_capacity_ah,
            pack_kwh=req.pack_kwh,
        )
    except ValueError as e:
        raise DomainValidationError(str(e)) from e

    return BatteryRULResponse(asset_id=req.asset_id, **result)


@router.get("/metadata", summary="Which model is loaded, and how well it scored")
def battery_metadata():
    """Training provenance and held-out metrics, read from the artifact itself.

    Mirrors /risk/metadata. `training_strategy` is the field that matters: it
    says whether the loaded model excludes right-censored rows (ADR-0005, and
    what every published figure is measured under) or treats their lower-bound
    labels as observations. Those are different models with different answers.
    """
    try:
        agent = get_battery_agent()
    except (FileNotFoundError, OSError) as e:
        raise ModelUnavailableError(f"battery model could not be loaded: {e}") from e
    return {
        "training_strategy": agent.training_strategy,
        "trained_on_cells": agent.trained_on_cells,
        "n_training_rows": agent.n_training_rows,
        "censored_cells_excluded": agent.censored_cells_excluded,
        "eol_soh_fraction": agent.eol_soh_fraction,
        "feature_cols": agent.feature_cols,
        "metrics": agent.metrics,
    }


@router.get("/live-fleet-status", response_model=LiveFleetStatusResponse)
def live_fleet_status():
    """Current simulated BMS reading for all 8 demo trucks. Read-only.

    Advancing is POST /battery/live-fleet-tick. Keeping this idempotent matters:
    a browser prefetch, an uptime probe or a second dashboard tab would
    otherwise consume simulation ticks that cannot be recovered without a reset.
    """
    return LiveFleetStatusResponse(readings=telemetry_simulator.current_fleet_readings())


@router.post("/live-fleet-tick", response_model=LiveFleetStatusResponse)
def live_fleet_tick():
    """Advances the simulated feed by one cycle and returns the fresh readings.

    See telemetry_simulator.py for what "live" means here: real historical NASA
    discharge cycles replayed in simulated time, not fabricated sensor values.
    """
    return LiveFleetStatusResponse(readings=telemetry_simulator.advance_fleet_tick())


@router.post("/live-fleet-reset")
def live_fleet_reset():
    """Resets every truck back to its original starting cycle, for a fresh demo run."""
    telemetry_simulator.reset_fleet()
    return {"status": "reset"}
