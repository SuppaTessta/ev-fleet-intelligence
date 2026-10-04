from fastapi import APIRouter, HTTPException

from app.agents.maintenance_agent import (
    InfeasibleScheduleError,
    MaintenanceJob,
    UnknownPriorityError,
    run_comparison,
)
from app.schemas import MaintenanceScheduleRequest, MaintenanceScheduleResponse

router = APIRouter(prefix="/maintenance", tags=["Maintenance Operations Optimiser"])


@router.post("/schedule", response_model=MaintenanceScheduleResponse)
def schedule(req: MaintenanceScheduleRequest):
    """Schedule a maintenance backlog against real workshop capacity.

    Returns every scheduler side by side, including the ones that lose. Read
    `improvement` with the docstrings in maintenance_agent: it compares the
    hours-based LP against an hours-aware greedy, and it is currently 0.
    """
    # The queue cap lives in MaintenanceScheduleRequest, not here: pydantic
    # validates before the handler body runs, so a second check in this function
    # is unreachable.
    try:
        jobs = [MaintenanceJob(j.asset_id, j.priority, j.reason) for j in req.jobs]
    except UnknownPriorityError as e:
        # a bad priority string is the caller's problem, not a server fault
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        result = run_comparison(
            jobs, n_bays=req.n_bays, technicians=req.technicians,
            hours_per_shift=req.hours_per_shift, shifts_per_day=req.shifts_per_day)
    except InfeasibleScheduleError as e:
        raise HTTPException(
            status_code=422,
            detail=f"no feasible schedule for this queue and capacity: {e}") from e

    return MaintenanceScheduleResponse(**result)
