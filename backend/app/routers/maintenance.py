from fastapi import APIRouter
from app.schemas import MaintenanceScheduleRequest, MaintenanceScheduleResponse
from app.agents.maintenance_agent import MaintenanceJob, run_comparison

router = APIRouter(prefix="/maintenance", tags=["Maintenance Operations Optimiser"])


@router.post("/schedule", response_model=MaintenanceScheduleResponse)
def schedule(req: MaintenanceScheduleRequest):
    jobs = [MaintenanceJob(j.asset_id, j.priority, j.reason) for j in req.jobs]
    result = run_comparison(jobs, n_bays=req.n_bays, technicians=req.technicians,
                             hours_per_shift=req.hours_per_shift, shifts_per_day=req.shifts_per_day)
    return MaintenanceScheduleResponse(**result)
