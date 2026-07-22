from pydantic import BaseModel, Field
from typing import List


class BatteryRULRequest(BaseModel):
    asset_id: str = Field(..., example="EV-TRUCK-014")
    capacity_history_ah: List[float] = Field(
        ..., min_items=1,
        example=[1.856, 1.846, 1.835, 1.826, 1.811, 1.799, 1.786],
        description="Recent discharge-cycle capacity readings in Ah, oldest first."
    )
    temp_battery_c: float = Field(25.0, description="Most recent battery temperature reading, deg C")
    discharge_time_s: float = Field(3000.0, description="Most recent discharge duration, seconds")
    ambient_temp_c: float = Field(24.0, description="Typical ambient operating temperature, deg C")
    discharge_current_a: float = Field(2.0, description="Typical/nominal discharge current, Amps")
    current_cycle_number: int = Field(
        None, description="Asset's true current cycle count. If omitted, falls back to the "
                           "length of capacity_history_ah, which understates cycle count for "
                           "any asset where you're only sending a recent window, not full history.")
    rated_capacity_ah: float = Field(
        None, description="Asset's known initial/rated capacity in Ah. If omitted, falls back to "
                           "the max of capacity_history_ah, which understates SOH for any window "
                           "that doesn't include the asset's true early-life peak.")
    pack_kwh: float = Field(
        None, description="Asset's actual pack size in kWh, e.g. 21.3 for a Tata Ace EV. "
                           "If provided, the response includes a business_impact block "
                           "(replacement cost, urgency framing). Omit to skip cost estimates.")


class BatteryBusinessImpact(BaseModel):
    planned_replacement_cost_inr: int
    unplanned_failure_cost_inr: int
    avoidable_cost_inr: int
    urgency: str
    assumption: str


class BatteryRULResponse(BaseModel):
    asset_id: str
    predicted_rul_cycles: float
    state_of_health_pct: float
    current_capacity_ah: float
    eol_threshold_ah: float
    risk_band: str
    business_impact: BatteryBusinessImpact | None = None


class QualityInspectionResponse(BaseModel):
    verdict: str
    confidence: float
    gradcam_overlay_png_base64: str


class SurfaceInspectionResponse(BaseModel):
    verdict: str = Field(..., description="One of: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches")
    confidence: float
    class_probabilities: dict[str, float] = Field(..., description="Softmax probability for all 6 defect classes")
    gradcam_overlay_png_base64: str


class ShipmentRiskRequest(BaseModel):
    shipment_id: str = Field(..., example="LITH-SUP1-014")
    lead_time_deviation_pct: float = Field(..., example=5.0, description="% deviation from this supplier's normal lead time")
    price_deviation_pct: float = Field(..., example=2.0, description="% deviation from this supplier's normal unit price")
    reject_rate_pct: float = Field(..., example=1.8, description="Incoming inspection reject rate, %")
    days_since_last_audit: float = Field(..., example=40.0)
    single_sourced: int = Field(..., example=0, description="1 if no backup supplier exists for this material")
    geopolitical_risk: float = Field(..., example=0.2, description="0-1 risk score for the supplier's region")
    supplier_concentration_pct: float = Field(..., example=25.0, description="This supplier's % share of total volume for the material")
    volume_deviation_pct: float = Field(..., example=5.0, description="% deviation from this supplier's normal order volume")
    reject_rate_trend: float = Field(0.0, description="Change in reject rate over the last 3 shipments from this supplier, if known. 0 = flat/unknown.")
    price_trend: float = Field(0.0, description="Change in unit price over the last 3 shipments from this supplier, if known. 0 = flat/unknown.")


class ShipmentRiskResponse(BaseModel):
    shipment_id: str
    risk_score: float
    flagged: bool
    risk_band: str
    reason: str


class FleetReadinessRequest(BaseModel):
    vehicle_id: str = Field(..., example="ICE-014")
    daily_distance_km: float = Field(..., example=45.0)
    avg_payload_kg: float = Field(..., example=350.0)
    dwell_time_hours: float = Field(..., example=10.0, description="Hours available at depot for charging")


class FleetReadinessResponse(BaseModel):
    vehicle_id: str
    readiness: str
    recommended_model: str | None
    confidence_pct: float
    price_inr_lakh: float | None
    reason: str


class CompoundRiskRequest(BaseModel):
    asset_id: str = Field(..., example="EV-TRUCK-04")
    battery_risk_band: str = Field(..., example="watch", description="Output from /battery/predict-rul")


class CompoundRiskResponse(BaseModel):
    asset_id: str
    battery_risk_band: str
    cell_supplier_id: str
    supplier_risk_level: str
    supplier_flagged_pct: float
    quality_verdict: str
    active_signal_count: int
    compound_score: int
    priority: str


class CarbonRequest(BaseModel):
    vehicle_model: str = Field(..., example="Tata Ace EV")
    daily_distance_km: float = Field(..., example=60.0)


class CarbonResponse(BaseModel):
    vehicle_model: str
    ev_g_co2_per_km: float
    diesel_equivalent_g_co2_per_km: float
    savings_pct: float
    daily_co2_savings_kg: float
    annual_co2_savings_kg: float
    diesel_comparison_source: str
    grid_factor_used: float


class MaintenanceJobInput(BaseModel):
    asset_id: str
    priority: str = Field(..., description="critical, watch, or routine")
    reason: str = "unspecified"


class MaintenanceScheduleRequest(BaseModel):
    jobs: list[MaintenanceJobInput]
    n_bays: int = Field(3, example=3)
    technicians: int = Field(4, example=4)
    hours_per_shift: float = Field(8.0, example=8.0)
    shifts_per_day: int = Field(2, example=2)


class ScheduleAssignment(BaseModel):
    asset_id: str
    priority: str
    scheduled_day: int
    deadline_day: int
    missed_deadline: bool


class ScheduleResult(BaseModel):
    method: str
    assignments: list[ScheduleAssignment]
    jobs_missing_deadline: int
    total_jobs: int
    solver_status: str | None = None


class MaintenanceScheduleResponse(BaseModel):
    daily_capacity_jobs: int
    greedy: ScheduleResult
    optimal: ScheduleResult
    improvement: int
    assumptions: str
