from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base for every request model.

    `allow_inf_nan=False` is the important part. Pydantic v2 accepts non-finite
    floats by default and Python's json parser accepts the bare tokens
    `Infinity` and `NaN`, so `{"daily_distance_km": Infinity}` validated
    cleanly, flowed through the arithmetic, and came back HTTP 200 carrying
    `Infinity` in the body -- which is not legal JSON, so strict clients cannot
    even parse the reply. Non-finite input is now a 422 at the boundary.

    `extra="forbid"` catches typo'd field names instead of silently ignoring
    them and applying a default.
    """
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


# Reusable constrained scalars, so a quantity that cannot be negative is
# rejected at the boundary rather than deep in the arithmetic.
Distance = Annotated[float, Field(ge=0, le=2000, description="kilometres")]
Percentage = Annotated[float, Field(ge=0, le=100)]
Deviation = Annotated[float, Field(ge=-100, le=10000, description="percent deviation")]
UnitInterval = Annotated[float, Field(ge=0, le=1)]


class BatteryRULRequest(StrictModel):
    asset_id: str = Field(..., min_length=1, max_length=64,
                          json_schema_extra={"example": "EV-TRUCK-014"})
    capacity_history_ah: list[Annotated[float, Field(gt=0, le=1000)]] = Field(
        ..., min_length=1, max_length=10000,
        description="Recent discharge-cycle capacity readings in Ah, oldest first.",
        json_schema_extra={"example": [1.856, 1.846, 1.835, 1.826, 1.811, 1.799, 1.786]})
    temp_battery_c: Annotated[float, Field(ge=-60, le=150)] = Field(
        25.0, description="Most recent battery temperature reading, deg C")
    ambient_temp_c: Annotated[float, Field(ge=-60, le=80)] = Field(
        24.0, description="Typical ambient operating temperature, deg C")
    discharge_current_a: Annotated[float, Field(ge=0, le=1000)] = Field(
        2.0, description="Typical/nominal discharge current, Amps")
    current_cycle_number: Annotated[int, Field(ge=1, le=1_000_000)] | None = Field(
        None, description="Asset true current cycle count. If omitted, falls back to the "
                           "length of capacity_history_ah, which understates cycle count when "
                           "only a recent window is sent rather than full history.")
    rated_capacity_ah: Annotated[float, Field(gt=0, le=1000)] | None = Field(
        None, description="Known initial/rated capacity in Ah. If omitted, falls back to the "
                           "max of the first 5 readings, which understates SOH for any window "
                           "not containing the early-life peak.")
    pack_kwh: Annotated[float, Field(gt=0, le=10000)] | None = Field(
        None, description="Pack size in kWh, e.g. 21.3 for a Tata Ace EV. If provided, the "
                           "response includes a business_impact block. Omit to skip cost estimates.")


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


class LiveTelemetryReading(BaseModel):
    asset_id: str
    depot: str
    battery_id: str
    cycle_number: int
    total_cycles_recorded: int
    historical_data_exhausted: bool
    timestamp: str
    voltage_v: float
    current_a: float
    temp_battery_c: float
    predicted_rul_cycles: float
    state_of_health_pct: float
    current_capacity_ah: float
    eol_threshold_ah: float
    risk_band: str
    business_impact: BatteryBusinessImpact | None = None


class LiveFleetStatusResponse(BaseModel):
    readings: list[LiveTelemetryReading]


class QualityInspectionResponse(BaseModel):
    verdict: str
    confidence: float
    gradcam_overlay_png_base64: str


class SurfaceInspectionResponse(BaseModel):
    verdict: str = Field(..., description="One of: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches")
    confidence: float
    class_probabilities: dict[str, float] = Field(..., description="Softmax probability for all 6 defect classes")
    gradcam_overlay_png_base64: str


class ShipmentRiskRequest(StrictModel):
    shipment_id: str = Field(..., min_length=1, max_length=64,
                             json_schema_extra={"example": "LITH-SUP1-014"})
    lead_time_deviation_pct: Deviation = Field(..., description="% deviation from normal lead time")
    price_deviation_pct: Deviation = Field(..., description="% deviation from normal unit price")
    reject_rate_pct: Percentage = Field(..., description="Incoming inspection reject rate, %")
    days_since_last_audit: Annotated[float, Field(ge=0, le=36500)] = Field(...)
    single_sourced: Annotated[int, Field(ge=0, le=1)] = Field(
        ..., description="1 if no backup supplier exists for this material")
    geopolitical_risk: UnitInterval = Field(..., description="0-1 risk score for the region")
    supplier_concentration_pct: Percentage = Field(
        ..., description="This supplier share of total volume for the material, %")
    volume_deviation_pct: Deviation = Field(..., description="% deviation from normal order volume")
    # These two trends are NOT in the same units.
    #
    # reject_rate_trend is a delta of a PERCENTAGE (reject_rate_pct.diff(3)/3),
    # so it lives in percentage points; the observed range is [-9.4, +9.3].
    #
    # price_trend is a delta of an ABSOLUTE PRICE (unit_price.diff(3)/3). Unit
    # prices in this dataset run from 2.71 to 69,461, so its real range is
    # [-5344.97, +9457.59]. Giving it the same +/-100 bound rejects 379 of the
    # 1,000 shipments in the project's own scored dataset -- and a caller who
    # clips to the bound to get past validation gets a different band for the
    # same shipment. A request schema has to accept the distribution the model
    # was fitted on.
    reject_rate_trend: Annotated[float, Field(ge=-100, le=100)] = Field(
        0.0, description="Change in reject rate over the last 3 shipments, in PERCENTAGE "
                          "POINTS. 0 = flat/unknown.")
    price_trend: Annotated[float, Field(ge=-1_000_000, le=1_000_000)] = Field(
        0.0, description="Change in unit price over the last 3 shipments, as an ABSOLUTE "
                          "currency delta (not a percentage) -- unit prices in this dataset "
                          "span 2.71 to 69,461, so values in the thousands are normal. "
                          "0 = flat/unknown.")


class ShipmentRiskResponse(BaseModel):
    shipment_id: str
    risk_score: float
    flagged: bool
    risk_band: str
    reason: str


class FleetReadinessRequest(StrictModel):
    vehicle_id: str = Field(..., min_length=1, max_length=64,
                            json_schema_extra={"example": "ICE-014"})
    daily_distance_km: Distance = Field(...)
    avg_payload_kg: Annotated[float, Field(ge=0, le=100000)] = Field(...)
    dwell_time_hours: Annotated[float, Field(ge=0, le=24)] = Field(
        ..., description="Hours available at depot for charging")


class FleetReadinessResponse(BaseModel):
    vehicle_id: str
    readiness: str
    recommended_model: str | None
    confidence_pct: float
    price_inr_lakh: float | None
    reason: str


class CompoundRiskRequest(StrictModel):
    asset_id: str = Field(..., min_length=1, max_length=64,
                          json_schema_extra={"example": "EV-TRUCK-04"})
    battery_risk_band: str = Field(
        ..., min_length=1, max_length=32,
        description="Output from /battery/predict-rul: healthy | watch | critical. Anything "
                    "else is treated as unknown and contributes no risk signal.",
        json_schema_extra={"example": "watch"})


class CompoundRiskResponse(BaseModel):
    asset_id: str
    battery_risk_band: str
    cell_supplier_id: str
    supplier_risk_level: str
    supplier_flagged_pct: float
    quality_verdict: str
    active_signal_count: int
    unknown_signal_count: int = Field(
        0, description="How many of the three inputs were unavailable. Missing data "
                        "contributes 0 to compound_score -- it is not treated as a risk signal.")
    asset_known_to_bom: bool = Field(
        True, description="False when the asset is absent from the bill-of-materials "
                           "linkage, in which case supplier and quality are unknown.")
    compound_score: int
    priority: str = Field(
        ..., description="routine | single-signal priority | compound priority | insufficient data")


class CarbonRequest(StrictModel):
    vehicle_model: str = Field(..., min_length=1, max_length=64,
                               json_schema_extra={"example": "Tata Ace EV"})
    daily_distance_km: Distance = Field(...)


class CarbonResponse(BaseModel):
    """Scope 1 (diesel tailpipe, eliminated) vs Scope 2 (purchased electricity for
    charging). Scope 3 is out of boundary and reported only as a sensitivity."""
    vehicle_model: str
    ev_g_co2_per_km: float
    diesel_equivalent_g_co2_per_km: float
    savings_pct: float
    daily_co2_savings_kg: float
    annual_co2_savings_kg: float
    diesel_comparison_source: str
    grid_factor_used: float
    charger_efficiency_used: float = Field(
        ..., description="Grid-meter-to-battery efficiency. Scope 2 is defined on PURCHASED "
                          "electricity, so the ~10% lost in the EVSE and onboard charger counts. "
                          "Omitting it previously overstated the Tata Ace saving as 19.7%.")
    kwh_per_km_at_battery: float
    kwh_per_km_at_meter: float = Field(
        ..., description="What the grid is actually billed for. This is the figure multiplied "
                          "by the grid emission factor.")
    breakeven_grid_factor_tco2_per_mwh: float = Field(
        ..., description="Grid carbon intensity at which this EV stops beating its diesel "
                          "equivalent. India's FY2024-25 weighted average is 0.7097.")
    scope3_sensitivity: dict = Field(
        ..., description="Out-of-boundary sensitivity, shown so the Scope 2 headline cannot be "
                          "mistaken for a lifecycle result. Not the published number.")


class MaintenanceJobInput(StrictModel):
    asset_id: str = Field(..., min_length=1, max_length=64)
    priority: str = Field(..., min_length=1, max_length=32,
                          description="critical, watch, or routine. Anything else is a 422 "
                                      "(it used to reach a dict lookup and raise KeyError -> 500).")
    reason: str = Field("unspecified", max_length=500)


class MaintenanceScheduleRequest(StrictModel):
    # max_length is the only cap: pydantic validates before the handler body
    # runs, so a second check in the router would be unreachable. The reason
    # lives in the description, where /openapi.json and /docs show it.
    jobs: list[MaintenanceJobInput] = Field(
        ..., min_length=1, max_length=500,
        description="At most 500 jobs. The scheduler builds a MIP whose size grows with "
                    "jobs x planning horizon, so a larger queue must be split.")
    n_bays: Annotated[int, Field(ge=1, le=100)] = 3
    technicians: Annotated[int, Field(ge=1, le=500)] = 4
    hours_per_shift: Annotated[float, Field(gt=0, le=24)] = 8.0
    shifts_per_day: Annotated[int, Field(ge=1, le=3)] = 2


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
    solver_notes: str | None = None
    peak_day_hours: float | None = Field(
        None, description="Busiest day's total job hours. Compare against "
                           "daily_capacity_hours: if it exceeds the budget, this "
                           "schedule is not physically executable.")


class MaintenanceScheduleResponse(BaseModel):
    """All five schedulers are returned, including the two that lose and the two
    that produce infeasible schedules. The platform previously reported a 37.5%
    improvement from the LP; that was an artifact of a bug in its own greedy
    baseline, and publishing the full comparison is the correction."""
    daily_capacity_jobs: int
    daily_capacity_hours: float
    greedy: ScheduleResult = Field(
        ..., description="Hours-aware EDF. The honest baseline.")
    optimal: ScheduleResult = Field(
        ..., description="Hours-based lexicographic LP. The production scheduler.")
    greedy_original_overflow: ScheduleResult = Field(
        ..., description="The original greedy, whose overflow path sent already-late "
                          "jobs to the EARLIEST free slot and so inflated its own miss "
                          "count. Retained as evidence, not as a baseline.")
    greedy_jobcount: ScheduleResult = Field(
        ..., description="Repaired EDF on job-slot capacity. Overbooks the workshop.")
    optimal_jobcount: ScheduleResult = Field(
        ..., description="The original job-count LP. Also overbooks the workshop, and "
                          "ties the repaired greedy exactly (Moore-Hodgson).")
    improvement: int = Field(
        ..., description="Jobs saved by the LP over the hours-aware greedy. Measured "
                          "like-for-like; currently 0.")
    improvement_vs_original_baseline: int
    assumptions: str


class FleetReadinessEconomicsRequest(FleetReadinessRequest):
    expected_cycles_remaining: Annotated[float, Field(ge=0, le=100000)] | None = Field(
        None, description="Predicted RUL from /battery/predict-rul. If the pack will not last "
                           "the 5-year horizon, a replacement is added to the EV's TCO. This is "
                           "the one place the battery model changes a rupee figure.")


class ReadinessEconomicsResponse(BaseModel):
    """Feasibility and economics reported separately, because they diverge."""
    vehicle_id: str
    readiness: str = Field(..., description="ready | not_yet_viable — physical gates only")
    verdict: str = Field(
        ..., description="ready_and_economic | ready_but_uneconomic | not_yet_viable. "
                          "A vehicle can be perfectly capable and still not worth buying.")
    recommended_model: str | None
    confidence_pct: float
    price_inr_lakh: float | None
    reason: str
    economics: dict | None = Field(
        None, description="5-year TCO breakdown vs the diesel equivalent, with break-even "
                           "daily distance and diesel price. None when nothing is feasible.")
