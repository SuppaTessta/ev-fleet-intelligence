"""Battery Asset Performance Management: remaining useful life from capacity fade.

Wraps the LightGBM model trained in train/train_battery_model.py over 19 NASA
PCoE cells across 5 temperature/current conditions. Given a short window of
recent discharge-cycle capacities plus operating conditions, returns predicted
RUL in cycles, state of health, and a risk band for triage.

State of health and the end-of-life threshold are relative to each asset's own
early-life capacity, not a fixed Ah number, because pack capacities differ.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app import config
from app.business_impact import battery_business_impact
from app.constants import ROLL_WINDOW

MODEL_PATH = config.BATTERY_MODEL


class BatteryAgent:
    def __init__(self, model_path: Path = MODEL_PATH):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_cols = artifact["feature_cols"]
        self.eol_soh_fraction = artifact["eol_soh_fraction"]

        # Provenance travels inside the artifact so a running service can say
        # which model it loaded, and callers can read the accuracy rather than
        # hardcode it. Served by GET /battery/metadata. See ADR-0005 for why
        # `training_strategy` is the field that matters.
        self.training_strategy = artifact.get("training_strategy", "unknown")
        self.trained_on_cells = artifact.get("trained_on_cells", [])
        self.n_training_rows = artifact.get("n_training_rows")
        self.censored_cells_excluded = artifact.get("censored_cells_excluded", [])
        self.metrics = artifact.get("metrics", {})

    def _engineer_features(self, capacity_history: list[float], temp_battery: float,
                            ambient_temp: float, discharge_current: float,
                            current_cycle_number: int, rated_capacity_ah: float) -> dict:
        capacities = np.array(capacity_history, dtype=float)
        current_capacity = capacities[-1]
        initial_capacity = rated_capacity_ah
        window = capacities[-ROLL_WINDOW:]

        fade_rate = 0.0
        if len(capacities) > ROLL_WINDOW:
            fade_rate = (capacities[-1] - capacities[-1 - ROLL_WINDOW]) / ROLL_WINDOW

        return {
            "discharge_cycle_num": current_cycle_number,
            "capacity": current_capacity,
            "capacity_pct_initial": current_capacity / initial_capacity,
            "roll_mean_capacity": window.mean(),
            # ddof=1 matches pandas .rolling().std() in the training pipeline.
            # numpy defaults to ddof=0, which would scale this feature by a
            # constant sqrt(4/5) relative to what the model was fitted on.
            "roll_std_capacity": window.std(ddof=1) if len(window) > 1 else 0.0,
            "fade_rate": fade_rate,
            "cumulative_fade": initial_capacity - current_capacity,
            "temp_battery": temp_battery,
            "ambient_temp": ambient_temp,
            "discharge_current": discharge_current,
        }

    def predict(self, capacity_history: list[float], temp_battery: float = 25.0,
                ambient_temp: float = 24.0, discharge_current: float = 2.0,
                current_cycle_number: int = None, rated_capacity_ah: float = None,
                pack_kwh: float = None) -> dict:
        if len(capacity_history) < 1:
            raise ValueError("capacity_history must contain at least one reading")
        if current_cycle_number is None:
            current_cycle_number = len(capacity_history)  # fallback only — pass the real value when known
        if rated_capacity_ah is None:
            # fallback only — accurate SOH needs the asset's true early-life capacity,
            # not the max of whatever window happens to be sent (understates SOH for
            # any window that doesn't include the asset's actual peak)
            rated_capacity_ah = max(capacity_history[: min(5, len(capacity_history))])

        feats = self._engineer_features(
            capacity_history, temp_battery, ambient_temp, discharge_current,
            current_cycle_number, rated_capacity_ah)
        initial_capacity = rated_capacity_ah
        x = pd.DataFrame([{c: feats[c] for c in self.feature_cols}])

        # Round first, then decide. The model sees raw capacities -- rounding
        # its features would break train/serve parity -- but every threshold
        # below is evaluated on the value actually RETURNED, so a caller can
        # re-derive risk_band from the response. Cutting the band on the
        # unrounded prediction lets a raw 19.9597 come back as
        # {"predicted_rul_cycles": 20.0, "risk_band": "critical"}, which
        # contradicts the documented "<20 = critical" rule using only the fields
        # the caller can see.
        rul_pred = round(max(0.0, float(self.model.predict(x)[0])), 1)
        current_capacity = round(capacity_history[-1], 3)
        eol_threshold_ah = round(initial_capacity * self.eol_soh_fraction, 3)
        soh_pct = round(100 * current_capacity / initial_capacity, 1)

        # Hard business-rule override: a pack already at/below its own EOL
        # threshold is at end of life *by definition* — don't let model noise
        # contradict a known physical fact.
        if current_capacity <= eol_threshold_ah:
            rul_pred = 0.0
            risk_band = "critical"
        elif rul_pred < 20:
            risk_band = "critical"
        elif rul_pred < 60:
            risk_band = "watch"
        else:
            risk_band = "healthy"

        result = {
            "predicted_rul_cycles": rul_pred,
            "state_of_health_pct": soh_pct,
            "current_capacity_ah": current_capacity,
            "eol_threshold_ah": eol_threshold_ah,
            "risk_band": risk_band,
        }
        if pack_kwh is not None:
            result["business_impact"] = battery_business_impact(risk_band, rul_pred, pack_kwh)
        return result


_agent_instance = None


def get_battery_agent() -> BatteryAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = BatteryAgent()
    return _agent_instance
