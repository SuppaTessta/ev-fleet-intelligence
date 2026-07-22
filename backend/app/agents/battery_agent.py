"""
Battery Asset Performance Management (APM) Agent.

Wraps the trained LightGBM RUL model (now trained across 19 batteries / 5
temperature-current conditions — see train/train_battery_model.py). Given a
short window of recent discharge-cycle capacity readings plus the asset's
operating temperature and discharge current, returns:
  - predicted Remaining Useful Life (in cycles)
  - state of health (% of this asset's own initial capacity)
  - a risk band for fleet-ops triage

State of health and the end-of-life threshold are computed relative to
EACH asset's own early-life capacity (robust to a single noisy first
reading — see training script docstring for why), not a fixed Ah number,
since different battery packs have different rated capacities.
"""

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import joblib
from app.business_impact import battery_business_impact

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "battery_rul_model.pkl"
ROLL_WINDOW = 5


class BatteryAgent:
    def __init__(self, model_path: Path = MODEL_PATH):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_cols = artifact["feature_cols"]
        self.eol_soh_fraction = artifact["eol_soh_fraction"]

    def _engineer_features(self, capacity_history: List[float], temp_battery: float,
                            time_s: float, ambient_temp: float, discharge_current: float,
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
            "roll_std_capacity": window.std() if len(window) > 1 else 0.0,
            "fade_rate": fade_rate,
            "cumulative_fade": initial_capacity - current_capacity,
            "temp_battery": temp_battery,
            "ambient_temp": ambient_temp,
            "discharge_current": discharge_current,
        }

    def predict(self, capacity_history: List[float], temp_battery: float = 25.0, time_s: float = 3000.0,
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
            capacity_history, temp_battery, time_s, ambient_temp, discharge_current,
            current_cycle_number, rated_capacity_ah)
        initial_capacity = rated_capacity_ah
        x = pd.DataFrame([{c: feats[c] for c in self.feature_cols}])
        rul_pred = max(0.0, float(self.model.predict(x)[0]))

        current_capacity = capacity_history[-1]
        soh_pct = round(100 * current_capacity / initial_capacity, 1)
        eol_threshold_ah = round(initial_capacity * self.eol_soh_fraction, 3)

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
            "predicted_rul_cycles": round(rul_pred, 1),
            "state_of_health_pct": soh_pct,
            "current_capacity_ah": round(current_capacity, 3),
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
