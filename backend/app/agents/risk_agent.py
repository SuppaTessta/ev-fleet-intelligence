"""
EV Supply Chain Risk & Traceability Agent.

Wraps the trained Isolation Forest. Given a shipment's features, returns
a risk score, a healthy/watch/critical band, and which named risk pattern
it most resembles, so the caller gets an actionable reason, not just a
number.

risk_band uses two thresholds, not one binary flag — this is a direct,
operationalized result of the lead-time analysis (see
train/analyze_risk_lead_time.py): the "critical" threshold is tuned for
precision on sudden spikes (~100% recall there), but gradual drift shows
up as a rising score well before it clears that bar. WATCH surfaces that
early signal instead of waiting for CRITICAL, which is where the real
lead-time advantage comes from on slow-drift cases.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "risk_model.pkl"
DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "supply_chain_shipments.csv"


class RiskAgent:
    def __init__(self, model_path: Path = MODEL_PATH):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_cols = artifact["feature_cols"]
        self.score_min = artifact["raw_score_min"]
        self.score_max = artifact["raw_score_max"]
        # WATCH threshold = top 25% of the full training score distribution —
        # see train/analyze_risk_lead_time.py for why this, not just the flag
        try:
            all_shipments = pd.read_csv(DATA_PATH)
            self.watch_threshold = float(all_shipments["risk_score"].quantile(0.75))
        except FileNotFoundError:
            self.watch_threshold = 0.3  # fallback if run before training data exists

    def _explain(self, f: dict) -> str:
        """Simple rule-based reason matching the injected archetypes —
        makes the score interpretable instead of a bare number."""
        if f["price_deviation_pct"] > 30 and f["lead_time_deviation_pct"] > 60:
            return "price spike combined with late delivery — possible shortage or panic-buying"
        if f["reject_rate_pct"] > 10:
            return "incoming inspection reject rate far above baseline — quality drift"
        if f.get("reject_rate_trend", 0) > 1.0:
            return "reject rate climbing over recent shipments — early-stage quality drift"
        if f["supplier_concentration_pct"] > 70 and f["geopolitical_risk"] > 0.5:
            return "high dependence on a single supplier in a geopolitically exposed region"
        if f["days_since_last_audit"] > 150 and f["volume_deviation_pct"] > 100:
            return "order volume spiked from a supplier that hasn't been audited recently"
        return "no single dominant driver — flagged on overall feature combination"

    def score(self, shipment: dict) -> dict:
        shipment = {"reject_rate_trend": 0.0, "price_trend": 0.0, **shipment}  # trend defaults to "flat" if unknown
        missing = [c for c in self.feature_cols if c not in shipment]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        x = pd.DataFrame([{c: shipment[c] for c in self.feature_cols}])
        x_scaled = self.scaler.transform(x)

        raw_score = self.model.score_samples(x_scaled)[0]
        risk_score = float(np.clip(1 - (raw_score - self.score_min) / (self.score_max - self.score_min), 0, 1))
        flagged = bool(self.model.predict(x_scaled)[0] == -1)

        if flagged:
            risk_band = "critical"
        elif risk_score >= self.watch_threshold:
            risk_band = "watch"
        else:
            risk_band = "healthy"

        return {
            "risk_score": round(risk_score, 4),
            "flagged": flagged,
            "risk_band": risk_band,
            "reason": self._explain(shipment) if risk_band != "healthy" else "within normal range",
        }


_agent_instance = None


def get_risk_agent() -> RiskAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RiskAgent()
    return _agent_instance
