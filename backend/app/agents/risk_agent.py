"""EV Supply Chain Risk & Traceability: anomaly detection over shipment records.

Wraps the trained Isolation Forest. Returns a risk score, a
healthy/watch/critical band, and the named risk pattern the shipment most
resembles, so the caller gets an actionable reason rather than a bare number.

The band uses two thresholds rather than the model's single binary flag. That
flag catches sudden spikes reliably and gradual drift poorly, so WATCH surfaces
the early signal from the continuous score instead of waiting for the flag --
see train/analyze_risk_lead_time.py.

Per-archetype recall of the binary flag, measured on the shipped 1,000-shipment
dataset (79 injected anomalies):

    price_spike_late_delivery    1.000  (14/14)
    stale_audit_volume_spike     1.000  (17/17)
    quality_drift                0.933  (14/15)
    concentration_geopolitical   0.462  (6/13)
    gradual_quality_decline      0.350  (7/20)

Two of five archetypes are caught reliably; the other three are not, which is
why the WATCH band exists.

The table above is a convenience copy. The authoritative values are computed by
train_risk_model.py, stored in the artifact, and served at GET /risk/metadata --
which is what the dashboard reads, so a stale comment here cannot reach a user.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app import config

MODEL_PATH = config.RISK_MODEL


class RiskAgent:
    def __init__(self, model_path: Path = MODEL_PATH):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_cols = artifact["feature_cols"]
        self.score_min = artifact["raw_score_min"]
        self.score_max = artifact["raw_score_max"]
        self.metrics = artifact.get("metrics", {})

        # The WATCH boundary lives in the artifact, not in this file and not in
        # a CSV read at startup. A decision boundary is part of the model: if it
        # can be recomputed from whatever data happens to be on the machine, the
        # same artifact serves different risk bands on different hosts.
        if "watch_threshold" not in artifact:
            raise KeyError(
                "risk_model.pkl predates the persisted watch_threshold. "
                "Re-run train/train_risk_model.py -- refusing to guess a "
                "decision boundary that determines who gets paged.")
        self.watch_threshold = float(artifact["watch_threshold"])

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
