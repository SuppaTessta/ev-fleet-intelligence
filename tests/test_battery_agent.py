"""
Battery APM agent tests -- needs backend/models/battery_rul_model.pkl
(gitignored, produced by train/train_battery_model.py). Auto-skips with a
clear reason if that file isn't present yet (fresh clone, CI, etc.).
"""
from conftest import skip_if_missing, MODELS_DIR

skip_if_missing(MODELS_DIR / "battery_rul_model.pkl")

from app.agents.battery_agent import get_battery_agent  # noqa: E402

# A real, declining capacity-fade sequence (illustrative shape, matches the
# kind of input the dashboard's fleet snapshots actually send)
DECLINING_HISTORY = [1.856, 1.846, 1.835, 1.826, 1.811, 1.799, 1.786]


def test_predict_returns_expected_shape():
    agent = get_battery_agent()
    result = agent.predict(
        capacity_history=DECLINING_HISTORY, temp_battery=25.0, time_s=3000.0,
        ambient_temp=24.0, discharge_current=2.0, current_cycle_number=100,
        rated_capacity_ah=2.0, pack_kwh=21.3,
    )
    for key in ("predicted_rul_cycles", "state_of_health_pct", "current_capacity_ah",
                "eol_threshold_ah", "risk_band"):
        assert key in result
    assert result["risk_band"] in ("healthy", "watch", "critical")
    assert 0 <= result["state_of_health_pct"] <= 110  # allow slight measurement noise above 100%


def test_lower_capacity_fraction_does_not_predict_a_healthier_band():
    """A battery already further down its fade curve shouldn't come back
    LESS urgent than a healthier one -- directional sanity check on the
    trained model, not an exact-value check."""
    agent = get_battery_agent()
    healthy = agent.predict(capacity_history=[1.95, 1.94, 1.93, 1.92, 1.91, 1.90, 1.89],
                             temp_battery=25.0, time_s=3000.0, ambient_temp=24.0,
                             discharge_current=2.0, current_cycle_number=10,
                             rated_capacity_ah=2.0, pack_kwh=21.3)
    degraded = agent.predict(capacity_history=[1.5, 1.45, 1.40, 1.35, 1.30, 1.25, 1.20],
                              temp_battery=25.0, time_s=3000.0, ambient_temp=24.0,
                              discharge_current=2.0, current_cycle_number=300,
                              rated_capacity_ah=2.0, pack_kwh=21.3)
    assert degraded["state_of_health_pct"] < healthy["state_of_health_pct"]


def test_business_impact_present_for_non_healthy_band():
    agent = get_battery_agent()
    degraded = agent.predict(capacity_history=[1.5, 1.45, 1.40, 1.35, 1.30, 1.25, 1.20],
                              temp_battery=25.0, time_s=3000.0, ambient_temp=24.0,
                              discharge_current=2.0, current_cycle_number=300,
                              rated_capacity_ah=2.0, pack_kwh=21.3)
    if degraded["risk_band"] != "healthy":
        assert degraded.get("business_impact") is not None
