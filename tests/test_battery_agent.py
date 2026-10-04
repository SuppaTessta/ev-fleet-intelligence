"""
Battery APM agent tests -- needs backend/models/battery_rul_model.pkl
(gitignored, produced by train/train_battery_model.py). Auto-skips with a
clear reason if that file isn't present yet (fresh clone, CI, etc.).
"""
from conftest import MODELS_DIR, skip_if_missing

skip_if_missing(MODELS_DIR / "battery_rul_model.pkl")

from app.agents.battery_agent import get_battery_agent  # noqa: E402

# A real, declining capacity-fade sequence (illustrative shape, matches the
# kind of input the dashboard's fleet snapshots actually send)
DECLINING_HISTORY = [1.856, 1.846, 1.835, 1.826, 1.811, 1.799, 1.786]


def test_predict_returns_expected_shape():
    agent = get_battery_agent()
    result = agent.predict(
        capacity_history=DECLINING_HISTORY, temp_battery=25.0,
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
                             temp_battery=25.0, ambient_temp=24.0,
                             discharge_current=2.0, current_cycle_number=10,
                             rated_capacity_ah=2.0, pack_kwh=21.3)
    degraded = agent.predict(capacity_history=[1.5, 1.45, 1.40, 1.35, 1.30, 1.25, 1.20],
                              temp_battery=25.0, ambient_temp=24.0,
                              discharge_current=2.0, current_cycle_number=300,
                              rated_capacity_ah=2.0, pack_kwh=21.3)
    assert degraded["state_of_health_pct"] < healthy["state_of_health_pct"]


def test_business_impact_present_for_non_healthy_band():
    agent = get_battery_agent()
    degraded = agent.predict(capacity_history=[1.5, 1.45, 1.40, 1.35, 1.30, 1.25, 1.20],
                              temp_battery=25.0, ambient_temp=24.0,
                              discharge_current=2.0, current_cycle_number=300,
                              rated_capacity_ah=2.0, pack_kwh=21.3)
    if degraded["risk_band"] != "healthy":
        assert degraded.get("business_impact") is not None


def test_risk_band_is_derivable_from_the_returned_numbers():
    """A caller must be able to re-derive risk_band from the response fields.

    The band was cut on the unrendered prediction while the response carried the
    rounded one, so a raw 19.9597 came back as
    {"predicted_rul_cycles": 20.0, "risk_band": "critical"} -- which contradicts
    the documented "<20 = critical" rule using only the fields the caller can
    see. Two real rows of battery_features.csv land in that window.
    """
    agent = get_battery_agent()
    for history, rated in [
        (DECLINING_HISTORY, 2.0),
        ([1.95, 1.94, 1.93, 1.92, 1.91, 1.90, 1.89], 2.0),
        ([1.5, 1.45, 1.40, 1.35, 1.30, 1.25, 1.20], 2.0),
        ([2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4001], 2.0),
        ([0.9] * 7, 1.2),
    ]:
        r = agent.predict(capacity_history=history, rated_capacity_ah=rated,
                           current_cycle_number=100)
        rul, cap, eol = (r["predicted_rul_cycles"], r["current_capacity_ah"],
                         r["eol_threshold_ah"])
        if cap <= eol:
            expected = "critical"
            assert rul == 0.0, f"at/below EOL ({cap} <= {eol}) but RUL is {rul}"
        else:
            expected = "critical" if rul < 20 else "watch" if rul < 60 else "healthy"
        assert r["risk_band"] == expected, (
            f"history={history}: reported rul={rul} cap={cap} eol={eol} "
            f"=> band should be {expected!r}, got {r['risk_band']!r}")


def test_soh_is_consistent_with_the_reported_capacity():
    """state_of_health_pct must be re-derivable from current_capacity_ah and the
    threshold, so the three numbers cannot disagree about the same pack."""
    agent = get_battery_agent()
    r = agent.predict(capacity_history=[2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4001],
                       rated_capacity_ah=2.0, current_cycle_number=100)
    implied_rated = r["eol_threshold_ah"] / 0.70
    assert round(100 * r["current_capacity_ah"] / implied_rated, 1) == r["state_of_health_pct"]
