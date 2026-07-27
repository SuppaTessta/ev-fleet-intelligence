"""
API-contract tests: HTTP status codes, request validation, and response
shape via FastAPI's TestClient -- complements the agent-level tests (which
check the actual prediction logic) rather than repeating them.

/carbon, /fleet-readiness, and /maintenance need no trained model or data
file at all, so their tests always run. Every other endpoint eventually
calls an agent that does need one -- those tests skip cleanly if it's
missing, same as the corresponding agent-level test file.
"""
from conftest import skip_if_missing, MODELS_DIR, DATA_PROCESSED_DIR

# ---------------------------------------------------------------- always run

def test_root_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_carbon_compute_savings(client):
    resp = client.post("/carbon/compute-savings", json={"vehicle_model": "Tata Ace EV", "daily_distance_km": 60.0})
    assert resp.status_code == 200
    assert "savings_pct" in resp.json()


def test_carbon_unknown_vehicle_is_a_400_not_a_500(client):
    resp = client.post("/carbon/compute-savings", json={"vehicle_model": "Not A Real EV", "daily_distance_km": 60.0})
    assert resp.status_code == 400


def test_fleet_readiness_score(client):
    resp = client.post("/fleet-readiness/score",
                        json={"vehicle_id": "TEST-01", "daily_distance_km": 45.0,
                              "avg_payload_kg": 350.0, "dwell_time_hours": 10.0})
    assert resp.status_code == 200
    assert resp.json()["readiness"] in ("ready", "not_yet_viable")


def test_maintenance_schedule(client):
    resp = client.post("/maintenance/schedule", json={
        "jobs": [{"asset_id": "EV-01", "priority": "critical", "reason": "test"}],
        "n_bays": 2, "technicians": 2, "hours_per_shift": 8.0, "shifts_per_day": 1,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "greedy" in body and "optimal" in body


def test_malformed_request_is_422_not_500(client):
    """Missing required fields should fail Pydantic validation cleanly, not
    reach the agent and blow up with an unhandled exception."""
    resp = client.post("/carbon/compute-savings", json={"vehicle_model": "Tata Ace EV"})  # missing daily_distance_km
    assert resp.status_code == 422


# ---------------------------------------------------------------- needs trained models/data

def test_battery_predict_rul(client):
    skip_if_missing(MODELS_DIR / "battery_rul_model.pkl")
    resp = client.post("/battery/predict-rul", json={
        "asset_id": "TEST-BATTERY", "capacity_history_ah": [1.856, 1.846, 1.835, 1.826, 1.811, 1.799, 1.786],
        "temp_battery_c": 25.0, "discharge_time_s": 3000.0, "ambient_temp_c": 24.0,
        "discharge_current_a": 2.0, "pack_kwh": 21.3,
    })
    assert resp.status_code == 200
    assert resp.json()["risk_band"] in ("healthy", "watch", "critical")


def test_risk_score_shipment(client):
    skip_if_missing(MODELS_DIR / "risk_model.pkl")
    resp = client.post("/risk/score-shipment", json={
        "shipment_id": "TEST-01", "lead_time_deviation_pct": 5.0, "price_deviation_pct": 2.0,
        "reject_rate_pct": 1.8, "days_since_last_audit": 40.0, "single_sourced": 0,
        "geopolitical_risk": 0.2, "supplier_concentration_pct": 25.0, "volume_deviation_pct": 5.0,
    })
    assert resp.status_code == 200
    assert "risk_score" in resp.json()


def test_compound_risk(client):
    skip_if_missing(DATA_PROCESSED_DIR / "supply_chain_shipments.csv")
    resp = client.post("/fleet/compound-risk", json={"asset_id": "EV-TRUCK-04", "battery_risk_band": "watch"})
    assert resp.status_code == 200
    assert resp.json()["priority"] in ("routine", "single-signal priority", "compound priority")


def test_live_fleet_status(client):
    skip_if_missing(DATA_PROCESSED_DIR / "battery_features.csv", MODELS_DIR / "battery_rul_model.pkl")
    resp = client.get("/battery/live-fleet-status")
    assert resp.status_code == 200
    assert len(resp.json()["readings"]) == 8
