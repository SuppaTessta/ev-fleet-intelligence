"""
Operational surface: health, readiness, error mapping, and input hardening.

These need no trained artifacts, so unlike most of the suite they run in CI.

Each test corresponds to a specific failure the platform used to have:
  - a missing model artifact was reported as HTTP 400 "you sent bad data",
    while `/` returned 200 with a hardcoded list of all seven agents as live
  - exception strings (absolute filesystem paths) were returned to callers
  - `Infinity` and `NaN` validated cleanly and came back inside a 200 response,
    producing a body that is not legal JSON
  - a GET advanced shared simulation state
"""
import json

import pytest
from conftest import PROJECT_ROOT

# Machine-independent markers for "the response leaked a filesystem path".
# The repository directory name is derived at runtime rather than hardcoded, so
# the check keeps working -- and keeps failing for the right reason -- in a
# clone with a different name.
PATH_LEAK_MARKERS = (PROJECT_ROOT.name, str(PROJECT_ROOT), ".pkl", ".keras",
                     "Traceback", "site-packages")


def _post_raw(client, path, raw: bytes):
    """Bypasses the JSON encoder so non-finite literals actually reach the API.
    httpx refuses to serialise float('inf'), so this is the only way to send it."""
    return client.post(path, content=raw, headers={"content-type": "application/json"})


# ---------------------------------------------------------------- probes

def test_health_is_cheap_and_always_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_reports_artifacts_truthfully(client):
    resp = client.get("/ready")
    assert resp.status_code in (200, 503)
    body = resp.json()
    assert set(body) >= {"ready", "artifacts", "missing", "agents_available",
                         "agents_degraded", "missing_runtime_dependencies"}
    # The contract that matters: ready==True iff nothing an agent needs is
    # absent. That is BOTH artifacts and libraries -- two 98 MB .keras files on
    # disk say nothing about whether this interpreter can load them, and
    # tensorflow-cpu has no wheel for Python 3.13+ or arm64 Linux. Asserting
    # only `not missing` let /ready report ready:true while /quality/inspect
    # returned 503, which is the same "200 while broken" shape the hardcoded
    # agents_live list used to produce.
    assert body["ready"] is (not body["missing"] and not body["missing_runtime_dependencies"])
    assert (resp.status_code == 200) is body["ready"]


def test_an_unloadable_library_degrades_its_agent_not_the_service(client):
    """A missing optional dependency must degrade one agent, not the process.

    TensorFlow was imported at module scope by the quality router, which
    app.main imports, so on an interpreter with no wheel for it uvicorn refused
    to start and ALL endpoints went down -- including /health and /ready, whose
    whole job is to report partial availability. Six of the seven agents never
    touch TensorFlow.
    """
    body = client.get("/ready").json()
    deps = body["missing_runtime_dependencies"]
    # /health must answer regardless
    assert client.get("/health").status_code == 200
    for dep, info in deps.items():
        assert info["agents"], f"{dep} is reported missing but blames no agent"
        for agent in info["agents"]:
            assert agent in body["agents_degraded"], (
                f"{agent} needs {dep}, which is missing, but is not reported degraded")
            assert agent not in body["agents_available"]
    # and the agents that need nothing missing are still served
    unaffected = set(body["agents_available"])
    assert {"carbon", "fleet_readiness", "maintenance"} <= unaffected


def test_root_reports_measured_availability_not_a_hardcoded_list(client):
    """`/` used to return agents_live: [all seven] unconditionally, so a deploy
    with zero model artifacts looked completely healthy."""
    body = client.get("/").json()
    assert "agents_live" not in body, "the hardcoded field should be gone"
    ready = client.get("/ready").json()
    assert body["agents_available"] == ready["agents_available"]
    assert body["ready"] == ready["ready"]


def test_every_response_carries_a_request_id(client):
    resp = client.get("/health")
    assert resp.headers.get("X-Request-ID"), "needed to correlate a report with the logs"


def test_supplied_request_id_is_echoed(client):
    resp = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert resp.headers["X-Request-ID"] == "abc123"


# ---------------------------------------------------------------- input hardening

@pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN"])
def test_non_finite_distance_is_rejected(client, literal):
    """These used to validate, compute, and return 200 with a body containing
    the bare token Infinity -- which strict JSON parsers reject outright."""
    resp = _post_raw(
        client, "/carbon/compute-savings",
        f'{{"vehicle_model": "Tata Ace EV", "daily_distance_km": {literal}}}'.encode())
    assert resp.status_code == 422
    json.loads(resp.text)  # the error body itself must be parseable JSON


def test_negative_distance_is_rejected(client):
    resp = client.post("/carbon/compute-savings",
                       json={"vehicle_model": "Tata Ace EV", "daily_distance_km": -500})
    assert resp.status_code == 422


def test_out_of_range_geopolitical_risk_is_rejected(client):
    """Documented as 0-1; previously accepted -500."""
    resp = client.post("/risk/score-shipment", json={
        "shipment_id": "T", "lead_time_deviation_pct": 5.0, "price_deviation_pct": 2.0,
        "reject_rate_pct": 1.8, "days_since_last_audit": 40.0, "single_sourced": 0,
        "geopolitical_risk": -500, "supplier_concentration_pct": 25.0,
        "volume_deviation_pct": 5.0})
    assert resp.status_code == 422


def test_unknown_field_is_rejected_rather_than_silently_ignored(client):
    resp = client.post("/carbon/compute-savings", json={
        "vehicle_model": "Tata Ace EV", "daily_distance_km": 60.0, "typo_field": 1})
    assert resp.status_code == 422


def test_unknown_vehicle_is_a_clean_422_with_no_stack_trace(client):
    resp = client.post("/carbon/compute-savings",
                       json={"vehicle_model": "Not A Real EV", "daily_distance_km": 60.0})
    assert resp.status_code == 422
    body = resp.json()
    assert "request_id" in body
    assert "Traceback" not in json.dumps(body)


def test_error_bodies_never_leak_filesystem_paths(client):
    """A blanket `except Exception as e: HTTPException(400, str(e))` hands the
    caller whatever the exception stringified to -- for a FileNotFoundError,
    the absolute path of the artifact that failed to load."""
    probes = [
        ("/carbon/compute-savings", {"vehicle_model": "nope", "daily_distance_km": 1.0}),
        ("/maintenance/schedule", {"jobs": [{"asset_id": "A", "priority": "urgent"}]}),
        ("/fleet-readiness/score", {"vehicle_id": "V", "daily_distance_km": -1,
                                     "avg_payload_kg": 1, "dwell_time_hours": 1}),
    ]
    for path, payload in probes:
        text = client.post(path, json=payload).text
        for leak in PATH_LEAK_MARKERS:
            assert leak not in text, f"{path} leaked {leak!r}"


def test_unknown_priority_is_422_not_500(client):
    resp = client.post("/maintenance/schedule",
                       json={"jobs": [{"asset_id": "EV-01", "priority": "urgent"}]})
    assert resp.status_code == 422


def test_oversized_job_queue_is_rejected_at_the_boundary(client):
    """422 from the schema cap, not 413.

    The router also raised a 413 with a domain explanation, but pydantic's
    max_length=500 fires before the handler body runs, so that branch was
    unreachable dead code and the caller only ever saw the generic 422. The
    duplicate guard is gone and the explanation now lives in the field
    description, where /openapi.json and /docs can show it. Renamed because the
    old name asserted a status the service never returned.
    """
    jobs = [{"asset_id": f"EV-{i}", "priority": "routine"} for i in range(5000)]
    assert client.post("/maintenance/schedule", json={"jobs": jobs}).status_code == 422
    spec = client.get("/openapi.json").json()
    field = spec["components"]["schemas"]["MaintenanceScheduleRequest"]["properties"]["jobs"]
    assert field["maxItems"] == 500
    assert "500" in field.get("description", ""), (
        "the reason for the cap must reach callers through the schema")


def test_a_500_still_carries_the_request_id_it_tells_you_to_quote():
    """The one response class where correlation matters was the one that lost it.

    FastAPI installs the catch-all Exception handler inside Starlette's
    ServerErrorMiddleware, which sits OUTSIDE the user middleware stack -- so by
    the time it built the 500, RequestContextMiddleware had already reset the
    ContextVar. The body read "Quote request_id - when reporting this" and no
    X-Request-ID header was set at all, even when the caller supplied one.

    Exercised on a throwaway app because the real service has no endpoint that
    reliably raises, and adding one to the shared app would leak into every
    other test using the `client` fixture.
    """
    import logging

    from app.errors import install_error_handlers
    from app.observability import RequestContextMiddleware
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)

    @app.get("/boom")
    def _boom():
        raise RuntimeError(r"C:\srv\secret-deploy\backend\models\battery_rul_model.pkl is cursed")

    logging.disable(logging.CRITICAL)   # the handler logs the traceback by design
    try:
        probe = TestClient(app, raise_server_exceptions=False)
        resp = probe.get("/boom", headers={"X-Request-ID": "MYID-123"})
        assert resp.status_code == 500
        assert resp.headers.get("X-Request-ID") == "MYID-123"
        body = resp.json()
        assert body["request_id"] == "MYID-123"
        assert "MYID-123" in body["detail"], "the body must quote a real id, not '-'"

        # and one is generated when the caller supplies none
        resp = probe.get("/boom")
        generated = resp.headers.get("X-Request-ID")
        assert generated and generated != "-"
        assert resp.json()["request_id"] == generated

        # the exception's own text must not escape
        for leak in (*PATH_LEAK_MARKERS, "secret-deploy", "cursed"):
            assert leak not in resp.text, f"500 body leaked {leak!r}"
    finally:
        logging.disable(logging.NOTSET)


def test_a_rejected_oversized_payload_is_not_echoed_back(client):
    """A 501-job queue rejected for being one over the limit came back with all
    501 jobs in the error body -- an error response larger than the request."""
    jobs = [{"asset_id": f"EV-{i}", "priority": "routine"} for i in range(501)]
    resp = client.post("/maintenance/schedule", json={"jobs": jobs})
    assert resp.status_code == 422
    assert len(resp.text) < 2000, (
        f"error body is {len(resp.text)} bytes for a one-over-the-limit rejection")
    assert "EV-500" not in resp.text, "the rejected payload was echoed back"


def test_a_pixel_bomb_is_413_not_500(client):
    """A tiny PNG declaring enormous dimensions must be refused, not crash.

    quality.py has a MAX_PIXELS guard written for exactly this, but PIL raises
    DecompressionBombError from Image.open itself once a header declares more
    than 2x its own MAX_IMAGE_PIXELS (~179 MP) -- above our 40 MP cap but
    reached FIRST -- and that exception inherits straight from Exception, so the
    except tuple never caught it. A 91-byte payload produced a 500.
    """
    import struct
    import zlib

    def _png(width: int, height: int) -> bytes:
        def chunk(tag: bytes, payload: bytes) -> bytes:
            return (struct.pack(">I", len(payload)) + tag + payload
                    + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                + chunk(b"IDAT", zlib.compress(b"\x00")) + chunk(b"IEND", b""))

    for width, height in ((60000, 60000), (10000, 10000)):
        bomb = _png(width, height)
        assert len(bomb) < 200, "the point is that the payload is tiny"
        for endpoint in ("/quality/inspect", "/quality/inspect-surface"):
            resp = client.post(endpoint, files={"file": ("bomb.png", bomb, "image/png")})
            assert resp.status_code == 413, (
                f"{endpoint} returned {resp.status_code} for a {width}x{height} "
                f"declaration in {len(bomb)} bytes")
            for leak in PATH_LEAK_MARKERS:
                assert leak not in resp.text
