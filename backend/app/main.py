"""EV Fleet Intelligence Platform -- API entrypoint.

Seven agents behind one FastAPI service:
  /battery              battery remaining useful life
  /quality              manufacturing defect classification
  /risk                 supply-chain anomaly detection
  /fleet-readiness      electrification readiness and 5-year TCO
  /fleet/compound-risk  cross-agent correlation over the above
  /carbon               EV vs diesel emissions
  /maintenance          workshop scheduling

Operational:
  /health   liveness. Process is up. Never touches disk.
  /ready    readiness. Which agents can actually answer, and why not.

    uvicorn app.main:app --reload --port 8000     # docs at /docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app import config
from app.errors import install_error_handlers
from app.observability import RequestContextMiddleware, configure_logging
from app.readiness import inventory
from app.routers import battery, carbon, fleet_command, fleet_readiness, maintenance, quality, risk

configure_logging()
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Report what is loadable, then load it, before the first request arrives."""
    _log_startup_inventory()
    _warm_agents()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="EV Fleet Intelligence Platform",
    description="Asset intelligence for industrial EV fleets and the quality-critical "
                "supply chains behind them: battery remaining-useful-life, manufacturing "
                "defect classification, supply-chain risk, electrification readiness, "
                "cross-agent compound risk, net-zero carbon tracking, and maintenance "
                "scheduling.",
    version=config.VERSION,
)

app.add_middleware(RequestContextMiddleware)
install_error_handlers(app)

for _router in (battery, quality, risk, fleet_readiness, fleet_command, carbon, maintenance):
    app.include_router(_router.router)


def _log_startup_inventory() -> None:
    """Say once, at boot, what is and isn't loadable, and name the cause."""
    inv = inventory(include_hashes=False)
    if inv["ready"]:
        log.info("startup: all artifacts present",
                 extra={"agents_available": inv["agents_available"]})
        return

    deps = sorted(inv["missing_runtime_dependencies"])
    causes = []
    if inv["missing"]:
        causes.append(f"missing artifacts {inv['missing']}")
    if deps:
        causes.append(f"missing packages {deps}")
    log.error(f"startup: {' and '.join(causes)} -- affected agents will return 503",
              extra={"missing": inv["missing"],
                     "agents_degraded": inv["agents_degraded"],
                     "missing_runtime_dependencies": deps})


def _warm_agents() -> None:
    """Load the artifact-backed models and exercise them once, at boot.

    The agents are lazy singletons, so the first caller otherwise pays ~7.8 s of
    joblib load and LightGBM import, and a further ~2.3 s for the first
    prediction. The dashboard's landing page opens with 16 model-backed calls
    against a 10 s per-call budget, so on a cold start it lost that race.

    It also matters to orchestration: the Dockerfile HEALTHCHECK polls /health
    and compose gates the dashboard on `service_healthy`. Uvicorn finishes
    lifespan startup before it serves, so paying the cost here is what makes
    that gate mean "can answer" rather than "process exists".

    The two ResNet50 classifiers are deliberately left lazy: 98 MB each, ~20 s,
    and optional on most deployments. Their first request stays slow, which is
    why api_client gives inference a 45 s budget against 10 s for reads.
    """
    import time

    from app.agents.battery_agent import get_battery_agent
    from app.agents.risk_agent import get_risk_agent

    def _warm_battery():
        get_battery_agent().predict(
            capacity_history=[1.0, 0.99, 0.98, 0.97, 0.96, 0.95],
            rated_capacity_ah=1.0, current_cycle_number=6)

    def _warm_risk():
        agent = get_risk_agent()
        agent.score(dict.fromkeys(agent.feature_cols, 0.0))

    for name, warm in (("battery", _warm_battery), ("risk", _warm_risk)):
        started = time.perf_counter()
        try:
            warm()
        except Exception as exc:
            # Never fatal. A missing artifact must still boot and degrade to 503
            # on its own routes rather than take the other six agents down.
            log.error("startup: could not preload agent",
                      extra={"agent": name, "error": str(exc)})
        else:
            log.info("startup: agent preloaded",
                     extra={"agent": name,
                            "load_ms": round((time.perf_counter() - started) * 1000, 1)})


@app.get("/health", tags=["ops"], summary="Liveness probe")
def health():
    """Is the process up? Does no I/O, so it stays cheap to poll and cannot fail
    for reasons unrelated to liveness."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"], summary="Readiness probe")
def ready():
    """Can each agent actually answer?

    503 when any required artifact is missing or any required library failed to
    import, with the specific cause, so an orchestrator gets a truthful answer.
    """
    inv = inventory()
    return JSONResponse(status_code=200 if inv["ready"] else 503, content=inv)


@app.get("/", tags=["ops"])
def root():
    inv = inventory(include_hashes=False)
    return {
        "platform": "EV Fleet Intelligence",
        "version": app.version,
        # measured from disk on every call, not a hardcoded list
        "agents_available": inv["agents_available"],
        "agents_degraded": inv["agents_degraded"],
        # why an agent is degraded, not just that it is: a missing artifact and
        # an unimportable library need different fixes
        "missing_artifacts": inv["missing"],
        "missing_runtime_dependencies": sorted(inv["missing_runtime_dependencies"]),
        "ready": inv["ready"],
        "docs": "/docs",
        "health": "/health",
        "readiness": "/ready",
    }
