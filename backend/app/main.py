"""
EV Fleet Intelligence Platform — API entrypoint.

Unifies seven agents behind one FastAPI service, all live:
  - /battery            Battery Asset Performance Management (RUL forecasting)
  - /quality             Manufacturing Quality Intelligence (defect classification)
  - /risk                    EV Supply Chain Risk & Traceability (anomaly detection)
  - /fleet-readiness   Fleet Electrification Readiness (expert-baseline scoring)
  - /fleet/compound-risk  Compound Risk correlation across the above three
  - /carbon              Net Zero Carbon Intelligence Tracker
  - /maintenance     Maintenance Operations Optimiser (LP scheduling)

Run with: uvicorn app.main:app --reload --port 8000
Docs at:  http://localhost:8000/docs
"""

from fastapi import FastAPI
from app.routers import battery, quality, risk, fleet_readiness, fleet_command, carbon, maintenance

app = FastAPI(
    title="EV Fleet Intelligence Platform",
    description="AI platform unifying battery asset performance, manufacturing quality, supply "
                 "chain risk, fleet electrification readiness, cross-agent compound risk, net-zero "
                 "carbon tracking, and maintenance scheduling optimization — built for ET AI "
                 "Hackathon 2.0, Problem Statement #3.",
    version="1.0.0",
)

app.include_router(battery.router)
app.include_router(quality.router)
app.include_router(risk.router)
app.include_router(fleet_readiness.router)
app.include_router(fleet_command.router)
app.include_router(carbon.router)
app.include_router(maintenance.router)


@app.get("/")
def root():
    return {
        "platform": "EV Fleet Intelligence",
        "agents_live": ["battery", "quality", "risk", "fleet_readiness", "compound_risk", "carbon", "maintenance"],
        "agents_pending": [],
        "docs": "/docs",
    }
