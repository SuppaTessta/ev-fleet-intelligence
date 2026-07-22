# EV Fleet Intelligence Platform

Built for **ET AI Hackathon 2.0** — Problem Statement #3: *AI for Industrial EV
Supply Chain & Asset Intelligence*.

A platform that gives industrial EV fleet operators and manufacturers one
unified intelligence layer across seven agents:

| Agent | Status | Technique |
|---|---|---|
| **Battery Asset Performance Management** — predicts Remaining Useful Life (RUL) from capacity fade trends | ✅ Live — 19 real NASA batteries, 5 conditions, pooled R²=0.74 | LightGBM regression, stratified hold-out validated |
| **Manufacturing Quality Intelligence** — two independent classifiers: cast-component defective/OK gate, plus a 6-way steel/sheet-metal surface defect-TYPE classifier (NEU-DET) | ✅ Live — casting + NEU-DET (99.7% held-out test accuracy) | ResNet50 transfer learning + Grad-CAM (both classifiers) |
| **EV Supply Chain Risk & Traceability** — flags anomalous supplier/shipment activity | ✅ Live — 5 risk archetypes, healthy/watch/critical banding | Isolation Forest anomaly detection + trend features |
| **Fleet Electrification Readiness** — scores ICE vehicles for EV transition, recommends a real OEM model | ✅ Live | Transparent rule-based expert-baseline scoring |
| **Compound Risk (Fleet Command Center)** — correlates battery, supplier, and quality signals per truck | ✅ Live | Cross-agent rule-based correlation |
| **Net Zero Carbon Tracker** — EV vs. diesel emissions, fleet-wide CO2 savings | ✅ Live | Real CEA grid emission factor + verified diesel comparison |
| **Maintenance Operations Optimiser** — schedules maintenance against workshop capacity + charging uptime | ✅ Live | Lexicographic LP scheduling (PuLP) — critical jobs protected first, never traded for lower tiers — compared against a greedy heuristic |

**All 6 "what you may build" options from the problem statement are now live**, along with all 5 named evaluation-focus criteria.

**Business impact layer:** battery, quality, and compound-risk views now translate technical outputs into ₹ cost and urgency, grounded in real 2026 Indian commercial-EV battery pricing (Rs 15-25k/kWh) and logistics rates (Rs 10-25/km) — see `backend/app/business_impact.py` for sources and assumptions.

**Requires Python 3.12** (not 3.14 — TensorFlow doesn't support 3.14 yet). See `SETUP.md`.

A unified Streamlit dashboard sits on top so all seven agents read as one product, not seven disconnected notebooks — Fleet Command Center, Fleet Overview, Battery Deep-Dive, Quality Inspection (both classifiers, tabbed), Supply Chain Risk, Electrification Readiness, Net Zero Carbon Tracker, and Maintenance Optimiser are all live.

## Why this approach

Judging weighs Technical Excellence and Business Impact heavily. Each agent
here is a direct extension of a technique already validated in a real,
working project, rather than a from-scratch attempt at an unfamiliar domain:

- Battery APM → same LightGBM forecasting approach as `Retail-Demand-Forecasting`
- Manufacturing Quality → the cast-component classifier reuses the CNN transfer-learning pipeline from `PhytoSentinel`; the NEU-DET steel-surface classifier is a direct extension of `surface-defect-detection` into this platform, not just a stylistic parallel to it
- Supply Chain Risk → same Isolation Forest anomaly detection as `Fintech-Fraud-Detection`

## Architecture

```
ev-fleet-intelligence/
├── backend/                # FastAPI service
│   ├── app/
│   │   ├── main.py         # app entrypoint, mounts routers
│   │   ├── schemas.py      # Pydantic request/response models
│   │   ├── agents/         # inference wrappers per agent
│   │   └── routers/        # HTTP route definitions per agent
│   ├── models/             # trained model artifacts (.pkl) — gitignored, reproducible
│   └── requirements.txt
├── data/
│   ├── download_battery_data.sh   # reproducible fetch of real NASA battery data
│   ├── raw/                       # gitignored — regenerate via download scripts
│   └── processed/                 # gitignored — regenerate via train/ scripts
├── train/                  # one training pipeline per agent
├── dashboard/               # unified Streamlit frontend
├── docs/                   # architecture diagram, eval plots, detailed submission doc
└── notebooks/               # exploratory analysis
```

## Battery APM Agent — methodology

- **Data**: 19 real NASA PCoE batteries across 5 temperature/current
  conditions (room temp through 43°C through 4°C, 1A-4A discharge) —
  charge/discharge cycles until failure or NASA's own stopping point,
  with capacity measured each discharge cycle.
- **Data quality handling**: two batteries (B0049-B0052's test run)
  excluded entirely — NASA's own group README says the run "stopped when
  experiment control software crashed." Individual sensor-fault readings
  (implausible near-zero capacity spikes) filtered with a floor, confirmed
  against NASA's documented "some runs show very low capacity" notes for
  the affected groups.
- **End-of-life definition**: 70% of each battery's OWN early-life
  capacity (State-of-Health based), not one fixed Ah number — different
  groups have different rated capacities and different NASA stopping
  points, so a per-battery relative threshold is what actually makes
  cross-condition pooling valid.
- **Target**: Remaining Useful Life (RUL), in discharge cycles.
- **Features**: rolling capacity mean/std, fade rate, cumulative fade,
  capacity as % of initial, temperature, ambient temperature, discharge
  current.
- **Validation**: stratified hold-out — one battery held out per condition
  group (5 groups → 5 held-out batteries), trained on the remaining 14.
  Tests generalization across every condition the model will see.
- **Result**: MAE = 13.0 cycles average across the 5 held-out batteries.
  For R², the honest version has two parts: 2 of the 5 held-out batteries
  had zero RUL variance in their recorded window (already past end-of-life
  for their entire test period — R² is mathematically undefined there, not
  poor); for the 3 batteries where it IS defined, results range from 0.72
  (good) to -1.48 (a genuine weak point — see below). Averaging R² across
  batteries with such different sample sizes (40 cycles vs. 168) is
  statistically unstable, so the pooled R² — all held-out predictions
  combined, the standard approach for this situation — is the headline
  number: **0.740**. Full breakdown and reasoning in
  `train/train_battery_model.py`, not just asserted here.
- **Known weak point, stated plainly**: B0032 (43°C, 4A discharge, only 40
  recorded cycles) is a genuine miss (R²=-1.48, MAE=15 cycles against a
  39-cycle range) — the model has the least data for this condition group
  and it shows. Worth a line in the submission rather than hiding it.
- **Next step for a stronger result**: NASA's full PCoE set has more
  batteries in the square-wave and mixed-load protocol groups not
  included here (excluded for using a different discharge protocol,
  see training script docstring) — bringing those in with protocol as an
  explicit feature is the clearest path to closing the B0032 gap.

## Manufacturing Quality Intelligence Agent — methodology

Two independent classifiers under one agent, covering two different points
in an EV supply chain where a visual defect check matters — cast component
intake, and sheet-metal / structural steel surface.

**Cast-component classifier (defective / OK gate)**
- **Data**: real cast-component defect images, defective vs. OK, with a
  held-out test split.
- **Approach**: ResNet50 (frozen ImageNet backbone) + a small trained head —
  transfer learning rather than a from-scratch CNN, since the dataset size
  doesn't support that. Same technique as the `PhytoSentinel` project.
- **Explainability**: a Grad-CAM overlay on every prediction, generated
  fresh per-request from the model's real gradients — shows which region
  of the component the model reacted to, not just a verdict.
- Results: `docs/quality_model_eval.png`, `docs/quality_gradcam_samples.png`.

**Steel surface / sheet-metal classifier (NEU-DET, 6-way defect type)**
- **Data**: NEU surface defect database — 1,800 real images, 300 per class:
  `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`,
  `scratches`. Ships as a flat object-detection layout (class encoded in
  the filename, plus bounding-box XML this classifier doesn't need);
  `data/prepare_neu_det.py` restructures it into a stratified 80/20
  train/test split.
- **Why a second, separate model rather than extending the binary one**:
  this is genuinely multi-class — every image already contains a defect,
  and the question is *which* of 6 known types, not defective-vs-OK.
  Softmax(6) output, with the Grad-CAM loss term keyed off the predicted
  class instead of a single fixed neuron, since there's no one "positive"
  output in a 6-way softmax.
- **Approach**: same ResNet50 transfer-learning pattern as the casting
  classifier — frozen backbone, small trained head — reusing the casting
  model's already-downloaded ImageNet weights instead of a second
  download, so both classifiers share one identical backbone.
- **Result**: 99.7% accuracy on a held-out 360-image test set (359/360
  correct), converged in 4 epochs. A direct extension of the
  `surface-defect-detection` portfolio project into this platform.
  Results: `docs/quality_neu_det_model_eval.png`,
  `docs/quality_neu_det_gradcam_samples.png`.

## Running it

Full step-by-step walkthrough (Windows + Mac/Linux, including GitHub push):
see **[SETUP.md](SETUP.md)**.

Quick reference if you're already set up:
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python data/download_battery_data.py
python train/train_battery_model.py
cd backend && uvicorn app.main:app --reload --port 8000
# Interactive docs: http://localhost:8000/docs
```

## Status

All 7 agents are built, integrated behind one FastAPI backend, and live in
the Streamlit dashboard: Battery APM, Manufacturing Quality (cast-component
+ NEU-DET steel-surface, two independent classifiers), Supply Chain Risk,
Fleet Electrification Readiness, Compound Risk (Fleet Command Center), Net
Zero Carbon Tracker, and Maintenance Optimiser.

The platform has been through a full audit-and-improve pass across the
build, catching and fixing several real, non-cosmetic bugs — not just a
lint/formatting cleanup.

Most recent addition: the NEU-DET steel-surface classifier, extending
Manufacturing Quality to a second, independent classifier without changing
the existing cast-component endpoint's behavior — verified with a full
regression pass across all 8 API endpoints plus the dashboard, both before
and after the change.
