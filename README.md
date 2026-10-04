# EV Fleet Intelligence Platform

Asset intelligence for industrial EV fleets and the quality-critical supply chains behind them.
Seven agents behind one FastAPI service and one dashboard, each trained on real public data, with
every published number regenerable from `train/`.

Originally built for the ET AI Hackathon 2026 (Problem Statement 3). Since rebuilt around
reproducible evaluation — which changed several of the headline results. Those changes are the
most interesting thing here, so they are documented rather than quietly absorbed:
see [What the rebuild changed](#what-the-rebuild-changed).

```bash
docker compose up -d          # dashboard on :8501, API and Swagger on :8000/docs
```

That is the whole install. Six of the seven agents answer immediately, because the artifacts they
need are small enough to ship with the repo (842 KB of CSVs, a 221 KB LightGBM model and a 2.2 MB
Isolation Forest).

The seventh will not. The two vision classifiers are 98 MB each, so a clone cannot carry them and
`/ready` returns **503 with `quality` degraded** until you train them yourself — roughly 30–60
minutes on CPU, covered in [`SETUP.md`](SETUP.md). That state is the point rather than an
embarrassment: `/ready` names exactly which agents can answer and which cannot, and the dashboard
says so on the page instead of rendering empty charts.

Without Docker you need **Python 3.12** — TensorFlow has no wheel for 3.13+ or arm64 Linux, and the
service deliberately still starts without it, reporting `missing_runtime_dependencies` rather than
failing to boot:

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate      # source .venv/bin/activate on macOS/Linux
pip install -r backend/requirements.txt
python -m uvicorn app.main:app --app-dir backend --port 8000     # terminal 1
python -m streamlit run dashboard/main.py                        # terminal 2
```

Beginner-friendly walkthrough: [`SETUP.md`](SETUP.md). Deployment checklist: [Deploying](#deploying).

---

## The agents

| Agent | What it does | Result | Data |
|---|---|---|---|
| **Battery APM** | Remaining useful life from capacity fade | MAE **15.9 ± 15.7 cycles** (median 3.6), pooled R² 0.467. Beats a fade-extrapolation baseline 5/5 folds | Real — 19 NASA PCoE cells |
| **Quality: casting** | Defective / OK gate on cast components | **97.3%**, defect recall 0.955, on a leakage-free part-level split | Real — 1,300 castings |
| **Quality: surface** | 6-way steel surface defect type | **99.7%** on the held-out NEU-DET split | Real — 1,800 images |
| **Supply Chain Risk** | Anomalous supplier/shipment activity | AUC-PR **0.807 ± 0.055** over 20 held-out splits (prevalence 0.079) | Synthetic |
| **Fleet Readiness** | Scores ICE vehicles for EV transition, and whether it pays | 77% feasible; **5-year TCO** gives the external criterion. Tata Ace EV breaks even at **68 km/day** | Synthetic fleet, sourced costs |
| **Compound Risk** | Correlates the above per truck | Rule-based | Illustrative linkage |
| **Net Zero Carbon** | EV vs. diesel emissions | **10.8–24.3%** Scope 2 savings, incl. charging loss | Real — CEA + IPCC factors |
| **Maintenance** | Schedules against workshop capacity | Lexicographic LP. **Ties a correct greedy 40/40** on unit-size jobs — see below | Deterministic |

Each agent's provenance — real / synthetic / illustrative — is shown in the dashboard next to its
numbers, not just here.

---

## What the rebuild changed

Four published results did not survive being checked. All four corrections are in the code, with
the evidence.

**The maintenance optimiser's 37.5% improvement was a bug in its own baseline.** The greedy
comparison sent already-late jobs to the *earliest* free slot, cannibalising capacity from jobs
that could still make their deadline. Reversing that one scan makes greedy tie the LP exactly, on
the published scenario and on 40/40 randomised ones. That is what Moore–Hodgson predicts for
unit-size jobs under a flat daily count — against that baseline the LP could never have won. The
hours-aware pair the dashboard actually compares is not unit-size, and there the LP does pull
ahead on a large mixed backlog, by one deadline in 240 jobs; the API returns `improvement` so the
page states whichever is true rather than asserting the tie. All five schedulers are returned, so
the negative result stays checkable.
→ [ADR-0001](docs/adr/0001-maintenance-lp-vs-greedy.md)

**The casting classifier's 99.44% was measured on a contaminated split.** The vendor's test set
shares **97.5% of its source parts** with training, because the split was made after augmentation.
Worse than an inflated number: the model trained on it scores **88.08%** on genuinely unseen
castings and rejects **28% of good parts**. Re-splitting the 1,300 originals *before* augmentation
gives an honest 97.3% — and better real-world performance.
→ [ADR-0003](docs/adr/0003-casting-split-leakage.md)

**The battery R² of 0.740 was scored against broken labels.** The end-of-life search started at
cycle 1, so one anomalously low break-in reading pinned `RUL ≡ 0` for a cell's entire life. It also
accepted crossings the cell later *recovered* from — B0033 crossed at cycle 138, spent 94% of its
remaining readings back above the threshold, and ended at 99.5% state of health. Requiring the
crossing to be sustained leaves **5 of 19 cells with a genuine EOL event**; the other 14 are
right-censored, including every cell the original evaluation held out.
→ [ADR-0005](docs/adr/0005-censored-rul-labels.md)

**The carbon saving was measured at the wrong boundary.** Scope 2 is *purchased* electricity, but
the agent used energy at the battery, omitting ~10% charging loss. Tata Ace: 19.7% → **10.8%**.

---

## Where it is weakest

Stated here rather than left to be discovered:

- **73.1% of battery rows are right-censored.** A censoring-aware objective is implemented and
  measured ([`backend/app/survival.py`](backend/app/survival.py)); exclusion still wins on median
  MAE, so it remains the default. The censoring-aware model wins only where a condition group has
  a single observed cell — see [ADR-0005](docs/adr/0005-censored-rul-labels.md). With 5 observed
  cells this comparison cannot support a switching policy.
- **Fleet Readiness still has no expert baseline.** The synthetic fleet is labelled by calling
  the scorer being evaluated, so "77% ready" is a property of the route mix. The 5-year TCO model
  ([`tco.py`](backend/app/agents/tco.py)) now supplies an *external* criterion — "is the EV
  cheaper than the diesel it replaces?" — and the two answers diverge: a Tata Ace EV is capable on
  a 40 km/day route and loses ₹150,505 over five years. That is a real second gate, but its input
  prices are sourced mid-range figures, not a specific operator's costs.
- **Supply Chain Risk is entirely synthetic.** No real multi-tier battery-material shipment
  dataset is public at this scale.
- **Compound Risk's truck→supplier linkage is a fixed table**, standing in for a bill-of-materials
  feed. The models it correlates are real; only the cross-linking is not.
- **Grad-CAM is coarse.** At 128×128 input the final conv layer is 4×4, so the overlays are
  attention, not defect localisation.
- **CI does not exercise the two vision models** (98 MB each, too large to commit). Their
  *preprocessing* is covered; their accuracy is not.
- **The vision agent needs TensorFlow, which has no wheel for every interpreter** (Python 3.13+,
  arm64 Linux). That used to stop the whole service booting, since `app.main` imports the quality
  router; the other six agents now start regardless and `/ready` reports
  `missing_runtime_dependencies: ["tensorflow"]` with `quality` degraded. Running on such a machine
  costs you the two classifiers, not the platform.
- **There is no authentication.** All 18 endpoints are open, by design: this runs as a local
  single-operator tool and adding auth would be scope no reviewer asked for. It becomes a real
  exposure the moment the service is containerised and reachable, so treat an auth layer as a
  prerequisite for any deployment beyond localhost.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
   real datasets ──▶│  train/          reproducible pipelines   │
   NASA · NEU-DET   │  evaluation/     leakage + metric checks   │
   casting · CEA    └────────────────┬─────────────────────────┘
                                     │ artifacts (+ thresholds, metrics)
                                     ▼
   ┌───────────────────────────────────────────────────────────┐
   │  backend/app                                              │
   │    agents/    battery · quality · risk · readiness         │
   │               carbon · maintenance · compound-risk         │
   │    routers/   one per agent, thin                          │
   │    errors · observability · readiness                      │
   │                                                            │
   │    /health  /ready   ← truthful artifact inventory         │
   └────────────────────────────┬──────────────────────────────┘
                                │ HTTP (typed client, one error path)
                                ▼
   ┌───────────────────────────────────────────────────────────┐
   │  dashboard/   main · api_client · theme · components       │
   │               pages_impl/  one module per section          │
   └───────────────────────────────────────────────────────────┘

   Compound Risk consumes Battery + Quality + Supply Chain outputs.
   Maintenance consumes Battery + Compound Risk. That wiring is the platform.
```

Details per subsystem: [`docs/`](docs/). Decisions and rejected alternatives: [`docs/adr/`](docs/adr/).

---

## Testing

```bash
python -m pytest tests/ -q          # 206 tests
python -m ruff check backend dashboard train data evaluation tests tools
```

Or `make test-ci` / `.\make.ps1 test-ci` to reproduce exactly what CI runs.

**202 of 206 tests run in CI** — measured by deleting the vision weights locally and running
against what CI actually has. Real LightGBM and Isolation Forest inference executes on every push.
Only the two ResNet50 classifiers skip.

Tests worth knowing about:

- `test_quality_parity.py` — asserts serving preprocessing is **bit-identical** to training. The
  bug it guards against cost defect recall 1.000 → 0.982 on the live endpoint while the eval
  script still reported 1.000.
- `test_ops_endpoints.py` — a missing artifact returns 503, not 400; error bodies never leak
  filesystem paths; `Infinity` and `NaN` are rejected rather than returned inside a 200.
- `test_dashboard_pages.py` — every section renders without raising **with the backend stopped**.
- `test_dashboard_theme.py` — recomputes WCAG contrast from the design tokens.
- `test_no_drift.py` — asserts the shipped `battery_rul_model.pkl` was trained under the censoring
  policy ADR-0005 decides on, with the hyperparameters the pipeline declares, and carries its own
  held-out metrics. It did not: the final fit used all 19 cells (the strategy the ADR rejects)
  while every published figure was measured with the 14 censored cells dropped, and it had lost
  `subsample_freq` so it was not bagged either. The served model was not the measured model, and
  nothing compared the two.
- `test_risk_agent.py` — asserts every one of the 1,000 shipments in the shipped dataset is
  acceptable to the API's own request schema, and that the endpoint reproduces the score stored for
  each flagged one. `price_trend` is an absolute currency delta but carried a ±100 percentage-point
  bound, so **379 of them were rejected with a 422** — including 35 of the 94 the dashboard renders
  as CRITICAL. Clipping to the bound to get past validation returned a *different band* for the
  same shipment.
- `test_telemetry_simulator.py` — asserts the ticking endpoint and the read-only one agree about the
  same state. `historical_data_exhausted` was computed before the position advanced, so one tick per
  truck reported `cycle_number == total_cycles_recorded` next to `exhausted: false`, and only the
  ticking path is what the live monitor polls.
- `test_tco.py` — asserts every rupee figure in a response reconciles with the others (components
  sum to the total, the saving equals the difference of the totals) and that the reported break-even
  distance actually is one where the EV wins. It reported 68.2 km/day against a true crossover of
  68.238, so the number contradicted its own field name.
- `test_fleet_readiness.py` — asserts a `not_yet_viable` reason never names a constraint the catalog
  clears. It claimed "no catalog option covers the required 138km range" while the Tata Ace's 154 km
  did, and never mentioned payload — the gate that actually excluded it.
- `evaluation/leakage_check.py` — fails the build if any image split shares a source part.

---

## Deploying

Two images, because the API and the dashboard fail independently: a missing model artifact should
degrade the API to 503 on some routes while the dashboard stays up and says so.

```bash
docker compose build
docker compose up -d
curl -s localhost:8000/ready | python -m json.tool   # 200 when every agent can answer
```

Dashboard on `:8501`, API and Swagger on `:8000/docs`.

**Model artifacts are not baked into the image.** Compose bind-mounts them read-only, so the image
stays reproducible and a retrain does not require a rebuild:

```yaml
volumes:
  - ./backend/models:/app/backend/models:ro
  - ./data/processed:/app/data/processed:ro
```

The four small artifacts are committed, so a clone has them already. The two vision classifiers are
98 MB each and are not; build them on the host (see [`SETUP.md`](SETUP.md)) or mount a volume that
already has them. Without them the API still starts and `/ready` returns 503 naming exactly what is
absent. Note that `train/` is deliberately **not** in the image — nothing inside a container can
mint an artifact it is then asked to trust.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `EV_FLEET_MODELS_DIR` | `backend/models` | Point the API at a mounted model volume without rebuilding |
| `EV_FLEET_DATA_DIR` | `data/processed` | Same, for the processed CSVs |
| `EV_FLEET_API_URL` | `http://127.0.0.1:8000` | Where the dashboard looks for the API. Compose sets this to `http://api:8000` |

### Before you ship

- [ ] `make lint` and `make test` green, on Python 3.12 with TensorFlow installed
- [ ] `python evaluation/leakage_check.py data/raw/casting_defect_clean --max-exact 0` passes, and
      the same for `data/raw/neu_det`
- [ ] `/ready` returns **200**, not 503 — anything less means an agent in the docs cannot answer
- [ ] `curl localhost:8000/battery/metadata` reports the `training_strategy` you expect. This is
      the field that says whether the served model matches the published numbers
- [ ] Artifact SHAs in `/ready` match the build you intended to ship
- [ ] `docker compose logs api | head` shows `startup: agent preloaded` for battery and risk, and
      no `startup: missing ...` line

### Rollback

Images are tagged `ev-fleet-intelligence-api:latest` and `-dashboard:latest`. There is no database
and no migration, so rollback is `docker compose down && git checkout <previous> && docker compose
up -d --build`. Model artifacts live outside the image, so a rollback of code does not roll back a
retrained model — if a retrain is the thing you are reverting, restore the previous `.pkl`/`.keras`
into the mounted directory and restart the API.

### Known limits before this faces anything but localhost

- **No authentication.** All 18 endpoints are open. This is a single-operator local tool; an auth
  layer is a prerequisite for exposing it further.
- **No rate limiting.** `/maintenance/schedule` can occupy a CPU for up to `SOLVER_TIME_LIMIT_S`
  per tier. The queue size and model dimensions are capped, but concurrency is not.
- **Single process, in-memory state.** The telemetry simulator keeps its position per process, so
  more than one replica will disagree about where the demo fleet is.

---

## Repository

```
backend/app/      FastAPI service — agents, routers, errors, observability, readiness
dashboard/        Streamlit UI — app shell, typed API client, theme, one module per page
train/            one pipeline per agent; every published number comes from here
evaluation/       leakage checks and metric harnesses
data/             dataset prep, including the leakage-free casting re-split
docs/             evaluation figures, ADRs
tests/            206 tests
```

Licence: [MIT](LICENSE). Changes: [CHANGELOG.md](CHANGELOG.md).
