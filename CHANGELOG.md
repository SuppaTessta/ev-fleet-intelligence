# Changelog

## 2.2.0 — captions that track their own data

A pass over the running dashboard rather than the source. Every number on screen was checked
against the response that produced it, which found explanatory prose that had been written from one
observed run and then frozen as a literal.

### Corrected results

| Claim | Was | Is | Cause |
|---|---|---|---|
| TCO break-even caption | "loses ₹150,505 on a 40 km/day route" always | **reads the response** | Hardcoded from one run. Selecting a battery age printed that loss beneath a metric reading −₹533,905; above the break-even distance it printed "loses" beneath a green *cheaper* badge |
| Electrification readiness caption | "77%" hardcoded | **quotes `ready_pct`** | Computed two lines above and then restated as a literal |
| Carbon savings range | "10.8–24.3%" hardcoded | **min/max of the rendered cards** | Would drift silently on the next CEA grid-factor revision |
| Maintenance tie claim | "the two tie" asserted unconditionally | **follows `improvement`** | The LP does win by one deadline on a 240-job mixed backlog, where jobs are no longer unit-size |

### Correctness

- **Live RUL trend drew fractional refresh counts.** `tick` is a 1-based counter encoded `:Q` with
  no step, so a two-point series rendered ticks at 1.00, 1.05 … 2.00. Constrained to integers.
- **Capacity column mixed decimal widths** — a 1.570 Ah pack rendered `1.57 Ah` beside `1.384 Ah`,
  because the API rounds to 3 dp and bare interpolation drops the trailing zero.
- **`use_container_width` was past its removal date.** Deprecated upstream with a stated removal of
  2025-12-31 and still called at 14 sites; migrated to `width="stretch"` after confirming in
  Streamlit's source that both elements resolve the old argument to exactly that. The floor moved to
  `streamlit>=1.62` — the oldest release this was actually run against, rather than the oldest that
  might work.

### Packaging

- **The Docker path now works from a clone.** `train/` is not in the image, so nothing inside a
  container could mint a missing artifact, and `docker compose up` on a fresh clone started with
  four of seven agents despite the README promising no Python toolchain was needed. The four small
  artifacts (842 KB of CSVs, a 221 KB LightGBM model, a 2.2 MB Isolation Forest) are now committed.
  The two 98 MB vision classifiers still are not, and `/ready` still reports 503 until they exist.
- `requirements.lock` regenerated and verified: 21/21 endpoints green on the rebuilt image under
  pandas 3.0.6, numpy 2.5.3, fastapi 0.142.2 and tensorflow-cpu 2.21.0.

### Removed

- Two orphaned evaluation figures (`quality_*_leaky_split.png`) that nothing referenced.

## 2.1.0 — audit fixes

An audit pass over the 2.0.0 tree. The theme repeats 2.0.0's: the numbers were right and the
artifact was not.

### Corrected results

| Claim | Was | Is | Cause |
|---|---|---|---|
| Battery model actually served | evaluated as `exclude` | **trained as `exclude`** | The final fit ran on all 19 cells — the `naive` strategy [ADR-0005](docs/adr/0005-censored-rul-labels.md) rejects — while every published figure was measured with the 14 censored cells dropped |
| Battery pooled R² of the shipped artifact | 0.769 (unpublished) | **0.467 (published)** | Same |
| Dashboard battery MAE caption | 21.2 ± 12.4 over 6 cells | **read from `/battery/metadata`** | Hardcoded from a superseded run; the shipped model scores 15.9 ± 15.7 over 5 |

### Correctness

- **The deployed battery model was not the evaluated one.** `train_battery_model.py` fitted the
  serving artifact on the full frame, censored rows included, so `/battery/predict-rul` answered
  from a model taught that a healthy cell near the end of its recording is nearly dead — the exact
  failure ADR-0005 exists to prevent, and one `business_impact.py` prices in rupees. The censoring
  policy is now a named `TRAINING_STRATEGY`, applied to the final fit, and recorded inside the
  artifact.
- **`subsample_freq` was missing from the shipped model, again.** 2.0.0 fixed the silent-no-op in
  `LGBM_PARAMS`, but the final fit and `stratified_eval` each kept an inline copy of the
  hyperparameters that had not been updated. Verified on the committed `.pkl`: `subsample_freq=0`.
  All three now read `LGBM_PARAMS`.
- **`/maintenance/schedule` could be made to build a 12.5-million-variable MIP.** `MAX_JOBS` capped
  the queue but nothing capped the horizon, which sizes itself as `total_hours / (daily_hours −
  longest_job)`. 500 critical jobs at 1 bay / 1 technician / 5.1h drives that denominator to 0.08
  and the horizon to 25,003 days. Now bounded by `MAX_HORIZON_DAYS` and `MAX_MODEL_BINARIES`, with
  a 422 that names the cause.
- **A legitimate max-size queue returned "no feasible schedule".** At `MAX_JOBS` on the *default*
  workshop, CBC hits its time limit with no complete assignment and the LP raised — which the
  router reported as infeasibility, for a queue the greedy scheduler had just solved. Both LPs now
  fall back to their greedy witness and say so in `solver_status`.
- **`data/prepare_neu_det.py` could not read the NEU-DET dataset as distributed.** It looked for a
  flat `IMAGES/` folder; the copy shipped with this project is `NEU-DET/{train,validation}/images/
  <class>/`. Both layouts are now discovered, images are pooled and re-split at the same seed so
  the published 99.7% stays reproducible either way, and duplicate filenames across the source
  tree are a hard error rather than a re-split leak.
- **`data/prepare_casting.py` defaulted to an absolute path on one developer's D: drive.**
  Replaced with an ordered candidate search.
- **TCO break-even of exactly 0.0 was reported as "not computable"** (truthiness test on a float),
  and the dashboard interpolated a `None` break-even straight into its caption as the word "None".

### Found by actually running it

The static pass above missed all of these. They were found by starting the API and the dashboard,
walking every section, and independently recomputing each endpoint's output from the raw data.

- **The service could not start without TensorFlow at all.** The quality router imports it at
  module scope and `app.main` imports the quality router, so on any interpreter with no
  tensorflow-cpu wheel (3.13+, arm64 Linux) uvicorn refused to boot and all 18 endpoints went
  down -- `/health` and `/ready` included, whose entire job is to report partial availability.
  Six of the seven agents never touch TensorFlow. The import is now absorbed, reported through
  `/ready` as `missing_runtime_dependencies`, and raised at the point of use as a 503.
- **`/ready` called an agent available on the strength of files it could not load.** Both 98 MB
  `.keras` artifacts present + no TensorFlow reported `quality` as ready. A missing library is now
  as disqualifying as a missing file.
- **The first model-backed request took 7.82 s; `/health` answered in 0.14 s.** The agents are lazy
  singletons, so whoever arrived first paid the LightGBM import and `joblib.load` — and the
  dashboard's landing page opens with 16 model-backed calls against a 10 s per-call budget. On a
  first run, with Streamlit booting alongside it, it lost that race and the first thing a new user
  saw was "The backend took longer than 10s to respond." It also made the Docker `HEALTHCHECK` and
  compose's `condition: service_healthy` gate meaningless. Artifact-backed agents are now loaded
  *and given one throwaway prediction* during lifespan startup: **first request 7.82 s → 0.031 s**,
  full landing-page chain from cold **0.19 s**.
- **`price_trend` was bounded as a percentage when it is an absolute currency delta.** It carried
  `reject_rate_trend`'s ±100 percentage-point bound, but it is `unit_price.diff(3)/3` over prices
  spanning 2.71 to 69,461, with a real range of [−5,344.97, +9,457.59]. **379 of the 1,000
  shipments in the project's own scored dataset were rejected with 422** — 35 of the 94 the
  dashboard renders as CRITICAL, and 27 of the 79 true injected anomalies. Worse, clipping to the
  bound to get past validation changed the verdict: COBALT-SUP1-014 is stored at 0.4196/flagged
  and came back 0.2644/"watch". The same "the table and the API disagree about the same shipment"
  failure the persisted `watch_threshold` was introduced to fix, arriving through the schema.
- **The tick endpoint and the read-only endpoint disagreed about the same state.**
  `historical_data_exhausted` was computed from the pre-increment position, so the tick that
  arrives at a cell's last recorded cycle returned `cycle_number == total_cycles_recorded` next to
  `historical_data_exhausted: false`, while a `GET` against that same unchanged state answered
  `true`. The dashboard's live monitor polls only the ticking path, so it rendered the final real
  reading as live for one tick per truck.
- **The battery response could not be used to re-derive its own verdict.** `risk_band` was cut on
  the unrounded prediction while the response carried the rounded one, so a raw 19.9597 came back
  as `{"predicted_rul_cycles": 20.0, "risk_band": "critical"}` — contradicting the documented
  "<20 = critical" rule using only the fields a caller can see. Two real rows of
  `battery_features.csv` land in that window. Same for the EOL override, which could return
  `current_capacity_ah == eol_threshold_ah` beside a non-zero RUL. Thresholds are now evaluated on
  the values that are actually returned; the model still sees the raw ones.
- **Compound risk shared one score table across all three signals**, so the quality vocabulary
  leaked into the battery slot: `battery_risk_band="defective"` — a value `/battery/predict-rul`
  cannot emit, and one the field's own OpenAPI description rules out — scored 2 and pushed a
  healthy truck from `routine` to `single-signal priority`, which the dashboard's `priority_map`
  turns into a 10-day maintenance deadline instead of 30. An out-of-vocabulary input must never
  escalate. Each slot now has its own vocabulary.
- **`annual_co2_savings_kg` was derived from the already-2dp-rounded daily figure**, multiplying
  that rounding by 300: a Tata Ace at 1 km/day reported 3.0 kg/yr against an exact 3.9. Rounded
  once at the end.
- **Per-archetype recall was hardcoded prose and had gone stale.** `concentration_geopolitical` was
  published as 0.385 (5/13) in `risk_agent.py` and rendered to users as "0.39", while the shipped
  dataset it names as its source gives **0.462 (6/13)**. The other four reproduce exactly. Now
  computed by `train_risk_model.py`, stored in the artifact, and served at `/risk/metadata` — same
  fix as the battery MAE caption and the WATCH threshold before it.
- **The Battery Deep-Dive capacity chart's x axis was a 1-based window position labelled
  "Discharge cycle".** A truck at cycle 140 of a 168-cycle cell was drawn as cycles 1–10,
  contradicting the "Cycle n/N" figure a few lines above and reading as a nearly-new cell. It now
  plots the real cycle numbers, ordinally, so ticks stay whole cycles.
- **The sidebar reported every degradation as "Missing model artifacts".** With the artifacts on
  disk and the library missing, that pointed the reader at a retraining step that could not have
  helped. Cause is now named.
- **Adding workshop capacity could turn a served request into a refused one.** `compute_horizon_hours`
  estimated `usable = daily_hours - longest` and applied that worst-case stranding allowance to
  *every* day. At 4.8 h/day against a 4.0 h critical job that reads 0.8, so 142 h of work "needed"
  178 days when a first-fit packs it into 32 — and the new `MAX_HORIZON_DAYS` guard then refused it
  with a fabricated "181-day plan". The same 93-job queue was served at 5.0 h/shift, refused at 5.5
  and 6.0, then served again at 6.5. It now runs the packer instead of guessing: exact for the
  greedy, a valid bound for the LP, monotone in capacity, and the DoS case still refuses — with a
  truthful 503-day count rather than 25,003.
- **A 500 carried no `X-Request-ID` and told the caller to quote `-`.** FastAPI installs the
  catch-all handler inside Starlette's `ServerErrorMiddleware`, outside the user middleware stack,
  so by the time it built the response `RequestContextMiddleware` had reset the context var. The one
  response class where correlation is load-bearing was the only one that lost it. The middleware now
  owns that response, because it is the last layer that still knows the id.
- **The router's 413 for oversized queues was unreachable dead code.** The schema's
  `max_length=500` fires before the handler body, so callers got a bare 422 and the domain
  explanation never reached them — and the error echoed all 501 rejected jobs back, making the
  error larger than the request. One guard now, the reason lives in the field description where
  `/openapi.json` shows it, and oversized collections are summarised rather than echoed (**501-job
  rejection: whole payload → 254 bytes**).
- **The dashboard and the Live Fleet Monitor disagreed about the same truck.**
  `dashboard/fleet_data.predict_battery` omitted `temp_battery_c` / `ambient_temp_c` /
  `discharge_current_a`, so the request schema supplied 25.0 °C against a recorded 32.1–41.1 °C
  across the three demo cells — `temp_battery` is a real LightGBM feature, so the dashboard was
  scoring its own fleet out of distribution. **6 of 8 trucks came back with a different RUL**
  (EV-TRUCK-07: 64.4 vs 79.1) and **38 of 504 reachable simulator states got a different risk
  band**, which propagates through `/fleet/compound-risk` into a different maintenance tier.
  Neither page flagged it, while `telemetry_simulator`'s docstring called the two "like-for-like".
- **The break-even fix above had a bug of its own.** `saving_at` is not monotonic once a pack
  replacement can fire, so with `expected_cycles_remaining=900` there are three sign changes, not
  one — and the bisection converged on the upper root. At 80 km/day the response reported
  `ev_cheaper: true`, a ₹62,690 saving and verdict `ready_and_economic` **while the break-even
  block beside it said the vehicle "never reaches the distance at which it would pay back"**. Now
  a coarse scan finds every crossing, `also_wins_between_km` reports the bounded winning stretch,
  and the note is checked against the distance actually asked about. The monotonic case (no
  replacement) keeps an exact single bisection: 0.18 ms vs 2.5 ms for the scan.
- **A pixel-bomb PNG returned 500, not 413.** `quality.py` has a `MAX_PIXELS` guard written for
  exactly this, but PIL raises `DecompressionBombError` from `Image.open` once a header declares
  more than 2× its own ~179 MP limit — above our 40 MP cap but reached first — and that exception
  inherits straight from `Exception`, so the `except` tuple never caught it. A **66-byte** payload
  crashed the endpoint.
- **The startup log said "missing artifacts" with `missing: []`** — `ready` is now false for a
  missing library too, so the operator's log carried the same misdiagnosis the sidebar had. It now
  names the cause: `startup: missing packages ['tensorflow']`.
- **Blocker text rounded the requirement to whole units, printing false inequalities.** At 123.3
  km/day it read "Tata Ace EV (154km range < 154km needed)" — arithmetically false as printed, with
  the true requirement 154.125 km. Now `154.1km needed`.
- **The shipped `fleet_readiness.csv` was not reproducible from its own columns.** The generator
  scored the unrounded draws and published rounded inputs, so `confidence_pct` differed on 8 of 60
  rows — and the CSV still carried the pre-fix blocker wording, including two rows asserting a
  charge-window blocker the catalog clears. Regenerated; all 60 rows now reproduce exactly.
- The `FLEET_LINKAGE` comment's supplier tie list omitted `COBALT-SUP1`. Four suppliers are tied at
  20.0%, not three — in the very comment added to correct an earlier false claim about the same
  supplier.

### Added

- `GET /battery/metadata` — training strategy, cells trained on, and held-out metrics, read from
  the artifact. Mirrors `/risk/metadata`; the dashboard now asks instead of hardcoding.
- `archetype_recall` on `GET /risk/metadata`, computed by `train_risk_model.py` and stored in the
  artifact, so the dashboard caption stops hardcoding a figure that had gone stale.
- `breakeven_is_reachable`, `max_feasible_daily_km` and `distance_breakeven_note` on the TCO
  response, so a break-even beyond the vehicle's own range gate is labelled rather than offered as
  advice.
- **The battery → TCO link now has a caller.** `expected_cycles_remaining` is documented as "the one
  place the RUL model changes a rupee figure that matters" and nothing passed it — the endpoint
  accepted it, `tco.py` implemented the replacement term, and the advertised wiring was dead in the
  UI. The TCO panel now offers a real truck's predicted RUL: picking a truck with a spent pack adds
  ₹383,400, moves the five-year saving from −₹150,505 to −₹533,905, and pushes the break-even to
  140.2 km/day — past the Tata Ace's own 123.2 km/day range gate, which is exactly the case the new
  `breakeven_is_reachable` flag exists to catch.
- Drift tests asserting the shipped battery artifact matches the declared strategy, the declared
  hyperparameters, and carries its own metrics.

### Hardening and readability

Structural work with no behaviour change, verified by an AST comparison that fails if a
comment-only edit touches an executable statement, plus the full suite after every step.

- **Dependencies are bounded.** All 16 runtime requirements were `>=` with no upper bound and no
  lockfile, in a project whose stated purpose is reproducibility. `pulp>=2.8` was the concrete
  hazard: 4.0 removes both `LpVariable(name, ...)` direct construction and `PULP_CBC_CMD`, and the
  maintenance agent uses both, so the next clean install would have broken scheduling outright.
- **Paths have one owner.** Seven modules each recomputed the repo root with
  `Path(__file__).resolve().parents[N]`, N varying between 2 and 3 by nesting depth. They now read
  `app.config`, which also makes the model and data directories overridable by environment
  variable so a container can mount a model volume without a rebuild.
- **`@app.on_event("startup")` replaced with a lifespan context manager.** It is deprecated and was
  emitting a warning on every import.
- **The served version is single-sourced** and matches the changelog, with a test that fails if
  they drift. `/` reported 2.0.0 while this file's newest entry said 2.1.0.
- **The API image stopped shipping the dashboard, the training pipelines and the evaluation
  harness.** Nothing under `backend/` imports them; they were image weight and attack surface in a
  container whose only job is to answer HTTP. The dashboard image now installs from a requirements
  file rather than inline pip arguments, so its dependency layer caches.
- **`.query("supplier_id == @supplier_id")` replaced with boolean indexing** -- opaque to static
  analysis, and slower.

### Readability

In-source prose was 31% of all lines (0.46 prose:code), with ~90 lines narrating what the code used
to do rather than what it does. That is the wrong place for archaeology: a fix belongs in the
commit message and the ADR, a test belongs in `tests/`, and the source should describe what is true
now. Cut to 26% and ~10 lines, keeping every sourced constant, every live constraint and every
non-obvious invariant. No executable statement changed.

### Fixed — documentation

- `.gitignore` used an inline comment, which the format does not support, so the explicit `.venv/`
  rule matched nothing.
- Censoring counts in `survival.py` and `docs/README.md` (6 cells / 13 censored / 63.6% of rows →
  5 / 14 / 73.1%), stale docstrings in `train_battery_model.py`, and the SETUP.md expectations for
  the battery run.
- SETUP.md had no step for the NEU-DET surface classifier at all, though `/ready` requires its
  artifact.
- Test counts (106 → 206, 68 of 72 → 202 of 206) in README, CHANGELOG and `conftest.py`.


## 2.0.0 — reproducibility rebuild

A rebuild around reproducible evaluation. Four published results did not survive being checked,
and correcting them is most of what changed. Breaking API changes, hence the major bump.

### Corrected results

| Claim | Was | Is | Cause |
|---|---|---|---|
| Maintenance improvement | 37.5% | **0%** | Bug in its own greedy baseline ([ADR-0001](docs/adr/0001-maintenance-lp-vs-greedy.md)) |
| Casting accuracy | 99.44% | **97.3%** | Test split shared 97.5% of parts with training ([ADR-0003](docs/adr/0003-casting-split-leakage.md)) |
| Casting defect recall | 1.000 | **0.955** | Same |
| Battery pooled R² | 0.740 | **0.467** | EOL labels broken for 6 of 19 cells; new leave-one-battery-out CV over the 5 that survive |
| Carbon saving (Tata Ace) | 19.7% | **10.8%** | Scope 2 measured at the battery, not the meter |
| Risk P/R/F1 | ~0.6–0.7 in-sample | **AUC-PR 0.807 ± 0.055** held out | Evaluation was 100% in-sample |

### Correctness

- **Vision train/serve skew.** Serving resized with PIL (BICUBIC by default) while training used
  `tf.image.resize`. Cost defect recall 1.000 → 0.982 on the live endpoint while the eval script
  still reported 1.000. Serving now decodes through TensorFlow and is bit-identical to training.
  Switching PIL to `BILINEAR` — the obvious fix — is *worse* (0.958).
- **Grad-CAM explained the wrong class.** The target was unconditionally `preds[:,0]` = P(ok), so
  every `defective` verdict shipped a map of the most OK-looking regions. Correlation with the
  corrected map on a real defective casting: **−0.39**.
- **Battery RUL labels.** The end-of-life search started at cycle 1, so one anomalously low
  break-in reading pinned `RUL ≡ 0` for a cell's whole life — 891 of 2,019 rows.
- **Unknown assets were escalated to the top of the maintenance queue.** `"unknown"` scored the
  same as `"watch"`, so an asset missing from the BOM linkage accumulated two "active signals".
- **LightGBM `subsample` was a silent no-op** without `subsample_freq`.
- **`roll_std_capacity` ddof mismatch** between pandas (train) and numpy (serve).
- **Risk decision threshold lived outside the model artifact**, recomputed at startup from a
  gitignored CSV with a silent `0.3` fallback against a real value of `0.1776`.
- Dashboard re-derived the backend's risk bands with a different threshold; 54 shipments the
  backend called CRITICAL displayed as WATCH.

### API — breaking

- `GET /battery/live-fleet-status` **no longer advances** the simulation. Use
  `POST /battery/live-fleet-tick`. A GET was mutating shared state.
- `discharge_time_s` removed from `/battery/predict-rul` — it was never read.
- Request models reject unknown fields, non-finite floats (`Infinity`/`NaN` previously returned
  **200** with a body that is not legal JSON), and out-of-range values.
- Unknown vehicle model returns **422**, not 400.
- `/` no longer returns a hardcoded `agents_live`; availability is measured from disk.
- `/maintenance/schedule` returns all five schedulers.
- `/carbon/compute-savings` gains `charger_efficiency_used`, `breakeven_grid_factor_tco2_per_mwh`
  and `scope3_sensitivity`.

### Added

- `GET /health` and `GET /ready` — truthful artifact inventory with hashes. A missing model is
  now **503**, not 400, and error bodies no longer leak filesystem paths.
- Structured JSON logging with request-ID correlation. There was previously no `logging` import
  anywhere in ~4,000 lines.
- `evaluation/leakage_check.py` — fails the build if an image split shares a source part.
- `data/prepare_casting.py` — part-level split before augmentation.
- Dashboard split into 10 modules; Altair charts; provenance badges; WCAG contrast fixed
  (`healthy` measured 3.98:1) and asserted by test.
- ADRs, `Makefile` / `make.ps1`, `ruff.toml`, MIT `LICENSE`.

### Testing

- **206 tests**, up from 47. **202 run in CI**, up from ~18 — and CI now executes real model
  inference, where previously it ran none while still installing TensorFlow.
- New: preprocessing parity, operational endpoints, dashboard pages (verified with the backend
  stopped), theme contrast.

## 1.0.0 — ET AI Hackathon 2026 submission

Seven agents behind one FastAPI service and one Streamlit dashboard.
