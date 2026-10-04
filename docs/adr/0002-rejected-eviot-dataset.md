# ADR-0002: Reject the EVIoT-PredictiveMaint dataset

**Status:** accepted

## Context

The platform's most-cited limitation is that it has **no live IoT/BMS/telematics feed** — it replays
historical NASA cycling data instead. The `EVIoT-PredictiveMaint` dataset appeared to close that gap
exactly:

- 175,393 rows at 15-minute intervals, 2020-01-01 → 2025-01-01
- 30 columns: `SoC`, `SoH`, battery V/I/temperature, `Charge_Cycles`, motor temperature, vibration,
  torque, RPM, brake pad wear, tyre pressure, suspension load, ambient conditions, load weight,
  speed, distance, route roughness
- **Labels already present**: `RUL`, `TTF`, `Failure_Probability`, `Component_Health_Score`,
  `Maintenance_Type`

It is the schema this project would have designed for itself.

## Investigation

It is uniformly-distributed random noise. Measured, not inferred:

**Every column has ≈ zero lag-1 autocorrelation.** Real 15-minute telemetry would be ~0.99.

| Column | lag-1 autocorrelation |
|---|---|
| `SoC` | +0.0019 |
| `SoH` | +0.0003 |
| `Battery_Voltage` | +0.0003 |
| `Battery_Temperature` | −0.0019 |
| `RUL` | −0.0004 |
| `Component_Health_Score` | −0.0015 |

**Every column is uniform between round bounds** — `SoH` [0.40, 1.00], `Battery_Voltage`
[200.0, 400.0], `Charge_Cycles` [100.0, 700.0], `RUL` [0, 300], `TTF` [0, 200]. These are
`np.random.uniform` arguments, not physical ranges.

**Monotonic counters are not monotonic.** `Charge_Cycles` *decreases* on 50.0% of consecutive rows.
So does `Brake_Pad_Wear`.

**Every pairwise correlation is ≈ 0.** `SoH`↔`RUL` = 0.001. `SoH`↔`Charge_Cycles` = 0.004. RUL is
*defined* by state of health; in real data these are near-perfectly correlated.

## Decision

Do not use it, for any agent.

The labels are statistically independent of the features, so the dataset is **unlearnable**: any
model trained on it can only achieve chance performance, and any reported accuracy would be an
artifact of the evaluation rather than a property of the model.

Record the rejection here, with the measurements.

## Consequences

- The "no live telemetry" limitation stands. The telemetry simulator continues to replay real NASA
  cycles and says plainly that it is a simulator.
- No agent's numbers are contaminated by a dataset that would have made them meaningless.

## Why this is worth an ADR

This is the most dangerous class of dataset: it has exactly the schema you want, arrives with
ready-made labels, and produces confident-looking results that mean nothing. Nothing about the file
name, the column list, or the row count would have warned anyone. A ten-line autocorrelation check
did.

The reusable lesson: **before training on an unfamiliar dataset, check that its features and labels
are actually related.** For time series, autocorrelation and a correlation matrix take a minute and
would have caught this before any model was fitted.

## Related

Datasets checked and **accepted** during the same pass:

- **C-MAPSS turbofan** — real. 100 engines, 128–362 cycles each, sensor drift with within-unit
  autocorrelation ≈ 0.6, ground-truth RUL for 100 test engines. The canonical public RUL benchmark,
  and still the right way to calibrate the battery methodology — but **deleted from local storage**
  during a cleanup pass, since it was never wired in. Re-download from the NASA PCoE repository if
  that benchmark is picked up.
- **AI4I 2020** — real. Tool wear autocorrelation 0.931, torque↔failure r = 0.19, five labelled
  failure modes, 3.4% class imbalance. A candidate for a production-line equipment-health agent.
