# ADR-0005: Right-censored battery cells — exclude by default, keep the censoring-aware objective

**Status:** accepted
**Supersedes:** pooled R² = 0.740, and the earlier "6 of 19 cells" figure

## Context

Of 19 NASA PCoE cells, most never reach their end-of-life threshold — the experiment stopped while
the cell was still healthy. For those rows `RUL` is a **lower bound**, not an observation.

This has now been got wrong twice, in opposite directions, plus a labelling bug underneath both.

## The labelling bug, in three layers

The EOL detector was wrong in ways that only became visible one at a time.

1. **Searched from cycle 1.** Several cells report an anomalously low break-in reading. One such
   reading set `eol_cycle = 1`, so `RUL ≡ 0` for that cell's entire life — six cells, 891 of 2,019
   rows. Fixed by skipping the break-in window, which the `initial_capacity` calculation already
   guarded against.

2. **A single dropout counted as EOL.** Capacity readings are noisy; one sample below a threshold
   is not death. Fixed by requiring three consecutive readings.

3. **A crossing the cell RECOVERED from counted as EOL.** Three consecutive readings still was not
   enough. B0033 crossed at cycle 138, then spent **94.3% of its remaining readings back above the
   threshold**, ending at **99.5% of its initial capacity**, with capacity ranging 0.399–1.885 Ah
   across the series. That is a measurement excursion on a noisy cell, not degradation. Fixed by
   requiring the crossing to be *sustained*.

Layer 3 was found by the study below, where B0033 alone moved the censoring-aware model's mean MAE
from ~14 to 31.7 cycles. **One mislabelled cell out of six dominated the comparison** — which is
what n=6 does.

After all three: **5 cells have a genuine EOL event, 14 are right-censored (73.1% of rows).** The
separation is now physically clean — every observed cell ends at ≤71.4% state of health, every
censored cell at ≥71.8%.

## The censoring question

Three treatments, leave-one-battery-out over the 5 observed cells
(`evaluation/battery_censoring_study.py`):

| Strategy | mean MAE | **median MAE** | pooled R² | wins |
|---|---|---|---|---|
| predict the training mean | 38.7 | — | −0.330 | — |
| linear fade extrapolation | 54.9 | — | −7.986 | — |
| **naive** — treat the bound as an observation | 17.8 | 14.2 | 0.759 | 1/5 |
| **exclude** — drop censored rows | **15.9** | **3.6** | 0.467 | **3/5** |
| **censored** — one-sided hinge | 21.1 | 13.9 | 0.653 | 1/5 |

Read the **median**, not the mean: `exclude` scores 2.6–3.6 on three cells and 32–38 on the other
two, so its mean sits between two clusters it never occupies.

## The actual finding

The aggregate hides a clean conditional result:

| Cell | Group | Same-group observed peers | exclude | censored | censoring-aware helps? |
|---|---|---|---|---|---|
| B0005 | G1 | **1** | 37.9 | **7.5** | **yes** |
| B0006 | G1 | **1** | 32.2 | **27.1** | **yes** |
| B0045 | G7 | 2 | **2.6** | 50.4 | no |
| B0046 | G7 | 2 | **3.4** | 6.5 | no |
| B0047 | G7 | 2 | **3.6** | 13.9 | no |

**Censoring-aware training wins on 2/2 folds where the condition group has one other observed cell,
and loses on 3/3 where it has two.** No exceptions.

That is exactly what the information content predicts. A censored row asserts only "at least this
many cycles remained" — weak evidence, worth having when there is nothing better for that operating
condition, worth ignoring when there is. On G1 the exclusion model trains on a single cell and its
error is 10× the G7 folds; the censored G1 cells (B0007, B0018) supply precisely the missing
constraint.

## Decision

- **`exclude` stays the production default.** Best median MAE, most per-fold wins, simplest.
- **The censoring-aware objective ships and is documented** — `backend/app/survival.py`, a one-sided
  squared hinge that penalises only under-prediction on censored rows.
- **Do not auto-switch on group scarcity.** The rule is clean across 5 observations, and deriving a
  policy from 5 observations is exactly the overfitting this study exists to expose.

## Consequences

- Published: **MAE 15.9 ± 15.7 cycles (median 3.6), pooled R² 0.467**, beating the fade baseline
  5/5 folds. Not 0.740.
- The battery agent is trained on 5 cells. That is the honest amount of usable data.

### Implementation status — corrected in 2.1.0

This decision was accepted and then not implemented, which is worth recording rather than quietly
patching. `train_battery_model.py` evaluated with exclusion and then fitted the **deployed**
artifact on the full frame — the `naive` strategy rejected above. So every published battery
figure described one model and `/battery/predict-rul` answered from another:

| | MAE | median MAE | pooled R² |
|---|---|---|---|
| what the README quoted (`exclude`) | 15.9 | 3.6 | 0.467 |
| what actually shipped (`naive`) | 17.0 | 14.0 | 0.769 |

Nothing failed, because nothing compared the artifact to the decision. Now:

- the policy is a single named constant, `TRAINING_STRATEGY`, applied to the final fit;
- it travels **inside** `battery_rul_model.pkl` alongside the cells trained on and the held-out
  metrics, and is served at `GET /battery/metadata`;
- `tests/test_no_drift.py` fails if the artifact and this ADR disagree.

Same shape as the risk agent's `watch_threshold`: a decision that determines what the model
answers belongs in the model file, not in a document the model never reads.
- A real fleet meeting a new cell chemistry sits in the *scarce* regime, where the censoring-aware
  model wins — so this is a live option, not a curiosity.

## Rejected alternatives

- **XGBoost `survival:aft` or scikit-survival.** The right tools with more data. Here they add a
  dependency and a distributional assumption (log-normal / log-logistic) that 5 cells can neither
  support nor refute. The hinge assumes only that censored labels are lower bounds.
- **Keep treating censored labels as observed.** Best pooled R² (0.759) — and conceptually wrong.
  It teaches the model that a healthy cell near the end of its recording is nearly dead, and
  `business_impact.py` prices premature replacements in rupees. A good number from a wrong model is
  the thing this project keeps finding and removing.
- **Drop B0033 as bad data.** Tempting, but the sustained-crossing rule reclassifies it as censored
  on a principled criterion rather than a per-cell exception, and the same rule applies to every
  cell.

## Implementation note worth keeping

Supplying a custom objective to LightGBM silently disables `boost_from_average`, so the model starts
from 0 and spends its budget climbing to the mean. Seeding `init_score` with the mean of the
observed targets recovered MAE 7.3 → 6.9 on one fold. Easy to miss, and it would have made the
censoring-aware model look worse than it is.
