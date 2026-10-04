# ADR-0001: Keep the maintenance LP, publish that it does not beat greedy

**Status:** accepted
**Supersedes:** the claim that lexicographic LP scheduling reduced missed deadlines by 37.5%

## Context

The Maintenance Optimiser shipped a lexicographic LP (PuLP/CBC) alongside an Earliest-Deadline-First
greedy heuristic, and published: *16/52 → 10/52 missed deadlines, a 37.5% reduction.* The greedy was
described in its own docstring as "a real, standard scheduling heuristic, not a strawman".

It was a strawman, unintentionally. `schedule_greedy` sent a job that could no longer make its
deadline to the **earliest** open slot:

```python
if assigned_day is None:                    # no capacity on or before the deadline
    for day in range(horizon_days):         # <-- earliest open slot
        ...
```

A job that is already late cannot become less late, but occupying an early slot consumes capacity a
later-deadline job still needs. Reversing that scan is the standard repair.

## Measurements

Published scenario — 52 jobs, 1 bay, 1 technician, 6.4 h/day:

| Scheduler | Missed | Peak day | Executable? |
|---|---|---|---|
| Greedy, original overflow | 14 | 8.0 h | **No — overbooked** |
| Greedy, overflow repaired | 8 | 8.0 h | **No — overbooked** |
| LP, job-count | 8 | 8.0 h | **No — overbooked** |
| Greedy, hours-aware | **11** | 6.0 h | Yes |
| **LP, hours-based** | **11** | 6.0 h | Yes |

Two findings, not one:

1. **Repairing the baseline erases the entire improvement.** Greedy ties the job-count LP exactly,
   with the same per-tier split. Across 25 randomised scenarios the LP won 0.
2. **Both original schedulers were overbooking the workshop** — 8.0 h of work into a 6.4 h day.
   They only looked good because they scheduled work that could not physically happen. With hours
   enforced the honest answer is 11 missed, 9 of them critical, not 6. The platform was
   *understating* the problem.

With an hours-based capacity constraint and an hours-aware greedy, across 40 randomised scenarios:
**LP better 0, greedy better 0, tied 40.**

## Why the LP cannot win here

With unit-size jobs, a uniform per-day job count, and deadlines determined only by tier, this is the
**Moore–Hodgson** setting, in which an EDF-based greedy is provably optimal for minimising the number
of tardy jobs. No solver can beat it. The 37.5% was never a result.

Modelling capacity in hours makes each day a knapsack and removes greedy's optimality guarantee — so
the LP *could* win. On these job-duration profiles it still does not.

## Decision

1. Repair the greedy overflow.
2. Make the LP capacity constraint hours-based, since job durations were already declared and used
   only to compute a scalar average before being discarded.
3. **Return all five schedulers from the API** and show them in the dashboard, including the two
   that overbook and the two that lose.
4. Keep the LP.

## Consequences

- The headline claim is gone. `improvement` is 0 on the published scenario and the API says so.
- Keeping the losing formulations runnable is what makes the negative result checkable rather than
  something a reader has to take on trust.
- The LP earns its place the moment the model gets harder — per-asset durations, heterogeneous bays,
  technician skill matching — which is exactly when a heuristic stops being safe.
- **A documented negative result about your own headline is worth more than the headline was.**

## Rejected alternatives

- **Delete the LP and ship greedy.** Simpler and honest, but discards the reusable formulation and
  makes the finding invisible.
- **Quietly fix greedy and restate the number.** The corrected number is 0. Presenting that without
  the reason would be more confusing than the bug.
- **Weighted-sum objective instead of lexicographic** (rejected earlier, during the original build):
  it let critical jobs slip to save several lower-tier ones, and the extreme weight ratios needed to
  prevent that made CBC numerically unstable.
- **Per-tier lateness tie-breaking** (rejected during this work): minimising each tier's *days late*
  immediately after its count packs 4.0 h critical jobs into the earliest days, stranding 2.4 h gaps
  that 2.5 h watch jobs cannot use. It cost 16 extra tardy watch jobs — 27 total against a plain
  greedy's 11 — while reporting "Optimal", because it was optimal for a question nobody asked.
  Lateness is now a global tie-break applied after every tier's count is locked.
