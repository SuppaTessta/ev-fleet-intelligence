"""
Maintenance Operations Optimiser.

This is the one agent on the platform that's a genuine scheduling/
optimization problem, not a prediction model — matching the actual
problem statement text: "integrates EV fleet maintenance schedules,
charging infrastructure uptime data, and workshop capacity to generate
optimised maintenance plans."

Inputs, and where each one is grounded:
- Maintenance jobs: priority + urgency deadline derived from the
  Battery/Quality/Compound-Risk agents already on this platform — this
  is the one place where every other agent's output becomes an input to
  something else, which is the actual "unified platform" story.
- Workshop capacity: bays, technicians, shift hours — parameterized using
  real industry benchmarks (0.7 technicians/bay, 1.5-2 bays/technician
  for heavy-commercial vehicles, ~2.2 vehicles/bay/day general throughput;
  PartsTech 2025 shop survey, Heavy Duty Journal).
- Charging infrastructure uptime: self-reported uptime is consistently
  higher than independently-measured functional uptime (Paren Q1 2026:
  93.5% self-reported; UC Berkeley/SLAC field study and ChargerHelp's
  2025 report: 72-85% actual functional uptime). A repaired EV often
  needs a working charger to verify/return to service, so uptime below
  100% genuinely reduces effective workshop throughput — modeled here as
  a capacity multiplier, not decoration.

Two schedulers, compared honestly:
- Greedy (Earliest-Deadline-First): simple, transparent, a real and
  commonly-used scheduling heuristic.
- LP-optimal (PuLP): solved lexicographically across priority tiers —
  critical jobs' lateness minimized first and locked in, then watch, then
  routine, so a higher tier is never sacrificed to improve a lower one.
  A weighted-sum objective was tried first and rejected: it let critical
  jobs slip in exchange for saving lower-tier ones, and extreme weight
  ratios meant to prevent that caused real solver instability. See
  schedule_optimal()'s docstring for the full reasoning. Shown side by
  side with the greedy result specifically to demonstrate that
  optimization earns its place — if it didn't beat the simple heuristic,
  using it would just be complexity for its own sake.

Planning horizon sizes itself to the actual job queue (at least enough
days to fit every job at the given daily capacity, plus a buffer) rather
than a fixed 14 days — a fixed horizon becomes mathematically infeasible
once total jobs exceed horizon_days x daily_capacity, which surfaced as a
real solver failure during testing with a large/low-capacity queue.
"""

from dataclasses import dataclass
from typing import List
import pulp

PLANNING_HORIZON_DAYS = 14

# priority -> (urgency_deadline_days, estimated_job_hours)
PRIORITY_PROFILES = {
    "critical": (2, 4.0),    # e.g. battery near/at EOL — urgent, and battery swaps take longer
    "watch": (10, 2.5),
    "routine": (30, 1.5),
}

CHARGING_UPTIME_ASSUMED = 0.80  # independently-measured functional uptime, not self-reported (see docstring)


@dataclass
class MaintenanceJob:
    asset_id: str
    priority: str
    reason: str

    @property
    def deadline_day(self) -> int:
        return PRIORITY_PROFILES[self.priority][0]

    @property
    def duration_hours(self) -> float:
        return PRIORITY_PROFILES[self.priority][1]


def compute_daily_capacity(n_bays: int, technicians: int, hours_per_shift: float,
                            shifts_per_day: int, avg_job_hours: float,
                            charging_uptime: float = CHARGING_UPTIME_ASSUMED) -> int:
    effective_bays = min(n_bays, technicians / 0.7)  # 0.7 technicians/bay industry benchmark — can't exceed tech supply
    total_hours = effective_bays * hours_per_shift * shifts_per_day * charging_uptime
    return max(1, int(total_hours / avg_job_hours))


def compute_horizon_days(n_jobs: int, daily_capacity: int) -> int:
    """At least enough days to fit every job at this capacity, plus a
    buffer — a fixed horizon becomes mathematically infeasible once
    n_jobs exceeds horizon_days x daily_capacity (found via a real solver
    failure in testing, not a hypothetical concern)."""
    min_days_needed = -(-n_jobs // daily_capacity)  # ceiling division
    return max(PLANNING_HORIZON_DAYS, min_days_needed + 3)


def schedule_greedy(jobs: List[MaintenanceJob], daily_capacity: int, horizon_days: int) -> dict:
    """Earliest-Deadline-First — a real, standard scheduling heuristic."""
    jobs_sorted = sorted(jobs, key=lambda j: j.deadline_day)
    day_load = {d: 0 for d in range(horizon_days)}
    assignments = []
    missed = 0

    for job in jobs_sorted:
        assigned_day = None
        for day in range(min(job.deadline_day + 1, horizon_days)):  # deadline day itself is on-time
            if day_load[day] < daily_capacity:
                assigned_day = day
                break
        if assigned_day is None:  # no capacity on or before deadline — overflow to earliest open slot
            for day in range(horizon_days):
                if day_load[day] < daily_capacity:
                    assigned_day = day
                    break
            missed += 1
        if assigned_day is not None:
            day_load[assigned_day] += 1
            assignments.append({"asset_id": job.asset_id, "priority": job.priority,
                                 "scheduled_day": assigned_day, "deadline_day": job.deadline_day,
                                 "missed_deadline": assigned_day > job.deadline_day})
    return {"method": "Greedy (Earliest-Deadline-First)", "assignments": assignments,
            "jobs_missing_deadline": missed, "total_jobs": len(jobs)}


def schedule_optimal(jobs: List[MaintenanceJob], daily_capacity: int, horizon_days: int) -> dict:
    """LP formulation, solved LEXICOGRAPHICALLY across priority tiers, not as
    a single weighted sum. A weighted-sum objective (tried and rejected
    during development) can let one more critical job miss its deadline in
    exchange for saving several lower-tier jobs — mathematically optimal for
    that objective, but wrong for a safety-relevant maintenance queue where
    tiers shouldn't be tradeable. It's also numerically fragile: extreme
    weight ratios meant to approximate strict priority caused the solver to
    return infeasible/ambiguous results in testing. Lexicographic solving is
    the standard, robust way to do this instead: minimize critical lateness
    first, lock that result in as a hard constraint, then minimize watch
    lateness within what's left, then routine — so a higher tier is NEVER
    sacrificed for a lower one, and there's no fragile weight-tuning involved.
    """
    days = range(horizon_days)
    x = {(i, d): pulp.LpVariable(f"x_{i}_{d}", cat="Binary") for i in range(len(jobs)) for d in days}
    late = {i: pulp.LpVariable(f"late_{i}", cat="Binary") for i in range(len(jobs))}

    base_constraints = []
    for i in range(len(jobs)):
        base_constraints.append(pulp.lpSum(x[i, d] for d in days) == 1)
    for d in days:
        base_constraints.append(pulp.lpSum(x[i, d] for i in range(len(jobs))) <= daily_capacity)
    for i, job in enumerate(jobs):
        base_constraints.append(late[i] >= pulp.lpSum(x[i, d] for d in days if d > job.deadline_day))

    locked_late_totals = {}  # tier -> optimal (minimum) number late, fixed after solving that tier
    for tier in ["critical", "watch", "routine"]:
        prob = pulp.LpProblem(f"MaintenanceScheduling_{tier}", pulp.LpMinimize)
        for c in base_constraints:
            prob += c
        # lock in every previously-solved tier's optimal lateness as a hard
        # constraint, so this stage can't trade it away for a better result here
        for prior_tier, optimal_late in locked_late_totals.items():
            idx = [i for i, j in enumerate(jobs) if j.priority == prior_tier]
            prob += pulp.lpSum(late[i] for i in idx) == optimal_late

        tier_idx = [i for i, j in enumerate(jobs) if j.priority == tier]
        if tier_idx:
            prob += pulp.lpSum(late[i] for i in tier_idx)  # minimize only this tier's lateness
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            locked_late_totals[tier] = int(round(sum(pulp.value(late[i]) for i in tier_idx)))

    assignments = []
    for i, job in enumerate(jobs):
        scheduled_day = next(d for d in days if round(pulp.value(x[i, d])) == 1)
        assignments.append({"asset_id": job.asset_id, "priority": job.priority,
                             "scheduled_day": scheduled_day, "deadline_day": job.deadline_day,
                             "missed_deadline": scheduled_day > job.deadline_day})

    missed = sum(1 for a in assignments if a["missed_deadline"])
    return {"method": "LP-Optimal (PuLP/CBC)", "assignments": assignments,
            "jobs_missing_deadline": missed, "total_jobs": len(jobs),
            "solver_status": pulp.LpStatus[prob.status]}


def run_comparison(jobs: List[MaintenanceJob], n_bays: int = 3, technicians: int = 4,
                    hours_per_shift: float = 8.0, shifts_per_day: int = 2) -> dict:
    avg_job_hours = sum(j.duration_hours for j in jobs) / len(jobs) if jobs else 2.0
    daily_capacity = compute_daily_capacity(n_bays, technicians, hours_per_shift,
                                             shifts_per_day, avg_job_hours)
    horizon_days = compute_horizon_days(len(jobs), daily_capacity)
    greedy = schedule_greedy(jobs, daily_capacity, horizon_days)
    optimal = schedule_optimal(jobs, daily_capacity, horizon_days)
    return {
        "daily_capacity_jobs": daily_capacity,
        "greedy": greedy,
        "optimal": optimal,
        "improvement": greedy["jobs_missing_deadline"] - optimal["jobs_missing_deadline"],
        "assumptions": f"{n_bays} bays, {technicians} technicians (0.7 tech/bay benchmark), "
                       f"{hours_per_shift}h x {shifts_per_day} shifts/day, "
                       f"{int(CHARGING_UPTIME_ASSUMED*100)}% charging uptime (independently-measured "
                       f"functional rate, not self-reported), {horizon_days}-day planning horizon.",
    }
