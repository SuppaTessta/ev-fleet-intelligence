"""Maintenance Operations Optimiser.

A scheduling problem rather than a prediction model: it takes a maintenance
backlog whose priorities come from the Battery, Quality and Compound-Risk agents
and fits it to real workshop capacity.

Where the inputs come from:
- Job urgency: the priority tiers in PRIORITY_PROFILES, derived from the other
  agents on this platform. This is where their output becomes an input.
- Workshop capacity: bays, technicians and shift hours, parameterised on
  industry benchmarks (0.7 technicians/bay, ~2.2 vehicles/bay/day for heavy
  commercial vehicles; PartsTech 2025 shop survey, Heavy Duty Journal).
- Charging uptime: self-reported uptime runs well above independently measured
  functional uptime (Paren Q1 2026: 93.5% self-reported; UC Berkeley/SLAC and
  ChargerHelp 2025: 72-85% actual). A repaired EV usually needs a working
  charger to return to service, so this is a capacity multiplier, not decoration.

FIVE SCHEDULERS ARE RETURNED, INCLUDING THE ONES THAT LOSE.

With a correctly implemented greedy baseline, greedy and the job-count LP tie
exactly. That is what Moore-Hodgson predicts for unit-size jobs under a flat
daily count, where EDF is provably optimal. The LP only earns its place once
capacity is measured in HOURS and unequal job durations turn each day into a
knapsack. All five schedulers are returned so the negative result is checkable
rather than something to take on trust. See ADR-0001.

The planning horizon sizes itself to the queue: a fixed horizon becomes
mathematically infeasible once the work exceeds horizon_days x daily capacity.
"""

import logging
from dataclasses import dataclass

import pulp

log = logging.getLogger("api.maintenance")

PLANNING_HORIZON_DAYS = 14

# priority -> (urgency_deadline_days, estimated_job_hours)
PRIORITY_PROFILES = {
    "critical": (2, 4.0),    # e.g. battery near/at EOL — urgent, and battery swaps take longer
    "watch": (10, 2.5),
    "routine": (30, 1.5),
}

CHARGING_UPTIME_ASSUMED = 0.80  # independently-measured functional uptime, not self-reported (see docstring)

MAX_JOBS = 500          # refuse larger queues rather than build a multi-million-variable MIP
SOLVER_TIME_LIMIT_S = 20  # bound CBC so one request can't pin a core indefinitely

# MAX_JOBS alone does not bound the model: the LP allocates one binary per
# (job, day) and the horizon sizes itself to the work, so a workshop whose day
# barely exceeds its longest job can drive the horizon arbitrarily high. Both
# caps are needed. 180 days is far beyond the longest deadline in
# PRIORITY_PROFILES (30), and the default workshop at MAX_JOBS needs ~36 days
# = 18,000 binaries, so 60,000 leaves ample headroom.
MAX_HORIZON_DAYS = 180
MAX_MODEL_BINARIES = 60_000


class UnknownPriorityError(ValueError):
    """Raised for a priority outside PRIORITY_PROFILES, so the router can map it
    to a 422 instead of leaking a bare KeyError as a 500."""


class InfeasibleScheduleError(RuntimeError):
    """Raised when no feasible schedule exists within the horizon, or the solver
    fails to find one. Explicit so callers never read values off an unsolved model."""


@dataclass
class MaintenanceJob:
    asset_id: str
    priority: str
    reason: str

    def __post_init__(self):
        if self.priority not in PRIORITY_PROFILES:
            raise UnknownPriorityError(
                f"unknown priority {self.priority!r}; expected one of "
                f"{sorted(PRIORITY_PROFILES)}")

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


def compute_horizon_hours(jobs: list["MaintenanceJob"], daily_hours: float) -> int:
    """Days of planning horizon needed for this queue at this hours budget.

    Sized by WORK, not job count: dividing a job count by a job-slot capacity
    under-sizes the horizon as soon as jobs have unequal durations, and too
    short a horizon makes the hours-aware schedulers genuinely infeasible.
    """
    if daily_hours <= 0 or not jobs:
        return PLANNING_HORIZON_DAYS
    # Pack, do not guess. Estimating usable hours as
    # `daily_hours - longest_job` and charging that to every day collapses when
    # the budget only slightly exceeds the longest job, inflating the horizon by
    # an order of magnitude and making capacity NON-MONOTONE -- adding hours
    # could turn an accepted queue into a rejected one. Running the packer is
    # O(jobs x days), costs microseconds, is exact for the greedy, and is a
    # valid bound for the LP, which has the same per-day hours constraint.
    return max(PLANNING_HORIZON_DAYS, _days_needed_to_pack(jobs, daily_hours) + 3)


def _days_needed_to_pack(jobs: list["MaintenanceJob"], daily_hours: float) -> int:
    """Days a first-fit-by-deadline packing needs at this hours budget.

    Mirrors the placement rule in schedule_greedy_hours but without a horizon
    ceiling, so it answers "how many days does this queue take?" rather than
    "does it fit in N?". Returns a day COUNT, not an index.
    """
    remaining: list[float] = []
    for job in sorted(jobs, key=lambda j: (j.deadline_day, -j.duration_hours)):
        for day, free in enumerate(remaining):
            if free >= job.duration_hours:
                remaining[day] = free - job.duration_hours
                break
        else:
            # No existing day has room. A job longer than a whole day can never
            # be placed; the hours-aware schedulers raise on it, and the caller
            # sees that rather than a horizon number.
            remaining.append(max(0.0, daily_hours - job.duration_hours))
    return len(remaining)


def compute_horizon_days(n_jobs: int, daily_capacity: int) -> int:
    """At least enough days to fit every job at this capacity, plus a
    buffer — a fixed horizon becomes mathematically infeasible once
    n_jobs exceeds horizon_days x daily_capacity (found via a real solver
    failure in testing, not a hypothetical concern)."""
    min_days_needed = -(-n_jobs // daily_capacity)  # ceiling division
    return max(PLANNING_HORIZON_DAYS, min_days_needed + 3)


def schedule_greedy(jobs: list[MaintenanceJob], daily_capacity: int, horizon_days: int,
                     overflow_latest: bool = True) -> dict:
    """Earliest-Deadline-First against a job-slot capacity.

    `overflow_latest` controls where a job that cannot make its deadline goes.
    True (correct) parks it in the LATEST open slot: the job cannot become less
    late, so occupying an early slot only steals capacity from a job that could
    still make its own deadline. False reproduces the original implementation
    and exists solely so run_comparison can publish the difference.

    That difference is why five schedulers are returned. With the overflow
    repaired, this greedy and the job-count LP return the same number of missed
    deadlines with the same per-tier split -- Moore-Hodgson, where EDF is
    provably optimal for unit-size jobs under a uniform per-day count and no LP
    can beat it. The optimiser only starts to earn its keep in
    schedule_optimal_hours, where unequal durations turn each day into a
    knapsack. See ADR-0001.
    """
    jobs_sorted = sorted(jobs, key=lambda j: j.deadline_day)
    day_load = dict.fromkeys(range(horizon_days), 0)
    assignments = []

    for job in jobs_sorted:
        assigned_day = None
        for day in range(min(job.deadline_day + 1, horizon_days)):  # deadline day itself is on-time
            if day_load[day] < daily_capacity:
                assigned_day = day
                break
        if assigned_day is None:
            # Can't make the deadline. Park it as late as possible so it stops
            # stealing early capacity from jobs that CAN still make theirs.
            scan = range(horizon_days - 1, -1, -1) if overflow_latest else range(horizon_days)
            for day in scan:
                if day_load[day] < daily_capacity:
                    assigned_day = day
                    break
        if assigned_day is None:
            raise InfeasibleScheduleError(
                f"no free slot for {job.asset_id} within a {horizon_days}-day horizon "
                f"at {daily_capacity} jobs/day")
        day_load[assigned_day] += 1
        assignments.append({"asset_id": job.asset_id, "priority": job.priority,
                             "scheduled_day": assigned_day, "deadline_day": job.deadline_day,
                             "missed_deadline": assigned_day > job.deadline_day})

    label = "Greedy (Earliest-Deadline-First)" if overflow_latest \
        else "Greedy EDF, original overflow (kept for comparison)"
    # counted from the assignments themselves, so this can never disagree with
    # the per-job flags the way the original counter could
    return {"method": label, "assignments": assignments,
            "jobs_missing_deadline": sum(1 for a in assignments if a["missed_deadline"]),
            "total_jobs": len(jobs)}


def schedule_greedy_hours(jobs: list[MaintenanceJob], daily_hours: float, horizon_days: int) -> dict:
    """EDF against an HOURS budget -- the honest counterpart to
    schedule_optimal_hours.

    The job-count schedulers treat a 4.0h critical job and a 1.5h routine job as
    one identical slot, so they will book 8.0h of work into a 6.4h day. Their
    better miss-counts are bought with schedules the workshop cannot execute,
    which makes them the wrong baseline for an hours-based LP.
    """
    jobs_sorted = sorted(jobs, key=lambda j: (j.deadline_day, -j.duration_hours))
    remaining = dict.fromkeys(range(horizon_days), daily_hours)
    assignments = []

    for job in jobs_sorted:
        day = None
        for d in range(min(job.deadline_day + 1, horizon_days)):
            if remaining[d] >= job.duration_hours:
                day = d
                break
        if day is None:  # already doomed -- park it as late as possible
            for d in range(horizon_days - 1, -1, -1):
                if remaining[d] >= job.duration_hours:
                    day = d
                    break
        if day is None:
            raise InfeasibleScheduleError(
                f"no day with {job.duration_hours}h free for {job.asset_id} "
                f"within {horizon_days} days at {daily_hours:.1f}h/day")
        remaining[day] -= job.duration_hours
        assignments.append({"asset_id": job.asset_id, "priority": job.priority,
                             "scheduled_day": day, "deadline_day": job.deadline_day,
                             "missed_deadline": day > job.deadline_day})

    return {"method": "Greedy EDF (hours-aware)", "assignments": assignments,
            "jobs_missing_deadline": sum(1 for a in assignments if a["missed_deadline"]),
            "total_jobs": len(jobs)}


def schedule_optimal_hours(jobs: list[MaintenanceJob], daily_hours: float, horizon_days: int) -> dict:
    """LP over an hours budget per day -- the formulation where optimising pays.

    Solved lexicographically by tier: critical tardiness is minimised and locked
    as a hard constraint before watch is considered, so a higher tier can never
    be traded for a lower one.

    Within each tier there is a second level: minimise the COUNT of tardy jobs
    first, then total days late subject to that count. Both are needed. Count
    alone is what the platform reports but is indifferent between one day late
    and three hundred, so CBC returns an arbitrary point on the optimal face.
    Days-late alone optimises a different objective from the one reported, and
    measurably so -- it spreads small delays across many jobs, producing 27
    tardy jobs where an hours-aware greedy managed 11.
    """
    if not jobs:
        return {"method": "LP-Optimal (PuLP/CBC, hours-based)", "assignments": [],
                "jobs_missing_deadline": 0, "total_jobs": 0, "solver_status": "Optimal",
                "solver_notes": None}

    days = range(horizon_days)
    x = {(i, d): pulp.LpVariable(f"x_{i}_{d}", cat="Binary")
         for i in range(len(jobs)) for d in days}
    tardy = {i: pulp.LpVariable(f"tardy_{i}", cat="Binary") for i in range(len(jobs))}
    late_days = {i: pulp.LpVariable(f"latedays_{i}", lowBound=0) for i in range(len(jobs))}

    def base_constraints():
        for i in range(len(jobs)):
            yield pulp.lpSum(x[i, d] for d in days) == 1
        for d in days:
            # HOURS, not job count -- the substantive change
            yield pulp.lpSum(jobs[i].duration_hours * x[i, d] for i in range(len(jobs))) <= daily_hours
        for i, job in enumerate(jobs):
            over = [d for d in days if d > job.deadline_day]
            yield tardy[i] >= pulp.lpSum(x[i, d] for d in over)
            yield late_days[i] >= pulp.lpSum((d - job.deadline_day) * x[i, d] for d in over)

    status_name = "Optimal"
    notes = []
    solved_any = False
    # only counts are locked between tiers. locked_days was left over from the
    # rejected per-tier lateness pass -- see the note in pass 1 for why that
    # ordering was wrong.
    locked_count = {}

    def solve_stage(name, objective, extra):
        nonlocal status_name
        prob = pulp.LpProblem(name, pulp.LpMinimize)
        for c in base_constraints():
            prob += c
        for c in extra:
            prob += c
        prob += objective
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=SOLVER_TIME_LIMIT_S))
        st = pulp.LpStatus[prob.status]
        # A complete assignment is what we actually need. CBC reports "Not Solved"
        # when the time limit stops it PROVING optimality even though the
        # incumbent is usable -- take it, but stop calling the result "Optimal".
        has_incumbent = all(
            any((pulp.value(x[i, d]) or 0) > 0.5 for d in days) for i in range(len(jobs)))
        if st == "Optimal":
            return
        if has_incumbent and st not in ("Infeasible", "Unbounded", "Undefined"):
            status_name = "Feasible (time-limited)"
            notes.append(f"{name}: hit the {SOLVER_TIME_LIMIT_S}s limit; best found solution used")
            return
        raise InfeasibleScheduleError(
            f"solver returned {st!r} at stage {name} "
            f"({len(jobs)} jobs, {horizon_days}-day horizon, {daily_hours:.1f}h/day)")

    def locks():
        for t, c in locked_count.items():
            idx = [i for i, j in enumerate(jobs) if j.priority == t]
            yield pulp.lpSum(tardy[i] for i in idx) <= c

    # Pass 1: tardy COUNT per tier, in strict priority order.
    #
    # Lateness is a tie-break and belongs after every count is fixed. Minimising
    # a tier's days-late immediately after its count packs the 4.0h critical
    # jobs as early as possible, leaving 2.4h of dead space per day that a 2.5h
    # watch job cannot use -- 27 tardy jobs against 11, while reporting
    # "Optimal", because it was optimal for a question nobody asked.
    for tier in ["critical", "watch", "routine"]:
        tier_idx = [i for i, j in enumerate(jobs) if j.priority == tier]
        if not tier_idx:
            continue

        # A tier whose deadline sits at or beyond the last schedulable day can
        # never be late, so its optimum is 0 by construction. Skip it. Without
        # this, CBC finds the 0 immediately and then burns the entire time limit
        # failing to PROVE optimality of a degenerate objective -- which is
        # exactly what made the routine tier report "Not Solved" after 20s.
        if all(jobs[i].deadline_day >= horizon_days - 1 for i in tier_idx):
            locked_count[tier] = 0
            notes.append(f"{tier}: lateness structurally impossible within the horizon, not solved")
            continue

        solve_stage(f"{tier}_count", pulp.lpSum(tardy[i] for i in tier_idx), list(locks()))
        solved_any = True
        locked_count[tier] = int(round(sum(pulp.value(tardy[i]) or 0 for i in tier_idx)))

    if not solved_any:
        # Every tier was skipped as structurally on-time, so no model has been
        # solved and the x variables have no values yet. Still need a feasible
        # assignment: solve once for feasibility alone.
        solve_stage("feasibility", pulp.lpSum(late_days.values()), [])
    elif any(locked_count.values()):
        # Pass 2: with every tier's tardy count locked, break remaining ties by
        # minimising total days late. Pure tie-break -- it cannot change a count.
        solve_stage("global_lateness", pulp.lpSum(late_days.values()), list(locks()))

    assignments = []
    for i, job in enumerate(jobs):
        chosen = [d for d in days if (pulp.value(x[i, d]) or 0) > 0.5]
        if not chosen:
            raise InfeasibleScheduleError(f"no day assigned to {job.asset_id} in the solved model")
        day = chosen[0]
        assignments.append({"asset_id": job.asset_id, "priority": job.priority,
                             "scheduled_day": day, "deadline_day": job.deadline_day,
                             "missed_deadline": day > job.deadline_day})

    return {"method": "LP-Optimal (PuLP/CBC, hours-based)", "assignments": assignments,
            "jobs_missing_deadline": sum(1 for a in assignments if a["missed_deadline"]),
            "total_jobs": len(jobs), "solver_status": status_name,
            "solver_notes": "; ".join(notes) or None}


def schedule_optimal(jobs: list[MaintenanceJob], daily_capacity: int, horizon_days: int) -> dict:
    """The original job-count LP, retained only as evidence.

    This formulation cannot beat a correctly implemented greedy (Moore-Hodgson;
    see schedule_greedy), and that is only demonstrable if the losing
    formulation stays runnable. Production traffic goes to
    schedule_optimal_hours.

    The lexicographic tier structure was always correct. A weighted-sum
    objective was tried first and rejected: it let critical jobs slip to save
    several lower-tier ones, and the extreme weight ratios needed to prevent
    that made the solver numerically unstable.
    """
    if not jobs:
        return {"method": "LP, job-count (superseded)", "assignments": [],
                "jobs_missing_deadline": 0, "total_jobs": 0, "solver_status": "Optimal"}

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

    status_name = "Optimal"
    solved_any = False
    locked_late_totals = {}  # tier -> optimal (minimum) number late, fixed after solving that tier
    for tier in ["critical", "watch", "routine"]:
        tier_idx = [i for i, j in enumerate(jobs) if j.priority == tier]
        if not tier_idx:
            continue
        if all(jobs[i].deadline_day >= horizon_days - 1 for i in tier_idx):
            locked_late_totals[tier] = 0  # cannot be late within the horizon; see schedule_optimal_hours
            continue
        solved_any = True
        prob = pulp.LpProblem(f"MaintenanceScheduling_{tier}", pulp.LpMinimize)
        for c in base_constraints:
            prob += c
        # lock each already-solved tier's optimal lateness as a hard constraint,
        # so this stage cannot trade it away for a better result here
        for prior_tier, optimal_late in locked_late_totals.items():
            idx = [i for i, j in enumerate(jobs) if j.priority == prior_tier]
            prob += pulp.lpSum(late[i] for i in idx) <= optimal_late

        prob += pulp.lpSum(late[i] for i in tier_idx)  # minimize only this tier's lateness
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=SOLVER_TIME_LIMIT_S))
        # read the status of the problem we actually just solved, not whichever
        # one happened to be constructed last
        status_name = pulp.LpStatus[prob.status]
        if status_name != "Optimal":
            raise InfeasibleScheduleError(
                f"solver returned {status_name!r} while optimising the {tier} tier")
        locked_late_totals[tier] = int(round(sum(pulp.value(late[i]) or 0 for i in tier_idx)))

    if not solved_any:
        # No tier could be late within the horizon, so nothing was solved and the
        # x variables hold no values. Solve once for feasibility.
        prob = pulp.LpProblem("MaintenanceScheduling_feasibility", pulp.LpMinimize)
        for c in base_constraints:
            prob += c
        prob += pulp.lpSum(late[i] for i in range(len(jobs)))
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=SOLVER_TIME_LIMIT_S))
        status_name = pulp.LpStatus[prob.status]
        if status_name not in ("Optimal", "Not Solved"):
            raise InfeasibleScheduleError(f"solver returned {status_name!r} on the feasibility solve")
        status_name = "Optimal"

    assignments = []
    for i, job in enumerate(jobs):
        chosen = [d for d in days if (pulp.value(x[i, d]) or 0) > 0.5]
        if not chosen:
            raise InfeasibleScheduleError(f"no day assigned to {job.asset_id} in the solved model")
        scheduled_day = chosen[0]
        assignments.append({"asset_id": job.asset_id, "priority": job.priority,
                             "scheduled_day": scheduled_day, "deadline_day": job.deadline_day,
                             "missed_deadline": scheduled_day > job.deadline_day})

    missed = sum(1 for a in assignments if a["missed_deadline"])
    return {"method": "LP, job-count (superseded)", "assignments": assignments,
            "jobs_missing_deadline": missed, "total_jobs": len(jobs),
            "solver_status": status_name}


def compute_daily_hours(n_bays: int, technicians: int, hours_per_shift: float,
                         shifts_per_day: int, charging_uptime: float = CHARGING_UPTIME_ASSUMED) -> float:
    """Workshop capacity in HOURS/day — the same arithmetic compute_daily_capacity
    already did internally, just returned before being divided into job slots."""
    effective_bays = min(n_bays, technicians / 0.7)
    return effective_bays * hours_per_shift * shifts_per_day * charging_uptime


def _solve_or_fall_back(solve, witness: dict, method: str) -> dict:
    """Run an LP; if it fails while a greedy witness succeeded, return the witness.

    An InfeasibleScheduleError from an LP means either "no schedule exists" or
    "CBC could not find one in SOLVER_TIME_LIMIT_S". Only the first deserves a
    422. `witness` is a schedule that demonstrably exists, so reaching this
    function rules out the first case -- which is why the exception is swallowed
    here and nowhere else.
    """
    try:
        return solve()
    except InfeasibleScheduleError as exc:
        log.warning("LP scheduler fell back to greedy", extra={"method": method,
                                                               "reason": str(exc)})
        return {**witness, "method": f"{method} -- fell back to greedy",
                "solver_status": "Fallback (greedy)",
                "solver_notes": f"the LP did not return a usable schedule ({exc}); a "
                                f"feasible greedy schedule is reported instead, so these "
                                f"numbers are the greedy row's, not an optimisation result"}


def run_comparison(jobs: list[MaintenanceJob], n_bays: int = 3, technicians: int = 4,
                    hours_per_shift: float = 8.0, shifts_per_day: int = 2) -> dict:
    """Run every scheduler and report all of them, including the ones that lose.

    Publishing the full comparison is the point: the LP has no advantage over a
    correct greedy in the job-count formulation, and a negative result you can
    check is worth more than one you have to take on trust.
    """
    if len(jobs) > MAX_JOBS:
        raise ValueError(f"job queue of {len(jobs)} exceeds the {MAX_JOBS}-job limit")

    avg_job_hours = sum(j.duration_hours for j in jobs) / len(jobs) if jobs else 2.0
    daily_capacity = compute_daily_capacity(n_bays, technicians, hours_per_shift,
                                             shifts_per_day, avg_job_hours)
    daily_hours = compute_daily_hours(n_bays, technicians, hours_per_shift, shifts_per_day)
    # sized by total WORK so the hours-aware schedulers are actually feasible;
    # the job-count schedulers get the same horizon so the comparison is fair
    horizon_days = max(compute_horizon_days(len(jobs), daily_capacity),
                       compute_horizon_hours(jobs, daily_hours))

    # Reject an over-capacity queue up front, with the reason, instead of
    # building the model that would express it. See MAX_HORIZON_DAYS.
    if horizon_days > MAX_HORIZON_DAYS:
        total_hours = sum(j.duration_hours for j in jobs)
        raise InfeasibleScheduleError(
            f"{len(jobs)} jobs ({total_hours:.0f}h of work) against {daily_hours:.1f}h/day "
            f"would need a {horizon_days}-day plan, beyond the {MAX_HORIZON_DAYS}-day "
            f"limit. Add capacity (bays, technicians, or shift hours) or split the queue")
    if len(jobs) * horizon_days > MAX_MODEL_BINARIES:
        raise InfeasibleScheduleError(
            f"{len(jobs)} jobs x {horizon_days} days exceeds the "
            f"{MAX_MODEL_BINARIES:,}-variable scheduling model limit. Add capacity "
            f"to shorten the horizon, or split the queue")

    # The greedy schedulers run FIRST, and the ordering is load-bearing: a
    # greedy result is a constructive proof that a feasible schedule exists, so
    # a subsequent LP failure is CBC running out of time rather than the queue
    # being impossible. Reporting those two the same way turns a solver timeout
    # into a false "no feasible schedule for this queue and capacity".
    greedy_original = schedule_greedy(jobs, daily_capacity, horizon_days, overflow_latest=False)
    greedy_jobcount = schedule_greedy(jobs, daily_capacity, horizon_days)
    greedy = schedule_greedy_hours(jobs, daily_hours, horizon_days)

    optimal_jobcount = _solve_or_fall_back(
        lambda: schedule_optimal(jobs, daily_capacity, horizon_days),
        greedy_jobcount, "LP, job-count (superseded)")
    optimal = _solve_or_fall_back(
        lambda: schedule_optimal_hours(jobs, daily_hours, horizon_days),
        greedy, "LP-Optimal (PuLP/CBC, hours-based)")

    def peak_hours(result):
        load = {}
        for a in result["assignments"]:
            load[a["scheduled_day"]] = load.get(a["scheduled_day"], 0.0) \
                + PRIORITY_PROFILES[a["priority"]][1]
        return round(max(load.values()), 1) if load else 0.0

    return {
        "daily_capacity_jobs": daily_capacity,
        "daily_capacity_hours": round(daily_hours, 1),
        # -- job-count family: both overbook the workshop, kept only to show why --
        "greedy_original_overflow": {**greedy_original, "peak_day_hours": peak_hours(greedy_original)},
        "greedy_jobcount": {**greedy_jobcount, "peak_day_hours": peak_hours(greedy_jobcount)},
        "optimal_jobcount": {**optimal_jobcount, "peak_day_hours": peak_hours(optimal_jobcount)},
        # -- hours-feasible family: the only two that produce executable schedules --
        "greedy": {**greedy, "peak_day_hours": peak_hours(greedy)},
        "optimal": {**optimal, "peak_day_hours": peak_hours(optimal)},
        # like-for-like: hours-aware greedy vs hours-based LP, both feasible
        "improvement": greedy["jobs_missing_deadline"] - optimal["jobs_missing_deadline"],
        "improvement_vs_original_baseline": (greedy_original["jobs_missing_deadline"]
                                              - optimal["jobs_missing_deadline"]),
        "assumptions": f"{n_bays} bays, {technicians} technicians (0.7 tech/bay benchmark), "
                       f"{hours_per_shift}h x {shifts_per_day} shifts/day = {daily_hours:.1f}h/day "
                       f"after {int(CHARGING_UPTIME_ASSUMED*100)}% charging uptime "
                       f"(independently-measured functional rate, not self-reported), "
                       f"{horizon_days}-day planning horizon.",
    }
