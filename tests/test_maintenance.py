"""
Maintenance Optimiser tests -- pure PuLP/greedy logic, no trained model or
data file needed, so these always run (including in CI).
"""
import pytest
from app.agents.maintenance_agent import (
    MAX_HORIZON_DAYS,
    MAX_JOBS,
    MAX_MODEL_BINARIES,
    InfeasibleScheduleError,
    MaintenanceJob,
    compute_daily_hours,
    compute_horizon_hours,
    run_comparison,
)


def test_empty_queue_is_handled():
    result = run_comparison([], n_bays=2, technicians=2, hours_per_shift=8.0, shifts_per_day=1)
    assert result["greedy"]["total_jobs"] == 0
    assert result["optimal"]["total_jobs"] == 0


def test_ample_capacity_zero_missed_deadlines_both_schedulers():
    """With generous capacity relative to the queue, neither scheduler should
    need to miss anything -- this is the "optimization isn't needed here"
    case, and both approaches should agree on that."""
    jobs = (
        [MaintenanceJob(f"EV-{i:02d}", "critical", "battery near EOL") for i in range(1, 3)]
        + [MaintenanceJob(f"EV-{i:02d}", "watch", "supplier flagged") for i in range(3, 8)]
        + [MaintenanceJob(f"EV-{i:02d}", "routine", "scheduled check") for i in range(8, 14)]
    )
    result = run_comparison(jobs, n_bays=3, technicians=4, hours_per_shift=8.0, shifts_per_day=2)
    assert result["greedy"]["jobs_missing_deadline"] == 0
    assert result["optimal"]["jobs_missing_deadline"] == 0


def test_optimal_never_worse_than_greedy_on_critical_tier():
    """The lexicographic guarantee this agent is built around: LP-optimal
    must never let MORE critical jobs miss their deadline than greedy does,
    even under severe contention -- it can only match or improve on the
    highest tier, never trade it away for lower tiers."""
    jobs = (
        [MaintenanceJob(f"EV-{i:02d}", "critical", "battery near EOL") for i in range(1, 13)]
        + [MaintenanceJob(f"EV-{i:02d}", "watch", "supplier flagged") for i in range(13, 33)]
        + [MaintenanceJob(f"EV-{i:02d}", "routine", "scheduled check") for i in range(33, 53)]
    )
    result = run_comparison(jobs, n_bays=1, technicians=1, hours_per_shift=8.0, shifts_per_day=1)

    def missed_by_tier(assignments):
        counts = {"critical": 0, "watch": 0, "routine": 0}
        for a in assignments:
            if a["missed_deadline"]:
                counts[a["priority"]] += 1
        return counts

    greedy_missed = missed_by_tier(result["greedy"]["assignments"])
    optimal_missed = missed_by_tier(result["optimal"]["assignments"])

    assert optimal_missed["critical"] <= greedy_missed["critical"]
    # under this specific severe-contention scenario the LP recovers its
    # improvement from the watch tier -- documented in the submission doc
    assert result["optimal"]["jobs_missing_deadline"] <= result["greedy"]["jobs_missing_deadline"]


def test_solver_reports_optimal_status():
    jobs = [MaintenanceJob("EV-01", "routine", "scheduled check")]
    result = run_comparison(jobs, n_bays=1, technicians=1, hours_per_shift=8.0, shifts_per_day=1)
    assert result["optimal"]["solver_status"] == "Optimal"


# ---------------------------------------------------------------- model size

def test_capacity_is_monotone_and_schedulable_queues_are_not_refused():
    """More capacity must never turn a served request into a refused one.

    compute_horizon_hours estimated `usable = daily_hours - longest` and applied
    that worst-case stranding allowance to EVERY day. At 4.8h/day against a 4.0h
    critical job it read 0.8, so 142h of work "needed" 178 days when a first-fit
    packs it into 32 -- and the MAX_HORIZON_DAYS guard then refused it, quoting a
    fabricated "181-day plan". The same 93-job queue was served at 5.0h/shift,
    refused at 5.5h and 6.0h, and served again at 6.5h.
    """
    jobs = ([MaintenanceJob("EV-C", "critical", "battery near EOL")]
            + [MaintenanceJob(f"EV-{i:03d}", "routine", "scheduled check") for i in range(92)])

    horizons = []
    for hours_per_shift in (5.0, 5.5, 6.0, 6.5, 8.0, 10.0):
        daily_hours = compute_daily_hours(1, 2, hours_per_shift, 1)
        horizons.append(compute_horizon_hours(jobs, daily_hours))
    assert horizons == sorted(horizons, reverse=True), (
        f"horizon grew as capacity grew: {horizons}")
    assert max(horizons) <= MAX_HORIZON_DAYS, (
        f"a 142h queue at 4-8h/day should never approach the {MAX_HORIZON_DAYS}-day cap: "
        f"{horizons}")

    # and the case that was refused now schedules, executably
    result = run_comparison(jobs, n_bays=1, technicians=2, hours_per_shift=6.0, shifts_per_day=1)
    budget = result["daily_capacity_hours"]
    assert len(result["optimal"]["assignments"]) == len(jobs)
    for key in ("greedy", "optimal"):
        assert result[key]["peak_day_hours"] <= budget + 1e-6, (
            f"{key} overbooks the workshop: {result[key]['peak_day_hours']} > {budget}")


def test_horizon_cannot_be_driven_past_the_limit():
    """A schema-valid request must not be able to size the MIP arbitrarily.

    MAX_JOBS capped the queue, but the horizon sizes itself as
    total_hours / (daily_hours - longest_job). A workshop whose day is barely
    longer than its longest job drives that denominator toward zero: 500
    critical jobs at 1 bay / 1 technician / 5.1h x 1 shift gives 4.08h/day
    against 4.0h jobs, a 25,003-day horizon, and 12,501,500 binaries. Nothing
    rejected it -- the request simply consumed the worker, on an API the README
    notes has no authentication in front of it.
    """
    jobs = [MaintenanceJob(f"EV-{i}", "critical", "battery near EOL") for i in range(MAX_JOBS)]
    daily_hours = compute_daily_hours(1, 1, 5.1, 1)
    horizon = compute_horizon_hours(jobs, daily_hours)
    assert horizon > MAX_HORIZON_DAYS, (
        "this scenario no longer reproduces the unbounded horizon -- pick another")
    # the estimate must also be TRUTHFUL: one 4.0h job fits per 4.08h day, so
    # 500 jobs genuinely need ~500 days. The old heuristic claimed 25,003.
    assert horizon < 2 * MAX_JOBS, f"horizon {horizon} is not a real day count"

    with pytest.raises(InfeasibleScheduleError) as exc:
        run_comparison(jobs, n_bays=1, technicians=1, hours_per_shift=5.1, shifts_per_day=1)
    # the message has to name the cause, since the caller never chose a horizon
    assert "capacity" in str(exc.value).lower() or "split the queue" in str(exc.value)


def test_the_default_workshop_at_max_jobs_still_returns_a_schedule():
    """The counterpart guard: bounding the model must not start rejecting
    queues that are merely large.

    This case also used to fail, differently. CBC hits its time limit on the
    routine tier with no complete assignment, the LP raised
    InfeasibleScheduleError, and the router turned that into 422 "no feasible
    schedule for this queue and capacity" -- for a queue the greedy scheduler
    had already solved. A greedy result is a constructive proof of feasibility,
    so the LP now falls back to it and says so rather than claiming the queue
    is impossible.
    """
    tiers = ["critical", "watch", "routine"]
    jobs = [MaintenanceJob(f"EV-{i}", tiers[i % 3], "mixed backlog") for i in range(MAX_JOBS)]
    result = run_comparison(jobs)   # documented default workshop

    assert result["optimal"]["total_jobs"] == MAX_JOBS
    assert len(result["optimal"]["assignments"]) == MAX_JOBS
    assert result["optimal"]["jobs_missing_deadline"] <= result["greedy"]["jobs_missing_deadline"]
    # whatever happened, the caller can tell whether they got an optimum
    assert result["optimal"]["solver_status"]


def test_model_size_stays_within_the_declared_bound():
    """Whatever horizon survives the guards, jobs x days stays buildable."""
    tiers = ["critical", "watch", "routine"]
    jobs = [MaintenanceJob(f"EV-{i}", tiers[i % 3], "mixed backlog") for i in range(MAX_JOBS)]
    daily_hours = compute_daily_hours(3, 4, 8.0, 2)
    horizon = compute_horizon_hours(jobs, daily_hours)
    assert horizon <= MAX_HORIZON_DAYS
    assert MAX_JOBS * horizon <= MAX_MODEL_BINARIES
