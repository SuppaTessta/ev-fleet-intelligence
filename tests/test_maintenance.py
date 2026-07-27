"""
Maintenance Optimiser tests -- pure PuLP/greedy logic, no trained model or
data file needed, so these always run (including in CI).
"""
from app.agents.maintenance_agent import MaintenanceJob, run_comparison


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
