"""
Guards against constants and tables that exist in more than one place drifting apart.

Some duplication here is deliberate. The demo fleet is defined in both
`backend/app/agents/telemetry_simulator.py` and `dashboard/fleet_data.py`
because the dashboard talks to the backend over HTTP and does not import its
internals -- that decoupling is the point of `api_client.py`, and breaking it to
share a tuple would be a worse trade. Sharing it properly needs a backend
endpoint that serves the demo fleet, which is a feature, not cleanup.

So instead of merging them, this fails the build if they stop matching. Every
other shared constant IS single-sourced in `backend/app/constants.py`; these
tests assert that too, so a future edit cannot reintroduce a second copy
unnoticed.

The failure mode being prevented is real: a duplicated feature-window constant
sat next to a duplicated feature *formula* (pandas ddof=1 vs numpy ddof=0), and
the mismatch shifted every served prediction until it was measured.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Order is no longer load-bearing. The dashboard entrypoint used to be
# dashboard/app.py, which shadowed the backend's `app` PACKAGE whenever the
# dashboard directory came first on sys.path -- `import app.agents...` then
# resolved to the single-file module and failed with "'app' is not a package".
# Renaming it to dashboard/main.py removed the collision at the source.
sys.path.insert(0, str(ROOT / "dashboard"))
sys.path.insert(0, str(ROOT / "backend"))

from app.agents import telemetry_simulator  # noqa: E402
from app.agents.compound_risk_agent import FLEET_LINKAGE  # noqa: E402
from app.constants import (  # noqa: E402
    BATTERY_COST_INR_PER_KWH,
    OPERATING_DAYS_PER_YEAR,
    ROLL_WINDOW,
)


def test_demo_fleet_matches_between_backend_and_dashboard():
    """Both define the same 8 (battery, cutoff, asset_id, depot) tuples. If they
    diverge, the Live Fleet Monitor and every other page describe different
    trucks under the same names."""
    import fleet_data

    assert list(telemetry_simulator.FLEET) == list(fleet_data.FLEET_CUTOFFS), (
        "the demo fleet definition has drifted between backend and dashboard")


def test_bom_linkage_covers_exactly_the_demo_fleet():
    """A truck present in the fleet but absent from FLEET_LINKAGE used to be
    escalated to top priority, because unknown scored the same as watch. The
    escalation is fixed, but a mismatch still means a silently unlinked truck."""
    import fleet_data

    fleet_ids = {asset_id for _, _, asset_id, _ in fleet_data.FLEET_CUTOFFS}
    assert set(FLEET_LINKAGE) == fleet_ids


def test_roll_window_is_single_sourced():
    """The train/serve feature window. Two copies is how the ddof mismatch hid."""
    from app.agents.battery_agent import ROLL_WINDOW as serving

    sys.path.insert(0, str(ROOT))
    from train.train_battery_model import ROLL_WINDOW as training

    assert serving == training == ROLL_WINDOW


def test_battery_cost_is_single_sourced():
    """business_impact and tco both price battery replacement. Two figures that
    disagree would put different rupee numbers on the same decision."""
    from app.agents import tco
    from app.business_impact import BATTERY_COST_PER_KWH_INR

    assert BATTERY_COST_PER_KWH_INR == BATTERY_COST_INR_PER_KWH
    assert tco.BATTERY_COST_INR_PER_KWH == BATTERY_COST_INR_PER_KWH


def test_operating_days_is_single_sourced():
    """The carbon agent annualises emissions and the TCO model annualises cost.
    Different year lengths would make the two disagree about the same vehicle."""
    from app.agents import tco

    assert tco.OPERATING_DAYS_PER_YEAR == OPERATING_DAYS_PER_YEAR


def test_shipped_battery_model_uses_the_documented_censoring_policy():
    """ADR-0005 decides `exclude`; the artifact has to agree.

    The final fit used to run on the full frame -- all 19 cells, including the
    14 whose RUL label only records when NASA switched the rig off -- while
    every published battery figure was measured with those cells dropped. The
    served model was therefore not the model any number in this repo described:
    17.0 MAE / median 14.0 / pooled R2 0.769 against a published 15.9 / 3.6 /
    0.467. Nothing failed, because nothing compared the two.
    """
    from conftest import MODELS_DIR, skip_if_missing

    skip_if_missing(MODELS_DIR / "battery_rul_model.pkl")
    import joblib

    from train.train_battery_model import TRAINING_STRATEGY

    artifact = joblib.load(MODELS_DIR / "battery_rul_model.pkl")
    assert artifact.get("training_strategy") == TRAINING_STRATEGY, (
        "battery_rul_model.pkl was not trained under the strategy the pipeline "
        "declares -- re-run train/train_battery_model.py")
    if TRAINING_STRATEGY == "exclude":
        assert artifact["trained_on_cells"] == ["B0005", "B0006", "B0045", "B0046", "B0047"], (
            "exclusion should leave exactly the five cells with a genuine EOL event")


def test_shipped_battery_model_uses_the_declared_hyperparameters():
    """LGBM_PARAMS is the only place hyperparameters are allowed to live.

    There were three copies. `subsample_freq` -- without which LightGBM's
    bagging silently never runs -- had been added to one of them, so the
    evaluated model was regularised and the shipped one was not.
    """
    from conftest import MODELS_DIR, skip_if_missing

    skip_if_missing(MODELS_DIR / "battery_rul_model.pkl")
    import joblib

    from train.train_battery_model import LGBM_PARAMS

    params = joblib.load(MODELS_DIR / "battery_rul_model.pkl")["model"].get_params()
    for key, expected in LGBM_PARAMS.items():
        assert params[key] == expected, (
            f"shipped model has {key}={params[key]!r}, pipeline declares {expected!r}")


def test_battery_metrics_travel_with_the_artifact():
    """The dashboard reads MAE from /battery/metadata rather than hardcoding it.

    It hardcoded 21.2 +/- 12.4 over "6 NASA cells" while the model scored
    15.9 +/- 15.7 over 5, and no test could see the difference because the
    caption had no source to be checked against.
    """
    from conftest import MODELS_DIR, skip_if_missing

    skip_if_missing(MODELS_DIR / "battery_rul_model.pkl")
    import joblib

    metrics = joblib.load(MODELS_DIR / "battery_rul_model.pkl").get("metrics", {})
    assert {"mae_mean", "mae_sd", "mae_median", "n_folds", "pooled_r2"} <= set(metrics), (
        "artifact carries no held-out metrics; the dashboard cannot report accuracy "
        "without hardcoding it again")


def test_published_metrics_have_exactly_one_source_the_artifact():
    """Numbers a UI shows must come from the artifact that produced them.

    Per-archetype recall existed as prose in FOUR places -- risk_agent.py's
    docstring, the dashboard caption, train/analyze_risk_lead_time.py's summary,
    and the README. concentration_geopolitical drifted to 0.385 (5/13) against
    the shipped data's 0.462 (6/13); three copies were corrected and the fourth
    was missed, which is the argument for having none. Same shape as the battery
    MAE caption (21.2 over "6 cells" against a real 15.9 over 5) and the risk
    WATCH threshold (a silent 0.3 fallback against a real 0.1776) before it.
    """
    import joblib
    from conftest import DATA_PROCESSED_DIR, MODELS_DIR, skip_if_missing

    skip_if_missing(MODELS_DIR / "risk_model.pkl",
                    DATA_PROCESSED_DIR / "supply_chain_shipments.csv")
    import pandas as pd

    stored = (joblib.load(MODELS_DIR / "risk_model.pkl")
              .get("metrics", {}).get("archetype_recall_shipped_dataset"))
    assert stored, ("risk_model.pkl carries no archetype recall; the dashboard would have to "
                    "hardcode it again. Re-run train/train_risk_model.py")

    # and what it stores must be what the shipped dataset actually says
    df = pd.read_csv(DATA_PROCESSED_DIR / "supply_chain_shipments.csv")
    for name, sub in df[df.is_anomaly == 1].groupby("anomaly_type"):
        entry = stored[str(name)]
        assert entry["n"] == len(sub)
        assert entry["recall"] == round(float(sub.flagged.mean()), 3), (
            f"{name}: artifact says {entry['recall']}, the data says "
            f"{round(float(sub.flagged.mean()), 3)}")


def test_battery_and_risk_metadata_endpoints_serve_their_artifacts():
    """Both metadata endpoints exist so no caller has to hardcode a metric."""
    from conftest import MODELS_DIR, skip_if_missing

    skip_if_missing(MODELS_DIR / "risk_model.pkl", MODELS_DIR / "battery_rul_model.pkl")

    import joblib
    from app.agents.battery_agent import get_battery_agent
    from app.agents.risk_agent import get_risk_agent

    battery = joblib.load(MODELS_DIR / "battery_rul_model.pkl")
    assert get_battery_agent().metrics == battery["metrics"]
    assert get_battery_agent().training_strategy == battery["training_strategy"]

    risk = joblib.load(MODELS_DIR / "risk_model.pkl")
    assert get_risk_agent().watch_threshold == risk["watch_threshold"]
    assert get_risk_agent().metrics == risk["metrics"]


def test_dashboard_and_live_monitor_agree_about_the_same_truck():
    """Two pages of the same product must not disagree about one truck.

    dashboard/fleet_data.predict_battery omitted temp_battery_c /
    ambient_temp_c / discharge_current_a, so BatteryRULRequest supplied its
    defaults -- 25.0 degC against a recorded 32.1-41.1 degC across the three
    demo cells. temp_battery is a real LightGBM feature, so the dashboard was
    scoring its own fleet out of distribution: 6 of 8 trucks came back with a
    different RUL from the Live Fleet Monitor (which passes the recorded value),
    and 38 of 504 reachable simulator states got a different RISK BAND -- which
    propagates through /fleet/compound-risk into a different maintenance tier.
    """
    from conftest import DATA_PROCESSED_DIR, MODELS_DIR, skip_if_missing

    skip_if_missing(DATA_PROCESSED_DIR / "battery_features.csv",
                    MODELS_DIR / "battery_rul_model.pkl")

    import fleet_data
    import pandas as pd
    from app.agents.battery_agent import get_battery_agent
    from app.agents.telemetry_simulator import FLEET, ROLL_WINDOW_HISTORY

    agent = get_battery_agent()
    df = pd.read_csv(DATA_PROCESSED_DIR / "battery_features.csv")
    snapshots = {a["asset_id"]: a for a in fleet_data.build_fleet_snapshots()}

    disagreements = []
    for battery_id, cutoff, asset_id, _depot in FLEET:
        hist = df[df.battery_id == battery_id].sort_values("discharge_cycle_num")
        window = hist[hist.discharge_cycle_num <= cutoff].tail(ROLL_WINDOW_HISTORY)
        latest = window.iloc[-1]
        rated = round(float(hist["capacity"].iloc[:5].max()), 3)

        # exactly what the simulator sends
        simulator = agent.predict(
            capacity_history=window["capacity"].round(3).tolist(),
            temp_battery=float(latest["temp_battery"]),
            ambient_temp=float(latest["ambient_temp"]),
            discharge_current=float(latest["discharge_current"]),
            current_cycle_number=int(cutoff), rated_capacity_ah=rated, pack_kwh=21.3)

        # exactly what the dashboard sends
        snap = snapshots[asset_id]
        dashboard = agent.predict(
            capacity_history=snap["capacity_history"],
            temp_battery=snap["temp_battery_c"],
            ambient_temp=snap["ambient_temp_c"],
            discharge_current=snap["discharge_current_a"],
            current_cycle_number=snap["current_cycle_number"],
            rated_capacity_ah=snap["rated_capacity_ah"], pack_kwh=snap["pack_kwh"])

        for field in ("predicted_rul_cycles", "risk_band", "state_of_health_pct",
                      "eol_threshold_ah"):
            if simulator[field] != dashboard[field]:
                disagreements.append((asset_id, field, dashboard[field], simulator[field]))

    assert not disagreements, (
        f"the dashboard and the live monitor disagree about {len(disagreements)} field(s): "
        f"{disagreements[:4]}")


def test_the_demo_fleet_snapshot_carries_real_operating_conditions():
    """The temperature sent must be one that occurs in the data, not a default."""
    from conftest import DATA_PROCESSED_DIR, skip_if_missing

    skip_if_missing(DATA_PROCESSED_DIR / "battery_features.csv")

    import fleet_data
    import pandas as pd

    df = pd.read_csv(DATA_PROCESSED_DIR / "battery_features.csv")
    recorded = set(df["temp_battery"].round(6))
    for asset in fleet_data.build_fleet_snapshots():
        for key in ("temp_battery_c", "ambient_temp_c", "discharge_current_a"):
            assert key in asset, f"{asset['asset_id']} does not carry {key}"
        assert round(asset["temp_battery_c"], 6) in recorded, (
            f"{asset['asset_id']} would be scored at {asset['temp_battery_c']} degC, which "
            f"appears nowhere in battery_features.csv")


def test_served_version_matches_the_changelog():
    """`/` reports app.version to every caller, so it has to be the real one.

    It said 2.0.0 while CHANGELOG.md's top entry was 2.1.0 -- two answers to
    "which build is this?" in one repository.
    """
    import re

    from app import config

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^##\s+([0-9]+\.[0-9]+\.[0-9]+)", changelog, re.MULTILINE)
    assert headings, "no version heading found in CHANGELOG.md"
    assert config.VERSION == headings[0], (
        f"app.config.VERSION is {config.VERSION} but the newest CHANGELOG entry "
        f"is {headings[0]}")


def test_model_and_data_locations_have_one_owner():
    """Paths are resolved in app.config, not recomputed per module.

    Seven modules each did their own `Path(__file__).resolve().parents[N]`, with
    N differing between 2 and 3 depending on nesting depth -- the kind of thing
    that breaks silently the first time a file moves. It also made the model
    directory impossible to override without editing source, which a container
    mounting a model volume needs to do.
    """
    import re

    offenders = []
    for path in (ROOT / "backend").rglob("*.py"):
        if path.name == "config.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"Path\(__file__\)\.resolve\(\)\.parents\[", line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert not offenders, (
        "these modules resolve project paths themselves instead of importing "
        f"app.config: {offenders}")
