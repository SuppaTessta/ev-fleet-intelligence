"""Battery RUL training pipeline.

Predicts remaining useful life in discharge cycles from real NASA PCoE cycling
data -- 19 cells across 5 experimental groups (room temperature to 43C to 4C,
1A-4A discharge), not a single condition.

DATA QUALITY (NASA's own group READMEs flag these)
- Group 49-52 excluded entirely: NASA reports the run "stopped when experiment
  control software crashed", with several runs showing implausible capacity and
  voltage. That is a documented-unreliable run, not noise to filter.
- Groups 45-48 and 53-56 are included, but readings below 0.3 Ah are dropped
  first. Capacity does not physically jump 1.6 -> 0.03 -> 2.6 Ah cycle to cycle;
  that is a measurement dropout.
- B0052 excluded separately: only 4 usable discharge cycles.

CROSS-CONDITION POOLING
Groups have different rated capacities and different NASA stopping points, so
end of life is defined as 70% of EACH cell's own early-life capacity rather than
one fixed Ah number. Ambient temperature and nominal discharge current are
explicit features so the model learns how condition affects fade instead of
being confounded by it.

VALIDATION
Leave-one-battery-out over the 5 cells with a genuine end-of-life event -- see
find_eol_cycle for why only 5 of 19 qualify. The original stratified hold-out
survives as stratified_eval for comparison only: every one of its held-out cells
is right-censored, so it scores against labels that record when NASA stopped
rather than when the cell died.

WHAT SHIPS
The deployed artifact is trained under TRAINING_STRATEGY ("exclude", per
ADR-0005) and with LGBM_PARAMS. Both matter, and both were once inconsistent
with the evaluation: the final fit ran on all 19 cells while the published
figures came from exclusion, and it carried its own inline hyperparameters that
had lost `subsample_freq`, so bagging never ran on the shipped model.

    exclude (published, and shipped)  MAE 15.9  median  3.6  R2 0.467
    naive                             MAE 17.0  median 14.0  R2 0.769
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ROLL_WINDOW is imported rather than redeclared: it is the feature window the
# SERVING agent also uses, and a second copy here is exactly how the ddof=0/1
# mismatch in the same pair of files went unnoticed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.constants import ROLL_WINDOW  # noqa: E402

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "nasa_battery"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[1] / "backend" / "models"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
for d in (PROC_DIR, MODEL_DIR, DOCS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# battery_id -> (group, ambient_temp_c, nominal_discharge_current_a)
# sourced from NASA's own per-group README files, not guessed
BATTERY_META = {
    "B0005": ("G1", 24, 2), "B0006": ("G1", 24, 2), "B0007": ("G1", 24, 2), "B0018": ("G1", 24, 2),
    "B0029": ("G3", 43, 4), "B0030": ("G3", 43, 4), "B0031": ("G3", 43, 4), "B0032": ("G3", 43, 4),
    "B0033": ("G4", 24, 4), "B0034": ("G4", 24, 4), "B0036": ("G4", 24, 2),
    "B0045": ("G7", 4, 1), "B0046": ("G7", 4, 1), "B0047": ("G7", 4, 1), "B0048": ("G7", 4, 1),
    "B0053": ("G9", 4, 2), "B0054": ("G9", 4, 2), "B0055": ("G9", 4, 2), "B0056": ("G9", 4, 2),
}
BATTERIES = list(BATTERY_META.keys())
HELD_OUT_PER_GROUP = ["B0007", "B0032", "B0034", "B0048", "B0056"]  # one per group, never trained on

MIN_PLAUSIBLE_CAPACITY_AH = 0.3  # sensor-fault floor, see module docstring
EOL_SOH_FRACTION = 0.70          # EOL = 70% of THIS battery's own initial capacity
BREAK_IN_CYCLES = 5              # ignore for EOL detection -- see find_eol_cycle
EOL_CONFIRM_CYCLES = 3           # consecutive readings needed to call EOL
EOL_SUSTAIN_FRACTION = 0.5       # ...and this share of later readings must stay below it

# How the deployed artifact treats right-censored rows. "exclude" is the
# decision recorded in docs/adr/0005-censored-rul-labels.md, and it is what
# every published battery number is measured under.
TRAINING_STRATEGY = "exclude"

# subsample without subsample_freq is a silent no-op in LightGBM -- bagging never
# runs, so the model was not regularised the way the code claimed.
LGBM_PARAMS = {
    "n_estimators": 300, "max_depth": 4, "learning_rate": 0.03,
    "min_child_samples": 30, "num_leaves": 15,
    "reg_alpha": 0.5, "reg_lambda": 0.5,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "random_state": 42, "verbosity": -1,
}
FEATURE_COLS = [
    "discharge_cycle_num", "capacity", "capacity_pct_initial",
    "roll_mean_capacity", "roll_std_capacity", "fade_rate", "cumulative_fade",
    "temp_battery", "ambient_temp", "discharge_current",
]
TARGET_COL = "RUL"


def load_battery(battery_id: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{battery_id}_discharge.csv")
    df = df[df["capacity"] >= MIN_PLAUSIBLE_CAPACITY_AH].copy()  # drop sensor-fault readings
    df = df.sort_values("cycle").reset_index(drop=True)
    df["discharge_cycle_num"] = np.arange(1, len(df) + 1)  # re-contiguous after dropping bad rows
    df["battery_id"] = battery_id
    group, amb_temp, current = BATTERY_META[battery_id]
    df["group"] = group
    df["ambient_temp"] = amb_temp
    df["discharge_current"] = current
    return df


def find_eol_cycle(capacity: np.ndarray, cycle_num: np.ndarray, threshold: float):
    """First cycle at which the pack is genuinely at end of life.

    Returns (eol_cycle_num, censored). `censored` means the cell never reached
    end of life inside its recorded window, so the returned cycle is where NASA
    stopped and the RUL derived from it is a LOWER BOUND, not an observation.

    Three conditions, each of which a naive `first index below threshold` misses:

    1. Skip the break-in window. Several cells report an anomalously low first
       reading -- B0034 reads 0.746 Ah at cycle 1 against 1.3-1.4 Ah for cycles
       2-20 -- which would pin RUL to 0 for that cell's entire life.
    2. Require EOL_CONFIRM_CYCLES consecutive readings. Capacity is noisy and one
       sample below a threshold is not death.
    3. Require the crossing to be SUSTAINED. B0033 crosses at cycle 138, then
       spends 94.3% of its remaining readings back above the threshold and ends
       at 99.5% of initial capacity. End of life is permanent; a cell that comes
       back was never at end of life.

    With all three, only 5 of 19 cells have a genuine EOL event. The other 14 are
    right-censored. See ADR-0005.
    """
    below = capacity <= threshold
    below[:BREAK_IN_CYCLES] = False   # break-in dips are not end of life

    run = 0
    for i, is_below in enumerate(below):
        run = run + 1 if is_below else 0
        if run < EOL_CONFIRM_CYCLES:
            continue
        start = i - EOL_CONFIRM_CYCLES + 1
        # does it STAY dead? a cell that comes back was never at end of life
        if below[start:].mean() >= EOL_SUSTAIN_FRACTION:
            return int(cycle_num[start]), False
        run = 0  # recovered -- keep looking for a later, permanent crossing
    return int(cycle_num[-1]), True


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # robust to a single anomalous first reading (found in B0034: cycle 1 read
    # 0.75 Ah while cycles 2-20 all read 1.3-1.4 Ah) — use the max of the first
    # 5 cycles rather than literally cycle 1 as the "initial capacity" reference
    initial_capacity = df["capacity"].iloc[:5].max()
    df["capacity_pct_initial"] = df["capacity"] / initial_capacity
    df["roll_mean_capacity"] = df["capacity"].rolling(ROLL_WINDOW, min_periods=1).mean()
    df["roll_std_capacity"] = df["capacity"].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)
    df["fade_rate"] = df["capacity"].diff(ROLL_WINDOW) / ROLL_WINDOW
    df["fade_rate"] = df["fade_rate"].fillna(0)
    df["cumulative_fade"] = initial_capacity - df["capacity"]

    eol_threshold_ah = initial_capacity * EOL_SOH_FRACTION  # per-battery, not fixed
    eol_cycle_num, censored = find_eol_cycle(
        df["capacity"].values, df["discharge_cycle_num"].values, eol_threshold_ah)
    df["RUL"] = (eol_cycle_num - df["discharge_cycle_num"]).clip(lower=0)
    df["eol_threshold_ah"] = eol_threshold_ah
    df["censored"] = censored
    df["eol_cycle_num"] = eol_cycle_num
    return df


def build_dataset() -> pd.DataFrame:
    frames = [engineer_features(load_battery(b)) for b in BATTERIES]
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(PROC_DIR / "battery_features.csv", index=False)
    return full


BASELINE_WINDOW = 20   # trailing cycles used to estimate the fade slope


def linear_fade_baseline(df: pd.DataFrame, window: int = BASELINE_WINDOW) -> np.ndarray:
    """Physical baseline: least-squares fade slope over a trailing window,
    extrapolated to the EOL threshold.

        RUL = (capacity_now - eol_threshold) / fade_per_cycle

    This exists because the RUL target is by construction a straight line of
    slope -1, so anything tracking the fade trend should score well. Publishing a
    learned model without showing it beats this proves nothing.

    The slope is fitted by regression rather than read off the `fade_rate`
    feature: that is a 5-cycle difference on a noisy signal, and dividing
    headroom by a noisy near-zero denominator produces wild extrapolations,
    which would make this a strawman rather than a real baseline.
    """
    cap = df["capacity"].values
    cyc = df["discharge_cycle_num"].values.astype(float)
    thr = df["eol_threshold_ah"].values
    out = np.empty(len(df))

    for i in range(len(df)):
        lo = max(0, i - window + 1)
        xs, ys = cyc[lo:i + 1], cap[lo:i + 1]
        if len(xs) >= 3:
            slope = np.polyfit(xs, ys, 1)[0]
        else:
            slope = 0.0
        fade = -slope                                   # positive while degrading
        out[i] = (cap[i] - thr[i]) / fade if fade > 1e-5 else 500.0
    # Cap at roughly the longest life observed: an unbounded extrapolation off a
    # near-flat segment is a divide-by-zero, not a prediction.
    return np.clip(np.nan_to_num(out, nan=500.0, posinf=500.0), 0, 500)


def leave_one_battery_out_eval(full: pd.DataFrame) -> tuple:
    """Leave-one-battery-out CV over the cells with a genuine EOL event.

    Censored cells are excluded from both training and scoring: their RUL is a
    lower bound, and treating a lower bound as an observation is what produced
    the original problem. TRAINING_STRATEGY applies the same rule to the
    deployed artifact, so this evaluation describes the model that ships.

    A censoring-aware alternative exists and was measured -- backend/app/
    survival.py implements a one-sided squared hinge, and
    evaluation/battery_censoring_study.py compares all three treatments:

        strategy   median MAE   per-fold wins
        naive          14.2          1/5
        exclude         3.6          3/5
        censored       13.9          1/5

    Exclusion wins overall, so it stays the default. The interesting result is
    conditional and the separation is clean: censoring-aware wins on both folds
    where the held-out cell's condition group has only ONE other observed cell,
    and loses on all three where it has two. A censored row says only "at least
    this many cycles remained" -- weak evidence, worth having when there is
    nothing better for that operating condition. Not switched on by default,
    because deriving a switching policy from five cells would fit the noise this
    study exists to expose.
    """
    genuine = sorted(full.loc[~full.censored, "battery_id"].unique())
    data = full[full.battery_id.isin(genuine)]
    results = {}

    for held_out in genuine:
        train_df = data[data.battery_id != held_out]
        test_df = data[data.battery_id == held_out]
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

        preds = np.clip(model.predict(test_df[FEATURE_COLS]), 0, None)
        base = linear_fade_baseline(test_df)
        y = test_df[TARGET_COL].values

        results[held_out] = {
            "group": BATTERY_META[held_out][0],
            "n": len(test_df),
            "mae": mean_absolute_error(y, preds),
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "r2": r2_score(y, preds) if test_df[TARGET_COL].var() > 0 else None,
            "baseline_mae": mean_absolute_error(y, base),
            "y_true": y, "y_pred": preds, "y_base": base,
            "cycle": test_df["discharge_cycle_num"].values,
        }
        r = results[held_out]
        r2s = f"{r['r2']:.3f}" if r["r2"] is not None else "undefined"
        print(f"[held out {held_out} ({r['group']}, n={r['n']})] "
              f"MAE={r['mae']:6.1f} | RMSE={r['rmse']:6.1f} | R2={r2s:>9s} "
              f"| linear-fade baseline MAE={r['baseline_mae']:6.1f}")

    return results, genuine


def stratified_eval(full: pd.DataFrame) -> dict:
    train_df = full[~full.battery_id.isin(HELD_OUT_PER_GROUP)]
    results = {}
    # LGBM_PARAMS, so the legacy comparison uses the same regularisation as
    # everything it is compared against.
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

    for held_out in HELD_OUT_PER_GROUP:
        test_df = full[full.battery_id == held_out]
        preds = np.clip(model.predict(test_df[FEATURE_COLS]), 0, None)
        mae = mean_absolute_error(test_df[TARGET_COL], preds)
        rmse = np.sqrt(mean_squared_error(test_df[TARGET_COL], preds))

        # R2 is mathematically undefined when the true values have zero
        # variance (nothing to explain) — this happens when a held-out
        # battery's ENTIRE recorded window is already past the EOL
        # threshold (RUL=0 throughout). Report that honestly instead of
        # sklearn's default 0.0, which reads as "no better than baseline"
        # when the real issue is there's no baseline variance to begin with.
        r2_defined = test_df[TARGET_COL].var() > 0
        r2 = r2_score(test_df[TARGET_COL], preds) if r2_defined else None

        group = BATTERY_META[held_out][0]
        results[held_out] = {"group": group, "mae": mae, "rmse": rmse, "r2": r2, "r2_defined": r2_defined,
                              "y_true": test_df[TARGET_COL].values, "y_pred": preds,
                              "cycle": test_df["discharge_cycle_num"].values}
        r2_str = f"{r2:.3f}" if r2_defined else "undefined (test set has zero RUL variance — see note below)"
        print(f"[Held out: {held_out} ({group})] MAE={mae:.1f} cycles | RMSE={rmse:.1f} cycles | R2={r2_str}")
    return results, model


def plot_eval(results: dict):
    fig, axes = plt.subplots(1, len(HELD_OUT_PER_GROUP), figsize=(19, 4), sharey=True)
    for ax, battery in zip(axes, HELD_OUT_PER_GROUP):
        r = results[battery]
        ax.plot(r["cycle"], r["y_true"], label="Actual RUL", linewidth=2)
        ax.plot(r["cycle"], r["y_pred"], label="Predicted RUL", linestyle="--")
        ax.set_title(f"{battery} ({r['group']})\nMAE={r['mae']:.1f} cyc")
        ax.set_xlabel("Discharge cycle")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("RUL (cycles)")
    fig.suptitle("Battery RUL — held out one battery per condition group (NASA PCoE, 5 groups)")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "battery_model_eval.png", dpi=150)
    plt.close(fig)


def plot_capacity_curves(full: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"G1": "#1A2E2A", "G3": "#E5484D", "G4": "#F2A93B", "G7": "#3B82F6", "G9": "#00A676"}
    for battery in BATTERIES:
        d = full[full.battery_id == battery]
        group = BATTERY_META[battery][0]
        ax.plot(d["discharge_cycle_num"], d["capacity_pct_initial"] * 100,
                color=colors[group], alpha=0.7, linewidth=1.2,
                label=group if battery == [b for b in BATTERIES if BATTERY_META[b][0] == group][0] else None)
    ax.axhline(70, color="red", linestyle=":", label="70% SoH (EOL)")
    ax.set_xlabel("Discharge cycle")
    ax.set_ylabel("State of Health (% of initial capacity)")
    ax.set_title("Capacity fade, normalized to SoH — 19 batteries, 5 conditions (NASA PCoE)")
    ax.legend(title="Group (temp, current)")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "battery_capacity_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"Loading {len(BATTERIES)} batteries across {len({v[0] for v in BATTERY_META.values()})} "
          f"condition groups (excluded: B0050, B0052, B0035, B0037, and all non-CC-protocol groups)\n")
    full = build_dataset()
    print(f"Dataset: {len(full)} cycle-level rows (after sensor-fault filtering)\n")

    censored_ids = sorted(full.loc[full.censored, "battery_id"].unique())
    genuine_ids = sorted(full.loc[~full.censored, "battery_id"].unique())
    print("--- Label audit ---")
    print(f"  cells with a genuine EOL event : {len(genuine_ids):2d}  {genuine_ids}")
    print(f"  right-censored (never hit EOL) : {len(censored_ids):2d}  {censored_ids}")
    print(f"  rows: {(~full.censored).sum()} genuine / {full.censored.sum()} censored "
          f"({100 * full.censored.mean():.1f}% censored)\n")

    print("--- Leave-one-battery-out CV over the genuine-EOL cells ---")
    results, genuine = leave_one_battery_out_eval(full)
    maes = [r["mae"] for r in results.values()]
    base_maes = [r["baseline_mae"] for r in results.values()]
    defined = [r["r2"] for r in results.values() if r["r2"] is not None]
    print(f"\n  model    MAE across {len(maes)} folds: {np.mean(maes):.1f} +/- {np.std(maes):.1f} cycles")
    print(f"  baseline MAE across {len(maes)} folds: {np.mean(base_maes):.1f} +/- {np.std(base_maes):.1f} cycles"
          f"   (linear fade extrapolation)")
    wins = sum(1 for r in results.values() if r["mae"] < r["baseline_mae"])
    print(f"  model beats the baseline on {wins}/{len(results)} folds")
    if defined:
        print(f"  mean R2 over the {len(defined)} folds where it is defined: {np.mean(defined):.3f}")
    all_t = np.concatenate([r["y_true"] for r in results.values()])
    all_p = np.concatenate([r["y_pred"] for r in results.values()])
    all_b = np.concatenate([r["y_base"] for r in results.values()])
    print(f"  pooled R2  model={r2_score(all_t, all_p):.3f}  baseline={r2_score(all_t, all_b):.3f}")

    print("\n--- (legacy) original stratified hold-out, for comparison ---")
    results_legacy, model = stratified_eval(full)
    avg_mae = np.mean([r["mae"] for r in results_legacy.values()])
    defined_r2 = [r["r2"] for r in results_legacy.values() if r["r2_defined"]]
    n_undefined = sum(1 for r in results_legacy.values() if not r["r2_defined"])
    legacy_true = np.concatenate([r["y_true"] for r in results_legacy.values()])
    legacy_pred = np.concatenate([r["y_pred"] for r in results_legacy.values()])
    print(f"  MAE={avg_mae:.1f} cycles | pooled R2={r2_score(legacy_true, legacy_pred):.3f}"
          f" | R2 undefined for {n_undefined} of {len(results_legacy)} held-out cells")
    print("  NOTE: this is the split that produced the published 0.740. Every one of its five")
    print("  held-out cells is right-censored, so it scores predictions against labels that")
    print("  encode when NASA stopped recording, not when the cell reached end of life.")

    plot_eval(results_legacy)
    plot_capacity_curves(full)

    # The artifact that actually gets served: trained under TRAINING_STRATEGY,
    # on the same rows and hyperparameters the leave-one-battery-out numbers
    # above were measured with.
    if TRAINING_STRATEGY == "exclude":
        deploy_df = full[~full.censored]
    elif TRAINING_STRATEGY == "naive":
        deploy_df = full
    else:
        raise SystemExit(f"unsupported TRAINING_STRATEGY {TRAINING_STRATEGY!r}; "
                         f"expected 'exclude' or 'naive' (see ADR-0005)")

    deploy_cells = sorted(deploy_df.battery_id.unique())
    print()
    print(f"--- Training final model, strategy={TRAINING_STRATEGY!r}: "
          f"{len(deploy_df)} rows / {len(deploy_cells)} cells (for deployment) ---")
    print(f"    cells: {deploy_cells}")

    final_model = LGBMRegressor(**LGBM_PARAMS)
    final_model.fit(deploy_df[FEATURE_COLS], deploy_df[TARGET_COL])

    # Metrics travel with the model, for the same reason the risk agent's
    # watch_threshold does: a published number that lives only in a README
    # cannot be checked against the artifact a service actually loaded.
    metrics = {
        "protocol": "leave-one-battery-out over the cells with a genuine EOL event",
        "training_strategy": TRAINING_STRATEGY,
        "n_folds": len(maes),
        "mae_mean": float(np.mean(maes)),
        "mae_sd": float(np.std(maes)),
        "mae_median": float(np.median(maes)),
        "pooled_r2": float(r2_score(all_t, all_p)),
        "per_fold_mae": {b: round(float(r["mae"]), 2) for b, r in results.items()},
        "baseline_fade_mae_mean": float(np.mean(base_maes)),
        "beats_baseline_folds": f"{wins}/{len(results)}",
    }
    joblib.dump({"model": final_model, "feature_cols": FEATURE_COLS,
                 "eol_soh_fraction": EOL_SOH_FRACTION,
                 "training_strategy": TRAINING_STRATEGY,
                 "trained_on_cells": deploy_cells,
                 "n_training_rows": int(len(deploy_df)),
                 "censored_cells_excluded": (sorted(censored_ids)
                                             if TRAINING_STRATEGY == "exclude" else []),
                 "metrics": metrics},
                MODEL_DIR / "battery_rul_model.pkl")
    print(f"Saved model to {MODEL_DIR / 'battery_rul_model.pkl'}")
    print(f"  published alongside it: MAE {metrics['mae_mean']:.1f} +/- "
          f"{metrics['mae_sd']:.1f} (median {metrics['mae_median']:.1f}), "
          f"pooled R2 {metrics['pooled_r2']:.3f}, beats fade baseline "
          f"{metrics['beats_baseline_folds']}")
    print(f"Saved plots to {DOCS_DIR}")
