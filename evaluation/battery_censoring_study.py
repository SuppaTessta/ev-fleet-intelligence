"""Does handling censoring properly improve the RUL model?

Three treatments of the 14 right-censored NASA cells, evaluated identically:

  naive     treat the censored label as an observed event time. Wrong: it
            teaches the model that a healthy cell near the end of its recording
            is nearly dead.
  exclude   drop censored rows. Honest, and discards 73.1% of the data.
  censored  one-sided squared hinge -- penalise only under-prediction. Uses
            every row without asserting anything the data does not support.
            See backend/app/survival.py.

Scored by leave-one-battery-out over the 5 cells with a genuine end-of-life
event, since those are the only rows where ground-truth RUL exists. The two
baselines from train_battery_model.py are carried through so the comparison
stays anchored.

The result is conditional and worth reading before quoting a single number:
exclusion wins on median MAE (3.6 vs 13.9), but the censoring-aware model wins
on every fold where the held-out cell's condition group has only one other
observed cell, and loses on every fold where it has two.

Writes docs/eval/battery_censoring.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.survival import censored_l2_objective, concordance_index  # noqa: E402

from train.train_battery_model import (  # noqa: E402
    FEATURE_COLS,
    LGBM_PARAMS,
    TARGET_COL,
    build_dataset,
    linear_fade_baseline,
)

OUT_DIR = ROOT / "docs" / "eval"
STRATEGIES = ("naive", "exclude", "censored")


def _fit(train_df: pd.DataFrame, strategy: str):
    """Returns (model, offset). `offset` is the base score to add back to
    predictions -- non-zero only for the custom objective."""
    params = dict(LGBM_PARAMS)

    if strategy in ("naive", "exclude"):
        if strategy == "exclude":
            train_df = train_df[~train_df.censored]
        model = LGBMRegressor(**params)
        model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])
        return model, 0.0

    # Censoring-aware. The objective needs the mask aligned to the training
    # rows, so it is rebuilt per fold.
    #
    # init_score matters here and is easy to miss: supplying a custom objective
    # turns off LightGBM's boost_from_average, so the model starts from 0 and
    # has to spend its budget climbing to the mean before it can fit anything.
    # Seeding it with the mean of the UNCENSORED targets recovers that. Measured
    # on one fold: MAE 7.3 -> 6.9.
    params["objective"] = censored_l2_objective(train_df["censored"].to_numpy())
    offset = float(train_df.loc[~train_df.censored, TARGET_COL].mean())
    model = LGBMRegressor(**params)
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL],
              init_score=np.full(len(train_df), offset))
    return model, offset


def run() -> dict:
    full = build_dataset()
    genuine = sorted(full.loc[~full.censored, "battery_id"].unique())
    print(f"{len(full)} rows | {int(full.censored.sum())} censored "
          f"({100 * full.censored.mean():.1f}%) | {len(genuine)} cells with a real EOL event\n")

    per_fold: dict[str, list] = {s: [] for s in STRATEGIES}
    per_fold["baseline_fade"] = []
    per_fold["baseline_mean"] = []

    for held_out in genuine:
        train_df = full[full.battery_id != held_out]
        test_df = full[(full.battery_id == held_out) & (~full.censored)]
        y = test_df[TARGET_COL].to_numpy()

        for strategy in STRATEGIES:
            model, offset = _fit(train_df, strategy)
            pred = np.clip(model.predict(test_df[FEATURE_COLS]) + offset, 0, None)
            per_fold[strategy].append({
                "battery": held_out, "n": len(test_df),
                "mae": mean_absolute_error(y, pred),
                "r2": r2_score(y, pred) if y.var() > 0 else None,
                "cindex": concordance_index(y, pred, np.zeros(len(y), dtype=bool)),
                "y_true": y, "y_pred": pred,
            })

        fade = linear_fade_baseline(test_df)
        per_fold["baseline_fade"].append({
            "battery": held_out, "mae": mean_absolute_error(y, fade),
            "y_true": y, "y_pred": fade})
        mean_pred = np.full(len(y), train_df.loc[~train_df.censored, TARGET_COL].mean())
        per_fold["baseline_mean"].append({
            "battery": held_out, "mae": mean_absolute_error(y, mean_pred),
            "y_true": y, "y_pred": mean_pred})

    return _summarise(per_fold)


def _summarise(per_fold: dict) -> dict:
    summary = {}
    for name, folds in per_fold.items():
        maes = [f["mae"] for f in folds]
        pooled_true = np.concatenate([f["y_true"] for f in folds])
        pooled_pred = np.concatenate([f["y_pred"] for f in folds])
        entry = {
            "mae_mean": float(np.mean(maes)),
            "mae_sd": float(np.std(maes)),
            "pooled_r2": float(r2_score(pooled_true, pooled_pred)),
            "per_fold_mae": {f["battery"]: round(f["mae"], 2) for f in folds},
        }
        if "cindex" in folds[0]:
            cs = [f["cindex"] for f in folds if not np.isnan(f["cindex"])]
            entry["cindex_mean"] = float(np.mean(cs)) if cs else None
        summary[name] = entry

    print(f"{'strategy':16s} {'MAE (cycles)':>18s} {'pooled R2':>10s} {'C-index':>9s}")
    print("-" * 58)
    for name in ("baseline_mean", "baseline_fade", "naive", "exclude", "censored"):
        e = summary[name]
        c = e.get("cindex_mean")
        print(f"  {name:14s} {e['mae_mean']:8.1f} +/- {e['mae_sd']:5.1f} "
              f"{e['pooled_r2']:10.3f} {c if c is None else f'{c:9.3f}'}")

    # The MEAN is misleading here. With 5 folds and bimodal errors, `exclude`
    # scores 2.6-3.6 on three cells and 32-38 on the other two; its mean sits
    # between two clusters it never occupies. Median and per-fold wins describe
    # it better.
    medians = {s: float(np.median(list(summary[s]["per_fold_mae"].values())))
               for s in STRATEGIES}
    wins = dict.fromkeys(STRATEGIES, 0)
    for cell in summary["exclude"]["per_fold_mae"]:
        wins[min(STRATEGIES, key=lambda s: summary[s]["per_fold_mae"][cell])] += 1
    print(f"\n  median MAE   {medians}")
    print(f"  per-fold wins {wins}")

    scarcity = _scarcity_analysis(summary)
    summary["_verdict"] = {
        "best_by_median_mae": min(medians, key=medians.get),
        "median_mae": medians,
        "per_fold_wins": wins,
        "scarcity": scarcity,
    }
    return summary


def _scarcity_analysis(summary: dict) -> dict:
    """Does censoring-aware training help where observed data is scarce?

    This is the actual finding. Censored rows carry only an inequality, so they
    should matter most when there is little else for that operating condition --
    and the split is clean: the censoring-aware model wins on every fold with
    ONE same-group observed peer and loses on every fold with two.
    """
    from train.train_battery_model import BATTERY_META

    observed = set(summary["exclude"]["per_fold_mae"])
    rows = []
    for cell in observed:
        group = BATTERY_META[cell][0]
        peers = sum(1 for b, (g, _, _) in BATTERY_META.items()
                    if g == group and b in observed and b != cell)
        rows.append({
            "cell": cell, "group": group, "same_group_observed_peers": peers,
            "mae_exclude": summary["exclude"]["per_fold_mae"][cell],
            "mae_censored": summary["censored"]["per_fold_mae"][cell],
            "censored_is_better": (summary["censored"]["per_fold_mae"][cell]
                                    < summary["exclude"]["per_fold_mae"][cell]),
        })
    rows.sort(key=lambda r: r["same_group_observed_peers"])

    print("\n  Does censored data help where observed data is scarce?")
    print(f"    {'cell':8s}{'group':7s}{'peers':>7s}{'exclude':>10s}{'censored':>10s}  helps?")
    for r in rows:
        print(f"    {r['cell']:8s}{r['group']:7s}{r['same_group_observed_peers']:>7d}"
              f"{r['mae_exclude']:10.1f}{r['mae_censored']:10.1f}"
              f"  {'YES' if r['censored_is_better'] else 'no'}")

    scarce = [r for r in rows if r["same_group_observed_peers"] <= 1]
    rich = [r for r in rows if r["same_group_observed_peers"] > 1]
    clean = (all(r["censored_is_better"] for r in scarce)
             and not any(r["censored_is_better"] for r in rich))
    print(f"    -> censoring-aware helps on {sum(r['censored_is_better'] for r in scarce)}"
          f"/{len(scarce)} data-scarce folds and "
          f"{sum(r['censored_is_better'] for r in rich)}/{len(rich)} data-rich folds"
          f"{'  (clean separation)' if clean else ''}")
    return {"rows": rows, "clean_separation": bool(clean),
            "n_scarce": len(scarce), "n_rich": len(rich)}


if __name__ == "__main__":
    result = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "battery_censoring.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
