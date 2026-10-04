"""Supply Chain Risk training pipeline.

No public dataset fits "EV battery supply chain fraud", so this one is
synthetic. What makes it a legitimate demo rather than noise: the generator is
grounded in documented supply-chain facts (China's dominance in cell and
rare-earth processing, DRC's dominance and ESG risk in cobalt, Australia and
Chile as lower-risk lithium sources), and anomalies are injected as five named
archetypes so evaluation means something -- precision and recall against known
injected anomalies.

Archetype 5 (gradual_quality_decline) exists because of a real finding: testing
detection lead time against the first four showed the model was LATE on slow
quality drift, having only ever seen sudden spikes. Trend features plus
gradual-decline training examples close that gap; see
train/analyze_risk_lead_time.py.

Outputs: data/processed/supply_chain_shipments.csv,
backend/models/risk_model.pkl, docs/risk_model_eval.png.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[1] / "backend" / "models"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
for d in (PROC_DIR, MODEL_DIR, DOCS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
# Isolation Forest's `contamination` IS the decision threshold -- it sets the
# quantile above which a point is called anomalous. Setting it from the true
# prevalence would be label leakage. It is kept at a round 0.09 as a DEPLOYMENT
# choice (a team tolerating ~9% of shipments flagged for review), and the
# headline metric is AUC-PR, which is threshold-free.
CONTAMINATION = 0.09
TEST_SIZE = 0.30
N_SPLITS = 20  # repeated stratified splits; one split on ~24 positives is noise

# material -> (region, geopolitical_risk 0-1, baseline unit price, price volatility)
MATERIALS = {
    "Lithium Carbonate":  ("Australia", 0.20, 14000, 0.10),
    "Cobalt":             ("DR Congo",  0.75, 33000, 0.18),
    "Nickel Sulfate":     ("Indonesia", 0.45, 21000, 0.12),
    "NMC Cells":          ("China",     0.55, 95,    0.08),
    "LFP Cells":          ("China",     0.55, 72,    0.08),
    "Rare Earth Magnets": ("China",     0.65, 58000, 0.15),
    "Copper Wire":        ("Chile",     0.25, 9200,  0.09),
    "Battery-Grade Aluminium": ("India", 0.15, 2600,  0.07),
    "Semiconductor Chips (BMS)": ("South Korea", 0.40, 3.2, 0.10),
    "Battery Management Systems": ("India", 0.20, 145, 0.06),
}
# explicit abbreviations — guaranteed unique, unlike truncating material names
# (which silently collided: "Battery-Grade Aluminium" and "Battery Management
# Systems" both truncate to "BATT")
MATERIAL_ABBREV = {
    "Lithium Carbonate": "LITH", "Cobalt": "COBALT", "Nickel Sulfate": "NICK",
    "NMC Cells": "NMC", "LFP Cells": "LFP", "Rare Earth Magnets": "REM",
    "Copper Wire": "COPPER", "Battery-Grade Aluminium": "ALUM",
    "Semiconductor Chips (BMS)": "CHIP", "Battery Management Systems": "BMS",
}
N_SUPPLIERS_PER_MATERIAL = 4
SHIPMENTS_PER_SUPPLIER = 25
TREND_WINDOW = 3  # shipments back, for computing this supplier's own recent trend

FEATURE_COLS = [
    "lead_time_deviation_pct", "price_deviation_pct", "reject_rate_pct",
    "days_since_last_audit", "single_sourced", "geopolitical_risk",
    "supplier_concentration_pct", "volume_deviation_pct",
    "reject_rate_trend", "price_trend",
]


def generate_suppliers():
    suppliers = []
    for material, (region, geo_risk, base_price, price_vol) in MATERIALS.items():
        # baseline market share split across this material's suppliers (Dirichlet
        # gives a realistic mix: some dominant suppliers, some minor ones)
        shares = RNG.dirichlet(np.ones(N_SUPPLIERS_PER_MATERIAL) * 1.3)
        # single-sourcing is decided per supplier below (share > 0.6). A
        # material-level variable was computed here and never read.
        for i, share in enumerate(shares):
            suppliers.append({
                "supplier_id": f"{MATERIAL_ABBREV[material]}-SUP{i+1}",
                "material": material, "region": region,
                "geopolitical_risk": geo_risk,
                "base_price": base_price, "price_vol": price_vol,
                "base_lead_time": RNG.uniform(18, 45),
                "base_reject_rate": RNG.uniform(0.5, 2.5),
                "market_share": share,
                "single_sourced": int(share > 0.6),
            })
    return pd.DataFrame(suppliers)


def generate_shipments(suppliers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in suppliers.iterrows():
        for j in range(SHIPMENTS_PER_SUPPLIER):
            lead_time = RNG.normal(s.base_lead_time, s.base_lead_time * 0.10)
            price = RNG.normal(s.base_price, s.base_price * s.price_vol * 0.5)
            reject_rate = max(0, RNG.normal(s.base_reject_rate, 0.5))
            volume = RNG.normal(1000, 100)
            rows.append({
                "shipment_seq": j, "shipment_id": f"{s.supplier_id}-{j+1:03d}",
                "supplier_id": s.supplier_id, "material": s.material, "region": s.region,
                "lead_time_days": lead_time, "unit_price": price, "reject_rate_pct": reject_rate,
                "order_volume_units": volume,
                "days_since_last_audit": RNG.uniform(5, 120),
                "single_sourced": s.single_sourced, "geopolitical_risk": s.geopolitical_risk,
                "supplier_concentration_pct": s.market_share * 100,
                "base_lead_time": s.base_lead_time, "base_price": s.base_price,
                "is_anomaly": 0, "anomaly_type": "none",
            })
    df = pd.DataFrame(rows)

    # ---- archetypes 1-4: single-shipment spikes (unchanged) ----
    n_spike_anomalies = int(len(df) * CONTAMINATION * 0.7)  # ~70% of the anomaly budget
    per_type = n_spike_anomalies // 4
    idx_pool = RNG.choice(df.index, size=n_spike_anomalies, replace=False)
    types = (["price_spike_late_delivery"] * per_type + ["quality_drift"] * per_type +
             ["concentration_geopolitical"] * per_type +
             ["stale_audit_volume_spike"] * (n_spike_anomalies - 3 * per_type))

    for idx, atype in zip(idx_pool, types):
        row = df.loc[idx]
        if atype == "price_spike_late_delivery":
            df.loc[idx, "unit_price"] = row.base_price * RNG.uniform(1.4, 1.9)
            df.loc[idx, "lead_time_days"] = row.base_lead_time * RNG.uniform(2.0, 3.2)
        elif atype == "quality_drift":
            df.loc[idx, "reject_rate_pct"] = RNG.uniform(15, 30)
        elif atype == "concentration_geopolitical":
            df.loc[idx, "supplier_concentration_pct"] = RNG.uniform(75, 95)
            df.loc[idx, "geopolitical_risk"] = min(0.95, row.geopolitical_risk + RNG.uniform(0.2, 0.3))
        elif atype == "stale_audit_volume_spike":
            df.loc[idx, "days_since_last_audit"] = RNG.uniform(180, 360)
            df.loc[idx, "order_volume_units"] = row.order_volume_units * RNG.uniform(2.5, 4.0)
        df.loc[idx, "is_anomaly"] = 1
        df.loc[idx, "anomaly_type"] = atype

    # ---- archetype 5: gradual_quality_decline (multi-shipment ramp, NOT a
    # single-row spike) — a supplier whose reject rate creeps up over ~8
    # consecutive shipments toward a moderate-bad level, the pattern a real
    # early-warning system needs to catch before it becomes obvious ----
    n_gradual_suppliers = max(1, int(len(suppliers) * 0.08))
    chosen_suppliers = RNG.choice(suppliers.supplier_id.unique(), size=n_gradual_suppliers, replace=False)
    for supplier_id in chosen_suppliers:
        sup_rows = df[df.supplier_id == supplier_id].sort_values("shipment_seq")
        start = RNG.integers(5, max(6, len(sup_rows) - 9))
        ramp_len = RNG.integers(6, 9)
        ramp_indices = sup_rows.iloc[start:start + ramp_len].index
        target_reject = RNG.uniform(9, 13)  # moderate-bad, not extreme like quality_drift's 15-30
        base_reject = df.loc[ramp_indices[0], "reject_rate_pct"]
        for step, idx in enumerate(ramp_indices):
            progress = (step + 1) / len(ramp_indices)
            df.loc[idx, "reject_rate_pct"] = base_reject + progress * (target_reject - base_reject) + RNG.normal(0, 0.4)
            df.loc[idx, "is_anomaly"] = 1
            df.loc[idx, "anomaly_type"] = "gradual_quality_decline"

    # ---- trend features: this supplier's own recent trajectory, not just
    # the absolute value — this is what lets the model catch a ramp before
    # it reaches an extreme absolute level ----
    df = df.sort_values(["supplier_id", "shipment_seq"]).reset_index(drop=True)
    df["reject_rate_trend"] = df.groupby("supplier_id")["reject_rate_pct"] \
        .transform(lambda s: s.diff(TREND_WINDOW) / TREND_WINDOW).fillna(0)
    df["price_trend"] = df.groupby("supplier_id")["unit_price"] \
        .transform(lambda s: s.diff(TREND_WINDOW) / TREND_WINDOW).fillna(0)

    df["lead_time_deviation_pct"] = 100 * (df.lead_time_days - df.base_lead_time) / df.base_lead_time
    df["price_deviation_pct"] = 100 * (df.unit_price - df.base_price) / df.base_price
    df["volume_deviation_pct"] = 100 * (df.order_volume_units - 1000) / 1000
    return df.drop(columns=["base_lead_time", "base_price"])


def train_and_evaluate(df: pd.DataFrame):
    """Fit on a training split, report on a held-out one.

    Isolation Forest is unsupervised, so the split is not about preventing label
    memorisation -- it never sees labels. It is about the METRIC: a score
    distribution fitted to a sample and evaluated on that same sample flatters
    itself, and the min/max used to normalise risk_score are themselves fitted
    quantities.

    AUC-PR is the headline because it needs no threshold. Precision, recall and
    F1 at the deployment threshold are reported alongside, clearly labelled as
    threshold-dependent.
    """
    # One held-out split is not enough: 30% of 79 anomalies is ~24 positives, and
    # metrics on 24 positives swing wildly with the seed. Repeat and report the
    # spread.
    aucs, precs, recs, f1s = [], [], [], []
    for seed in range(N_SPLITS):
        tr, te = train_test_split(df, test_size=TEST_SIZE,
                                   stratify=df.is_anomaly, random_state=seed)
        sc = StandardScaler().fit(tr[FEATURE_COLS])
        m = IsolationForest(n_estimators=300, contamination=CONTAMINATION,
                            random_state=seed).fit(sc.transform(tr[FEATURE_COLS]))
        tr_raw = m.score_samples(sc.transform(tr[FEATURE_COLS]))
        lo, hi = float(tr_raw.min()), float(tr_raw.max())
        te_x = sc.transform(te[FEATURE_COLS])
        te_risk = np.clip(1 - (m.score_samples(te_x) - lo) / (hi - lo), 0, 1)
        te_flag = (m.predict(te_x) == -1).astype(int)
        aucs.append(average_precision_score(te.is_anomaly, te_risk))
        pp, rr, ff, _ = precision_recall_fscore_support(
            te.is_anomaly, te_flag, average="binary", zero_division=0)
        precs.append(pp); recs.append(rr); f1s.append(ff)

    print(f"\n--- HELD-OUT over {N_SPLITS} stratified {int((1-TEST_SIZE)*100)}/"
          f"{int(TEST_SIZE*100)} splits (mean +/- sd) ---")
    print(f"  AUC-PR    {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}   "
          f"(prevalence baseline {df.is_anomaly.mean():.3f})")
    print(f"  precision {np.mean(precs):.2f} +/- {np.std(precs):.2f}   "
          f"recall {np.mean(recs):.2f} +/- {np.std(recs):.2f}   "
          f"F1 {np.mean(f1s):.2f} +/- {np.std(f1s):.2f}")
    print(f"  AUC-PR range across splits: {min(aucs):.3f} to {max(aucs):.3f} "
          f"-- the spread is why one split is not a result")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df.is_anomaly, random_state=42)
    print(f"\nreference split for the shipped artifact: {len(train_df)} train / "
          f"{len(test_df)} held out ({int(test_df.is_anomaly.sum())} true anomalies)")

    scaler = StandardScaler().fit(train_df[FEATURE_COLS])
    model = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=42)
    model.fit(scaler.transform(train_df[FEATURE_COLS]))

    # normalisation bounds come from TRAIN only -- using the full set would let
    # test-set extremes define the scale the test set is then scored against
    train_raw = model.score_samples(scaler.transform(train_df[FEATURE_COLS]))
    score_min, score_max = float(train_raw.min()), float(train_raw.max())

    def score(frame):
        x = scaler.transform(frame[FEATURE_COLS])
        raw = model.score_samples(x)
        risk = np.clip(1 - (raw - score_min) / (score_max - score_min), 0, 1)
        return risk, (model.predict(x) == -1).astype(int)

    test_risk, test_flag = score(test_df)
    p, r, f1, _ = precision_recall_fscore_support(
        test_df.is_anomaly, test_flag, average="binary", zero_division=0)
    auc_pr = average_precision_score(test_df.is_anomaly, test_risk)
    auc_roc = roc_auc_score(test_df.is_anomaly, test_risk)
    baseline = test_df.is_anomaly.mean()

    print("\n--- HELD-OUT performance (the honest numbers) ---")
    print(f"  AUC-PR   {auc_pr:.3f}   (random baseline = prevalence = {baseline:.3f}, "
          f"so {auc_pr / baseline:.1f}x better than chance)")
    print(f"  AUC-ROC  {auc_roc:.3f}")
    print(f"  at the {CONTAMINATION:.0%} deployment threshold: "
          f"precision {p:.2f} | recall {r:.2f} | F1 {f1:.2f}")

    in_risk, in_flag = score(train_df)
    ip, ir, if1, _ = precision_recall_fscore_support(
        train_df.is_anomaly, in_flag, average="binary", zero_division=0)
    print(f"  (in-sample, for comparison: precision {ip:.2f} | recall {ir:.2f} | F1 {if1:.2f}, "
          f"AUC-PR {average_precision_score(train_df.is_anomaly, in_risk):.3f})")

    # per-archetype recall on the held-out split
    print("\n  held-out recall by archetype:")
    scored_test = test_df.assign(flagged=test_flag)
    for archetype, sub in scored_test[scored_test.is_anomaly == 1].groupby("anomaly_type"):
        print(f"    {archetype:28s} n={len(sub):3d}  recall={sub.flagged.mean():.3f}")

    # score the FULL frame for the persisted CSV the dashboard reads
    all_risk, all_flag = score(df)
    df = df.assign(risk_score=all_risk, flagged=all_flag)

    # WATCH threshold, computed here and persisted into the artifact so the
    # decision boundary travels with the model rather than being recomputed
    # from whatever data happens to be on the serving host.
    watch_threshold = float(np.quantile(all_risk, 0.75))
    print(f"\n  WATCH threshold (75th pct of the scored shipment set): {watch_threshold:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df[df.is_anomaly == 0].risk_score, bins=30, alpha=0.6, label="Normal", color="#00A676")
    axes[0].hist(df[df.is_anomaly == 1].risk_score, bins=30, alpha=0.6, label="Injected anomaly", color="#E5484D")
    axes[0].axvline(watch_threshold, color="#F2A93B", linestyle="--", label="WATCH threshold")
    axes[0].axvline(df[df.flagged == 1].risk_score.min(), color="black", linestyle="--", label="CRITICAL flag")
    axes[0].set_xlabel("Risk score"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"Risk score separation (held-out AUC-PR {auc_pr:.3f})")
    axes[0].legend(fontsize=8)

    conc = df.groupby("material")["supplier_concentration_pct"].max().sort_values()
    axes[1].barh(conc.index, conc.values, color="#1A2E2A")
    axes[1].set_xlabel("Max single-supplier share (%)")
    axes[1].set_title("Supplier concentration risk by material")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "risk_model_eval.png", dpi=150)
    plt.close(fig)

    # Per-archetype recall on the SHIPPED dataset -- the basis risk_agent.py and
    # the dashboard caption both name. Persisted so the number the UI shows is
    # read from the same artifact that produced it, the way watch_threshold is.
    scored_all = df.assign(flagged=all_flag)
    archetype_recall = {
        str(name): {"n": int(len(sub)), "recall": round(float(sub.flagged.mean()), 3)}
        for name, sub in scored_all[scored_all.is_anomaly == 1].groupby("anomaly_type")
    }
    print()
    print("  per-archetype recall on the full shipped dataset (what the UI reports):")
    for name, e in sorted(archetype_recall.items(), key=lambda kv: -kv[1]["recall"]):
        print(f"    {name:28s} n={e['n']:3d}  recall={e['recall']:.3f}")

    metrics = {
        # headline: mean over N_SPLITS repeated splits, with the spread
        "auc_pr_mean": float(np.mean(aucs)), "auc_pr_sd": float(np.std(aucs)),
        # per-archetype recall on the shipped dataset, so the UI never hardcodes it
        "archetype_recall_shipped_dataset": archetype_recall,
        "precision_mean": float(np.mean(precs)), "recall_mean": float(np.mean(recs)),
        "f1_mean": float(np.mean(f1s)), "n_splits": N_SPLITS,
        # the single reference split the shipped artifact was fitted on
        "auc_pr_reference_split": float(auc_pr), "auc_roc_reference_split": float(auc_roc),
        "prevalence": float(baseline), "n_train": len(train_df), "n_test": len(test_df),
        "contamination_threshold": CONTAMINATION,
    }
    return model, scaler, df, score_min, score_max, watch_threshold, metrics


if __name__ == "__main__":
    print("Generating synthetic supply chain shipment data...")
    suppliers = generate_suppliers()
    df = generate_shipments(suppliers)
    print(f"{len(df)} shipments across {suppliers.supplier_id.nunique()} suppliers, "
          f"{len(MATERIALS)} materials. {df.is_anomaly.sum()} true injected anomalies "
          f"across 5 archetypes.\n")

    df.to_csv(PROC_DIR / "supply_chain_shipments.csv", index=False)

    print("Training Isolation Forest...")
    model, scaler, df_scored, score_min, score_max, watch_threshold, metrics = \
        train_and_evaluate(df)

    # watch_threshold travels WITH the model. Recomputing a decision boundary at
    # startup from a gitignored CSV meant the artifact alone did not determine
    # the model's behaviour.
    joblib.dump({"model": model, "scaler": scaler, "feature_cols": FEATURE_COLS,
                 "raw_score_min": score_min, "raw_score_max": score_max,
                 "watch_threshold": watch_threshold, "metrics": metrics},
                MODEL_DIR / "risk_model.pkl")
    df_scored.to_csv(PROC_DIR / "supply_chain_shipments.csv", index=False)
    print(f"\nSaved model to {MODEL_DIR / 'risk_model.pkl'}")
    print(f"Saved plots to {DOCS_DIR}")
