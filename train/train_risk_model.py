"""
EV Supply Chain Risk & Traceability Agent — training pipeline.

No public dataset fits "EV battery supply chain fraud" — so this one is
synthetic, same as the honest tradeoff flagged from the start. What makes
it a legitimate demo rather than noise: the generator is grounded in real,
well-documented supply-chain facts (China's dominance in cell/rare-earth
processing, DRC's dominance and ESG risk in cobalt, Australia/Chile as
lower-risk lithium sources), and anomalies are injected as 5 named,
realistic risk archetypes rather than random noise — so evaluation means
something (precision/recall against known-injected anomalies), matching
the same Isolation Forest technique as Fintech-Fraud-Detection.

Archetype #5 (gradual_quality_decline) exists because of a real finding:
testing "detection lead time" against the first 4 archetypes showed the
model was actually LATE on a slow quality drift — it only ever saw sudden
spikes in training, never a gradual ramp through moderate values. Trend
features (rate of change vs. this supplier's own recent shipments) plus
gradual-decline training examples close that gap. See
train/analyze_risk_lead_time.py for the before/after.

Outputs
-------
- data/processed/supply_chain_shipments.csv
- backend/models/risk_model.pkl
- docs/risk_model_eval.png
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[1] / "backend" / "models"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
for d in (PROC_DIR, MODEL_DIR, DOCS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
CONTAMINATION = 0.09  # slightly higher now that 5 archetypes are injected instead of 4

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
        single_sourced = N_SUPPLIERS_PER_MATERIAL == 1 or shares.max() > 0.6
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
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])

    model = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=42)
    model.fit(X)
    raw_scores = model.score_samples(X)  # higher = more normal
    df["risk_score"] = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    df["flagged"] = (model.predict(X) == -1).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        df.is_anomaly, df.flagged, average="binary", zero_division=0)
    print(f"Precision: {precision:.2f}  Recall: {recall:.2f}  F1: {f1:.2f}")
    print(f"\nFlagged {df.flagged.sum()} of {len(df)} shipments "
          f"({df.flagged.sum()/len(df)*100:.1f}%) as anomalous.")
    print("\nOf flagged shipments, by true anomaly type:")
    print(df[df.flagged == 1].anomaly_type.value_counts())

    # ---- eval plot: risk score distribution, normal vs true-anomalous ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df[df.is_anomaly == 0].risk_score, bins=30, alpha=0.6, label="Normal", color="#00A676")
    axes[0].hist(df[df.is_anomaly == 1].risk_score, bins=30, alpha=0.6, label="Injected anomaly", color="#E5484D")
    axes[0].axvline(df[df.flagged == 1].risk_score.min(), color="black", linestyle="--", label="Flagging threshold")
    axes[0].set_xlabel("Risk score"); axes[0].set_ylabel("Count")
    axes[0].set_title("Risk score separation"); axes[0].legend(fontsize=8)

    conc = df.groupby("material")["supplier_concentration_pct"].max().sort_values()
    axes[1].barh(conc.index, conc.values, color="#1A2E2A")
    axes[1].set_xlabel("Max single-supplier share (%)")
    axes[1].set_title("Supplier concentration risk by material")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "risk_model_eval.png", dpi=150)
    plt.close(fig)

    return model, scaler, df, float(raw_scores.min()), float(raw_scores.max())


if __name__ == "__main__":
    print("Generating synthetic supply chain shipment data...")
    suppliers = generate_suppliers()
    df = generate_shipments(suppliers)
    print(f"{len(df)} shipments across {suppliers.supplier_id.nunique()} suppliers, "
          f"{len(MATERIALS)} materials. {df.is_anomaly.sum()} true injected anomalies "
          f"across 5 archetypes.\n")

    df.to_csv(PROC_DIR / "supply_chain_shipments.csv", index=False)

    print("Training Isolation Forest...")
    model, scaler, df_scored, score_min, score_max = train_and_evaluate(df)

    joblib.dump({"model": model, "scaler": scaler, "feature_cols": FEATURE_COLS,
                 "raw_score_min": score_min, "raw_score_max": score_max},
                MODEL_DIR / "risk_model.pkl")
    df_scored.to_csv(PROC_DIR / "supply_chain_shipments.csv", index=False)
    print(f"\nSaved model to {MODEL_DIR / 'risk_model.pkl'}")
    print(f"Saved plots to {DOCS_DIR}")
