"""
Downloads real NASA PCoE battery cycling data for batteries B0005, B0006,
B0007 (pre-converted CSV mirror), and B0018 (raw .mat, parsed locally with
parse_mat_battery.py — NASA's own distribution only ships B0018 as .mat).

Source: NASA Prognostics Center of Excellence Battery Data Set
  B. Saha and K. Goebel (2007), "Battery Data Set", NASA Ames Research Center.
CSV mirror (B0005/6/7): https://github.com/huzaifi18/RUL_prediction
Raw .mat mirror (B0018): https://github.com/XiuzeZhou/NASA

Usage: python data/download_battery_data.py
"""

import urllib.request
from pathlib import Path
from parse_mat_battery import parse_mat_to_discharge_csv

CSV_BASE = "https://raw.githubusercontent.com/huzaifi18/RUL_prediction/562f109b/data/NASA"
MAT_BASE = "https://raw.githubusercontent.com/XiuzeZhou/NASA/main/dataset"
OUT_DIR = Path(__file__).resolve().parent / "raw" / "nasa_battery"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for battery in ["B0005", "B0006", "B0007"]:
        print(f"Fetching {battery} discharge cycle data (CSV)...")
        urllib.request.urlretrieve(f"{CSV_BASE}/discharge/train/{battery}_discharge.csv",
                                    OUT_DIR / f"{battery}_discharge.csv")

    print("Fetching B0018 raw data (.mat) and parsing to the same CSV schema...")
    mat_path = OUT_DIR / "B0018.mat"
    urllib.request.urlretrieve(f"{MAT_BASE}/B0018.mat", mat_path)
    df = parse_mat_to_discharge_csv(mat_path, "B0018")
    df.to_csv(OUT_DIR / "B0018_discharge.csv", index=False)

    print(f"\nDone. Files in {OUT_DIR}:")
    for f in sorted(OUT_DIR.glob("*_discharge.csv")):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
