"""
Parses every new battery .mat file (copied from the full Kaggle NASA
dataset) into the standard discharge CSV format, in one command.

Run this AFTER copying the .mat files into data/raw/nasa_battery/ and
BEFORE running train/train_battery_model.py.

Usage: python data/parse_new_batteries.py
"""

from pathlib import Path
from parse_mat_battery import parse_mat_to_discharge_csv

RAW_DIR = Path(__file__).resolve().parent / "raw" / "nasa_battery"

NEW_BATTERIES = [
    "B0029", "B0030", "B0031", "B0032",  # G3: 43C, 4A
    "B0033", "B0034", "B0036",            # G4: 24C, 2A/4A
    "B0045", "B0046", "B0047", "B0048",  # G7: 4C, 1A
    "B0053", "B0054", "B0055", "B0056",  # G9: 4C, 2A
]

if __name__ == "__main__":
    ok, missing = [], []
    for b in NEW_BATTERIES:
        mat_path = RAW_DIR / f"{b}.mat"
        if not mat_path.exists():
            missing.append(b)
            continue
        df = parse_mat_to_discharge_csv(mat_path, b)
        df.to_csv(RAW_DIR / f"{b}_discharge.csv", index=False)
        print(f"  {b}: parsed {len(df)} discharge cycles")
        ok.append(b)

    print(f"\nDone: {len(ok)} parsed, {len(missing)} missing.")
    if missing:
        print(f"Missing .mat files (check they were copied to {RAW_DIR}):")
        print(" ", ", ".join(missing))
