"""
Parses NASA PCoE battery .mat files (the original MATLAB format NASA
publishes) into the same per-discharge-cycle CSV schema already used for
B0005/6/7 — so any additional battery can be pooled into the same training
pipeline without special-casing.

Source .mat files: NASA Ames Prognostics Center of Excellence.
Convention used (since the raw format stores full sub-second time series
per cycle, but our schema is one row per cycle): voltage/current/temperature
columns take the *last* reading in the cycle (end-of-discharge state);
Capacity is already a per-cycle scalar in NASA's format, and time is the
total discharge duration in seconds.

Usage: python data/parse_mat_battery.py <path_to_BXXXX.mat> <battery_id>
"""

import sys
from pathlib import Path

import pandas as pd
import scipy.io as sio

OUT_DIR = Path(__file__).resolve().parent / "raw" / "nasa_battery"


def parse_mat_to_discharge_csv(mat_path: Path, battery_id: str) -> pd.DataFrame:
    mat = sio.loadmat(mat_path)
    cycles = mat[battery_id][0, 0]["cycle"][0]

    rows = []
    cycle_num = 0
    for c in cycles:
        cycle_num += 1
        if c["type"][0] != "discharge":
            continue
        data = c["data"][0, 0]
        if data["Capacity"].size == 0:
            continue  # known gap in some NASA files: a few cycles have no capacity reading
        t = c["time"][0]  # [year, month, day, hour, min, sec]
        date_str = f"{int(t[2]):02d} {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(t[1])-1]} {int(t[0])}, {int(t[3]):02d}:{int(t[4]):02d}:{t[5]:05.2f}"

        rows.append({
            "cycle": cycle_num,
            "amb_temp": int(c["ambient_temperature"][0, 0]),
            "date_time": date_str,
            "voltage_battery": float(data["Voltage_measured"][0, -1]),
            "current_battery": float(data["Current_measured"][0, -1]),
            "temp_battery": float(data["Temperature_measured"][0, -1]),
            "current_load": float(data["Current_load"][0, -1]),
            "voltage_load": float(data["Voltage_load"][0, -1]),
            "time": float(data["Time"][0, -1]),
            "capacity": float(data["Capacity"][0, 0]),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data/parse_mat_battery.py <path_to_BXXXX.mat> <battery_id>")
        sys.exit(1)

    mat_path, battery_id = Path(sys.argv[1]), sys.argv[2]
    df = parse_mat_to_discharge_csv(mat_path, battery_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{battery_id}_discharge.csv"
    df.to_csv(out_path, index=False)
    print(f"Parsed {len(df)} discharge cycles from {battery_id} -> {out_path}")
    print(f"Capacity range: {df.capacity.min():.3f} - {df.capacity.max():.3f} Ah")
