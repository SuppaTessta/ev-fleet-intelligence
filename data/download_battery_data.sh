#!/usr/bin/env bash
# Downloads real NASA PCoE battery cycling data (discharge, cycle-level capacity
# readings) for batteries B0005, B0006, B0007.
#
# Source: NASA Prognostics Center of Excellence Battery Data Set
#   B. Saha and K. Goebel (2007), "Battery Data Set", NASA Ames Research Center.
# Mirrored as CSV (converted from the original .mat format) at:
#   https://github.com/huzaifi18/RUL_prediction
#
# Usage: bash data/download_battery_data.sh
set -e
BASE="https://raw.githubusercontent.com/huzaifi18/RUL_prediction/562f109b/data/NASA"
OUT="$(dirname "$0")/raw/nasa_battery"
mkdir -p "$OUT"

for b in B0005 B0006 B0007; do
  echo "Fetching ${b} discharge cycle data..."
  curl -sL -o "${OUT}/${b}_discharge.csv" "${BASE}/discharge/train/${b}_discharge.csv"
done

echo "Done. Files in ${OUT}:"
ls -la "${OUT}"
