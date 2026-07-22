"""
Fleet Electrification Readiness Agent — thin wrapper.

The actual scoring engine (EV_CATALOG + score_vehicle) lives in
train/train_fleet_readiness.py since it's pure logic with no trained
artifact to load — this just makes it importable from the backend
regardless of the working directory uvicorn is started from.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_fleet_readiness import score_vehicle, EV_CATALOG  # noqa: E402
