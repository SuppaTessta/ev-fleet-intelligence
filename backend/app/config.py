"""Filesystem locations and service metadata, resolved once at import.

Paths are overridable by environment variable so a container can be pointed at
a mounted model volume without rebuilding the image:

    EV_FLEET_MODELS_DIR   default <repo>/backend/models
    EV_FLEET_DATA_DIR     default <repo>/data/processed
"""

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent

# Kept in step with the top entry of CHANGELOG.md by tests/test_no_drift.py.
VERSION = "2.2.0"


def _directory(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return Path(override).expanduser().resolve() if override else default


MODELS_DIR = _directory("EV_FLEET_MODELS_DIR", BACKEND_ROOT / "models")
PROCESSED_DATA_DIR = _directory("EV_FLEET_DATA_DIR", PROJECT_ROOT / "data" / "processed")

BATTERY_MODEL = MODELS_DIR / "battery_rul_model.pkl"
RISK_MODEL = MODELS_DIR / "risk_model.pkl"
QUALITY_CASTING_MODEL = MODELS_DIR / "quality_model.keras"
QUALITY_SURFACE_MODEL = MODELS_DIR / "quality_model_neu_det.keras"

BATTERY_FEATURES_CSV = PROCESSED_DATA_DIR / "battery_features.csv"
SHIPMENTS_CSV = PROCESSED_DATA_DIR / "supply_chain_shipments.csv"
