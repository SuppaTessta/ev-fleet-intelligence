"""
Shared fixtures for the test suite.

Four agents (compound risk, maintenance, carbon, fleet readiness) are pure
logic with no trained artifact -- those tests always run, including in CI.

Four more (battery, both quality classifiers, risk) need trained model
files and/or processed data that are gitignored by design (see
train/train_*.py and data/*.py -- reproducible locally, not committed).
Those test modules call skip_if_missing() at import time so they report a
clear, honest "skipped: model not found, run train/train_X.py first"
instead of erroring confusingly -- this works correctly whether run in CI
(always skips), on a fresh clone before training (skips), or locally after
training (runs for real). No special pytest flags needed either way.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def skip_if_missing(*paths: Path):
    """Call at module level in a data-dependent test file. Skips the whole
    module with a clear, actionable reason if any required file is absent."""
    for p in paths:
        if not p.exists():
            pytest.skip(
                f"skipping: {p.relative_to(PROJECT_ROOT)} not found -- "
                f"this is gitignored and reproducible locally, not a bug. "
                f"See README.md / SETUP.md for the training step that produces it.",
                allow_module_level=True,
            )


MODELS_DIR = PROJECT_ROOT / "backend" / "models"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def client():
    from app.main import app
    return TestClient(app)
