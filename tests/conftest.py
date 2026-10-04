"""
Shared fixtures for the test suite.

WHAT ACTUALLY RUNS IN CI -- measured, not assumed
-------------------------------------------------
**202 of 206 tests (98%)**, measured by running the suite inside the deployment
image with EV_FLEET_MODELS_DIR pointed at a directory holding exactly what CI
has -- the two committed .pkl artifacts and neither 94 MB .keras. Only
tests/test_quality_agents.py (4 tests) skips.

All 206 pass on that image with the vision weights mounted, which is the only
environment where the two ResNet50 classifiers can run at all: tensorflow-cpu
has no wheel for Python 3.13+.

This used to be ~18 of 47 (38%) with **no model code executing at all**, while
the job still spent minutes installing tensorflow-cpu to run none of it. Three
changes closed the gap:

1. CI runs train_risk_model.py and train_fleet_readiness.py. Both synthesise
   their own data and finish in seconds, so risk, compound-risk and readiness
   tests exercise real fitted models rather than skipping.
2. battery_rul_model.pkl (340 KB) and battery_features.csv (556 KB) are
   committed -- see .gitignore for the reasoning. Under 1 MB buys real LightGBM
   inference plus the whole telemetry simulator.
3. test_quality_parity.py lost its skip guard. Those tests compare
   preprocessing tensors and never load a model, so gating them on a 98 MB
   artifact meant the single guard against the highest-impact bug in the vision
   pipeline never ran where it mattered.

Still skipped: the two ResNet50 classifiers, 98 MB each -- too large to commit,
too slow to retrain per run. Their preprocessing is covered by the parity tests;
their accuracy is not, and that is stated rather than papered over.

An earlier version of this docstring claimed compound risk "always runs" (it did
not -- it needed a gitignored CSV) and omitted test_telemetry_simulator
entirely, so a reader trusting it would have concluded ~22 tests ran.
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
                f"skipping: {p} not found -- "
                f"this is gitignored and reproducible locally, not a bug. "
                f"See README.md / SETUP.md for the training step that produces it.",
                allow_module_level=True,
            )


# Resolved by the application, not recomputed here. EV_FLEET_MODELS_DIR and
# EV_FLEET_DATA_DIR move where the service looks for artifacts, and a test that
# checked a different directory would decide an artifact is present while the
# code under test cannot find it -- failing instead of skipping.
from app import config  # noqa: E402  (needs the sys.path insert above)

MODELS_DIR = config.MODELS_DIR
DATA_PROCESSED_DIR = config.PROCESSED_DATA_DIR


@pytest.fixture(scope="session")
def client():
    """TestClient over the real app.

    The ImportError branch exists because `app.main` pulls in the quality
    router, which imports TensorFlow at module scope. On an interpreter with no
    TensorFlow wheel -- Python 3.13+, for instance -- that turned every test
    using this fixture into an ERROR (25 of them) plus 2 collection errors,
    which reads as "the suite is broken" rather than "this environment cannot
    run the vision agent". Same treatment as skip_if_missing: skip, and say what
    would fix it.
    """
    try:
        from app.main import app
    except ImportError as exc:
        pytest.skip(f"cannot import the API ({exc}). The quality agent needs "
                    f"tensorflow-cpu, which has no wheel for this Python. "
                    f"See SETUP.md -- the project targets Python 3.12.")
    return TestClient(app)
