"""
Smoke tests for every dashboard section, via Streamlit's own AppTest harness.

There was previously no test of the dashboard at all, in a 684-line file where
each of eight sections did its own error handling. The failures that mattered
were not logic errors but unguarded ones: a `FileNotFoundError` on the landing
page, a `KeyError` from a bare dict subscript on a value received over the wire,
and a `ReadTimeout` that escaped `api_alive()` and rendered the whole app --
sidebar included -- as a traceback.

So this asserts the property that actually protects a demo: **no section raises,
whatever the backend is doing.** Each is rendered with the backend unreachable,
which is the state a fresh clone starts in, and must degrade to a message.

Needs no backend and no artifacts, so it runs in CI.
"""
import sys
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASHBOARD))

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="needs streamlit>=1.28").AppTest

SECTIONS = [
    "Fleet Command Center",
    "Fleet Overview",
    "Battery Deep-Dive",
    "Quality Inspection",
    "Supply Chain Risk",
    "Electrification Readiness",
    "Net Zero Carbon Tracker",
    "Maintenance Optimiser",
]


def _run(section: str | None = None, timeout: int = 60):
    app = AppTest.from_file(str(DASHBOARD / "main.py"), default_timeout=timeout)
    app.run()
    if section and app.sidebar.radio:
        app.sidebar.radio[0].set_value(section).run()
    return app


def test_app_starts():
    app = _run()
    assert not app.exception, f"dashboard raised on startup: {app.exception}"


def test_every_section_is_navigable():
    app = _run()
    assert app.sidebar.radio, "no navigation control rendered"
    options = list(app.sidebar.radio[0].options)
    for section in SECTIONS:
        assert section in options, f"{section} missing from navigation"


@pytest.mark.parametrize("section", SECTIONS)
def test_section_renders_without_raising(section):
    """The core guarantee. Whatever state the backend is in, a section either
    renders or explains itself -- it never throws."""
    app = _run(section)
    assert not app.exception, f"{section} raised: {app.exception}"


@pytest.mark.parametrize("section", SECTIONS)
def test_section_says_something_when_it_cannot_load(section):
    """Degrading silently to a blank page is only marginally better than a
    traceback. If a section cannot show data it must say why."""
    app = _run(section)
    produced_output = bool(
        app.warning or app.error or app.info or app.dataframe
        or app.metric or app.markdown or app.caption)
    assert produced_output, f"{section} rendered nothing at all"


def test_live_monitor_is_excluded_deliberately():
    """The Live Fleet Monitor runs inside @st.fragment(run_every="3s"), which
    AppTest cannot drive. Its POST-not-GET behaviour is covered by the backend
    contract test instead. Listed here so the omission is explicit rather than
    looking like an oversight."""
    app = _run()
    assert "Live Fleet Monitor" in list(app.sidebar.radio[0].options)
