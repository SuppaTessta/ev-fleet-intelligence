"""
EV Fleet Intelligence — unified dashboard.

Presentation only. Every backend call goes through api_client, which owns
timeouts, retries and the one place a failure becomes a message a human can act
on. This file is a router and a sidebar.

It replaces a single 684-line script in which eight sections each repeated their
own availability check, their own error handling (or none), and — in the supply
chain section — their own copy of a risk-banding rule the backend already owned.

Run:  streamlit run dashboard/main.py   (from the project root)
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit executes this file as a script, not a package, so sibling modules
# are not importable without help. Kept to the dashboard directory only.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import health  # noqa: E402
from pages_impl import battery_pages, command_center, ops_pages, supply_chain  # noqa: E402
from theme import PAGE_CSS  # noqa: E402

st.set_page_config(page_title="EV Fleet Intelligence", page_icon="🔋", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# section label -> (renderer, agent it needs or None)
SECTIONS = {
    "Fleet Command Center": command_center.render,
    "Fleet Overview": battery_pages.render_overview,
    "Battery Deep-Dive": battery_pages.render_deep_dive,
    "Live Fleet Monitor": battery_pages.render_live_monitor,
    "Quality Inspection": ops_pages.render_quality,
    "Supply Chain Risk": supply_chain.render,
    "Electrification Readiness": ops_pages.render_readiness,
    "Net Zero Carbon Tracker": ops_pages.render_carbon,
    "Maintenance Optimiser": ops_pages.render_maintenance,
}


def render_sidebar() -> str:
    st.sidebar.title("🔋 EV Fleet Intelligence")
    st.sidebar.caption("Asset intelligence for industrial EV fleets")
    section = st.sidebar.radio("Navigate", list(SECTIONS))
    st.sidebar.divider()

    # One cached health call for the whole app. This used to be an uncached
    # boolean invoked on every rerun in every section, and it caught only
    # ConnectionError -- so a merely slow backend raised ReadTimeout at module
    # scope and rendered the entire page, sidebar included, as a traceback.
    status = health()
    if not status["reachable"]:
        st.sidebar.error("Backend not reachable")
        st.sidebar.code("cd backend\nuvicorn app.main:app --reload --port 8000", language="bash")
        if status.get("hint"):
            st.sidebar.caption(status["hint"])
    elif status.get("agents_degraded"):
        # "up but partially degraded" is a real state the old boolean could not
        # express, and the old root endpoint reported all seven agents live
        # regardless of whether a single model file existed
        st.sidebar.warning("Backend up — some agents degraded")
        st.sidebar.caption("Degraded: " + ", ".join(status["agents_degraded"])
                            + ". Other sections still work.")
        # Name the actual cause. "Missing model artifacts" was printed for every
        # degradation, including the case where the artifacts are on disk and the
        # library that loads them is not -- which points the reader at a
        # retraining step that cannot fix it.
        if status.get("missing_artifacts"):
            st.sidebar.caption("Missing artifacts: "
                                + ", ".join(status["missing_artifacts"]))
        if status.get("missing_runtime_dependencies"):
            st.sidebar.caption("Missing Python packages: "
                                + ", ".join(status["missing_runtime_dependencies"])
                                + " — `pip install -r backend/requirements.txt` on a "
                                  "Python version that has a wheel (see SETUP.md).")
    else:
        st.sidebar.success("Backend connected — all agents available")

    return section


def main() -> None:
    section = render_sidebar()
    SECTIONS[section]()


main()
