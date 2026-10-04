"""
Shared UI pieces: risk pills, provenance badges, and the empty/error states.

The error states are the point. Every section previously repeated

    if not api_alive():
        st.warning("Start the backend (see sidebar) to ...")
        st.stop()

eight times, and every other failure mode -- backend up but model artifact
missing, request rejected, file not found -- either crashed or dumped raw
backend text onto the page.
"""

from __future__ import annotations

import streamlit as st
from api_client import ApiError
from theme import PRIORITY_ICONS, band

# Where each agent's numbers come from. Shown in the UI rather than buried in
# the README, because the honest thing about this platform is which parts rest
# on real data and which do not -- and that belongs where the numbers are.
PROVENANCE = {
    "real": ("REAL DATA", "Trained on real public datasets."),
    "synthetic": ("SYNTHETIC", "Generated data. No real dataset of this kind is public at this scale."),
    "illustrative": ("ILLUSTRATIVE", "A documented stand-in for a system this project has no access to."),
}


def risk_pill(risk_band: str) -> str:
    tokens = band(risk_band)
    return (f'<span class="risk-pill" style="background:{tokens.tint};color:{tokens.text}">'
            f'<span class="risk-dot" style="background:{tokens.accent}"></span>'
            f'{risk_band.upper()}</span>')


def show_risk_pill(risk_band: str, container=None) -> None:
    (container or st).markdown(risk_pill(risk_band), unsafe_allow_html=True)


def priority_icon(priority: str) -> str:
    return PRIORITY_ICONS.get(priority, "⚪")


def provenance_badge(kind: str) -> None:
    label, explanation = PROVENANCE.get(kind, PROVENANCE["illustrative"])
    st.markdown(f'<span class="provenance" title="{explanation}">{label}</span>',
                unsafe_allow_html=True)
    st.caption(explanation)


def show_api_error(err: ApiError, what: str = "This section") -> None:
    """One rendering for every backend failure.

    Deliberately does not print the raw response body. It shows our message, the
    hint if there is one, and the request id -- which is the thing that actually
    lets someone find the failure in the server log.
    """
    st.error(f"{what} could not load: {err.message}")
    if err.hint:
        st.info(err.hint)
    if err.request_id:
        st.caption(f"Request ID `{err.request_id}` — quote this when reporting the problem; "
                   f"the full detail is in the backend log under that id.")


def require_agent(agent: str, what: str) -> bool:
    """Guard for a section that needs a specific agent.

    Distinguishes the three cases the old boolean `api_alive()` collapsed into
    one: backend down, backend up but this agent's artifact missing, and fine.
    """
    from api_client import health

    status = health()
    if not status["reachable"]:
        st.warning(f"{what} needs the backend, which is not reachable.")
        st.code("cd backend && uvicorn app.main:app --reload --port 8000", language="bash")
        if status.get("hint"):
            st.caption(status["hint"])
        return False
    if agent not in status.get("agents_available", []):
        st.warning(f"{what} needs the **{agent}** agent, which this backend cannot serve.")
        # Two different causes, two different fixes. Saying "artifact is missing"
        # for both sent readers to re-run a training script when the real problem
        # was an uninstallable dependency.
        deps = status.get("missing_runtime_dependencies") or []
        if deps:
            st.caption("The backend is missing the Python package(s) "
                       + ", ".join(deps)
                       + ". The model files may well be present — they cannot be loaded "
                         "without it. See SETUP.md for the supported Python version.")
        else:
            st.caption("The backend is running and other sections still work. "
                       "Check `/ready` for exactly which artifact is absent, then run the "
                       "matching script in `train/`.")
        return False
    return True


def require_file(path, what: str, build_command: str) -> bool:
    """Guard for a section reading a generated file directly.

    Previously only some sections checked. The Fleet Command Center -- the
    default landing page -- read battery_features.csv with no guard at all, so a
    fresh clone rendered a raw FileNotFoundError as the first thing on screen.
    """
    if path.exists():
        return True
    st.warning(f"{what} needs `{path.name}`, which has not been generated yet.")
    st.code(build_command, language="bash")
    st.caption("This file is gitignored by design and rebuilt from source data.")
    return False


def empty_state(message: str, detail: str | None = None) -> None:
    st.info(message)
    if detail:
        st.caption(detail)
