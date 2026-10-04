"""
The only module that knows the backend exists.

WHAT THIS REPLACES
------------------
13 scattered `requests.get/post` call sites with timeouts of 2, 10 and 30
seconds, no shared session, and no try/except anywhere. Three consequences, all
of which a user could hit in a demo:

1. `api_alive()` caught only `requests.exceptions.ConnectionError`. A slow
   backend raises `ReadTimeout`, which is NOT a subclass of it, and the call sat
   at module scope -- so a sluggish backend rendered the entire dashboard,
   sidebar included, as a Python traceback.
2. Several calls had no `raise_for_status()`, so an error body was parsed as if
   it were a result and failed later with a confusing KeyError.
3. Failures that were caught printed `resp.text` straight to the page. With the
   old backend that string contained absolute filesystem paths.

Now: one `Session`, one timeout policy, one place that turns a failure into a
message a human can act on, and a single `ApiError` for callers to render.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# Distinct budgets per class of call: probes must fail fast to keep the UI
# responsive, inference needs room for a CPU-bound ResNet50 forward pass.
TIMEOUT_PROBE = 2.0
TIMEOUT_READ = 10.0
TIMEOUT_INFERENCE = 45.0


class ApiError(Exception):
    """A backend call failed in a way worth showing the user.

    `detail` is safe to render: it is either our own message or the backend's
    structured `detail` field, never a raw exception string.
    """

    def __init__(self, message: str, *, status: int | None = None,
                 request_id: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.request_id = request_id
        self.hint = hint


def _base_url() -> str:
    """Where the backend lives, in precedence order.

    1. `EV_FLEET_API_URL` -- how Docker Compose points the dashboard at the
       `api` service, since 127.0.0.1 inside a container is the container itself.
    2. `st.secrets["api_base_url"]` -- for Streamlit Cloud style deployments.
    3. localhost -- so `streamlit run` needs no configuration at all.
    """
    from_env = os.environ.get("EV_FLEET_API_URL")
    if from_env:
        return from_env.rstrip("/")
    try:
        return st.secrets.get("api_base_url", DEFAULT_BASE_URL)
    except Exception:
        # st.secrets raises outright if no secrets.toml exists anywhere
        return DEFAULT_BASE_URL


@st.cache_resource
def _session() -> requests.Session:
    """One connection pool for the whole app rather than a fresh TCP handshake
    per widget interaction."""
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    return session


def _request(method: str, path: str, *, timeout: float, **kwargs) -> dict:
    url = f"{_base_url()}{path}"
    try:
        resp = _session().request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectTimeout as e:
        raise ApiError("The backend did not accept the connection in time.",
                       hint="Is it still starting up?") from e
    except requests.exceptions.ReadTimeout as e:
        # the specific case that used to crash the whole page
        raise ApiError(f"The backend took longer than {timeout:.0f}s to respond.",
                       hint="It may be loading a model on the first request.") from e
    except requests.exceptions.ConnectionError as e:
        raise ApiError("Cannot reach the backend.",
                       hint="Start it with: uvicorn app.main:app --port 8000") from e
    except requests.RequestException as e:
        raise ApiError("The request to the backend failed.") from e

    request_id = resp.headers.get("X-Request-ID")
    if resp.status_code >= 400:
        raise ApiError(_readable_error(resp), status=resp.status_code,
                       request_id=request_id, hint=_hint_for(resp.status_code))
    try:
        return resp.json()
    except ValueError as e:
        raise ApiError("The backend returned a response that is not valid JSON.",
                       status=resp.status_code, request_id=request_id) from e


def _readable_error(resp: requests.Response) -> str:
    """Extract the backend's structured `detail`. Never returns raw resp.text --
    that is what leaked filesystem paths into the UI."""
    try:
        payload = resp.json()
    except ValueError:
        return f"The backend returned HTTP {resp.status_code}."
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list) and detail:
        first = detail[0]
        field = ".".join(str(p) for p in first.get("loc", [])[1:]) or "input"
        return f"{field}: {first.get('msg', 'is invalid')}"
    return f"The backend returned HTTP {resp.status_code}."


def _hint_for(status: int) -> str | None:
    if status == 503:
        return "A model artifact is missing. Check /ready, then run the training step."
    if status == 413:
        return "The request was too large."
    if status == 422:
        return "The request was rejected as invalid."
    return None


def get(path: str, *, timeout: float = TIMEOUT_READ) -> dict:
    return _request("GET", path, timeout=timeout)


def post(path: str, payload: dict | None = None, *,
         timeout: float = TIMEOUT_READ, **kwargs) -> dict:
    return _request("POST", path, timeout=timeout, json=payload, **kwargs)


def post_file(path: str, filename: str, data: bytes, content_type: str,
              *, timeout: float = TIMEOUT_INFERENCE) -> dict:
    return _request("POST", path, timeout=timeout,
                    files={"file": (filename, data, content_type)})


@st.cache_data(ttl=5, show_spinner=False)
def health() -> dict:
    """Backend availability plus which agents are actually usable.

    Cached for 5s because this was previously called on every rerun in every
    section -- two extra round trips per interaction. Returns a dict rather than
    a bool so the sidebar can distinguish "down" from "up but degraded", which
    the old boolean could not express and which now matters: the backend reports
    per-agent availability from disk instead of a hardcoded list.
    """
    try:
        return {"reachable": True, **get("/", timeout=TIMEOUT_PROBE)}
    except ApiError as e:
        return {"reachable": False, "error": e.message, "hint": e.hint,
                "agents_available": [], "agents_degraded": []}


def agent_available(agent: str) -> bool:
    status = health()
    return status["reachable"] and agent in status.get("agents_available", [])
