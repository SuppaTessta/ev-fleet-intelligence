# ADR-0004: Refactor Streamlit rather than rewrite the frontend in React

**Status:** accepted

## Context

The dashboard was a single 684-line Streamlit script. Streamlit caps how product-like a UI can
look, and a React/Next.js rewrite would demonstrate full-stack range.

## Decision

Keep Streamlit. Split it into an app shell, a typed API client, a theme module, shared components,
and one module per page. Do a real design pass — Altair charts, measured contrast, genuine
empty/loading/error states — rather than a framework change.

## Reasoning

The 684 lines were not a Streamlit problem. They were an architecture problem:

- Eight sections each repeated their own availability check and their own error handling, or none
- The Supply Chain page re-derived a risk-banding rule the backend owned, using a threshold of 0.6
  against the backend's real 0.1776 — so 54 shipments the backend called CRITICAL displayed as
  WATCH, and two tabs on the same page disagreed about the same shipment
- `api_alive()` caught only `ConnectionError`, so a slow backend raising `ReadTimeout` rendered the
  entire app as a traceback
- The landing page issued 16 sequential HTTP calls per rerun

A rewrite would have carried every one of those defects into a new framework, because none of them
are presentation-layer problems. Fixing them in place is both the smaller change and the one that
addresses the actual faults.

The estimate was 25–40 hours for a rewrite against 12 for the refactor, on a solo project with
substantive correctness work still outstanding in the ML pipelines. **A mediocre React app signals
worse than an excellent Streamlit one**, and the hours were better spent on the label bug and the
split leakage.

## Consequences

- Ceiling on visual polish accepted. Streamlit's layout primitives constrain the design.
- The refactor produced the seam that mattered anyway: `api_client.py` is the only module that
  knows the backend exists, so a future frontend in any framework has one integration point.
- 19 dashboard tests now exist via Streamlit's `AppTest`, verified with the backend stopped. An
  equivalent React harness would have been a larger investment for the same guarantee.

## Revisit if

The dashboard needs real-time collaboration, complex client-side state, or a mobile layout — none
of which Streamlit does well, and all of which would justify the cost.
