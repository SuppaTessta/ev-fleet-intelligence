"""
Accessibility and robustness of the dashboard's design tokens.

The palette previously carried a comment asserting it met WCAG AA. It did not:
`healthy` measured 3.98:1 and `watch` sat at exactly 4.50, where any tweak to
the tint would silently break it. Recomputing the ratios from the tokens turns
that claim into something that fails a build instead of aging quietly.

Needs no backend and no model artifacts, so it runs in CI.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

from theme import (  # noqa: E402
    PRIORITY_ICONS,
    RISK_BANDS,
    WCAG_AA_NORMAL,
    band,
    contrast_ratio,
)


@pytest.mark.parametrize("name", list(RISK_BANDS))
def test_band_text_meets_wcag_aa_on_its_own_pill(name):
    tokens = RISK_BANDS[name]
    ratio = contrast_ratio(tokens.text, tokens.tint)
    assert ratio >= WCAG_AA_NORMAL, (
        f"{name}: {tokens.text} on {tokens.tint} is {ratio:.2f}:1, below AA ({WCAG_AA_NORMAL})")


@pytest.mark.parametrize("name", list(RISK_BANDS))
def test_band_text_has_headroom_not_just_a_pass(name):
    """A token sitting at exactly 4.50 is one rounding away from failing. Require
    a real margin so the palette can be adjusted without silently regressing."""
    tokens = RISK_BANDS[name]
    assert contrast_ratio(tokens.text, tokens.tint) >= 5.0


@pytest.mark.parametrize("name", ["healthy", "watch", "critical", "unknown"])
def test_colour_is_never_the_only_carrier_of_meaning(name):
    """The accent dots do NOT meet the 3:1 non-text bar -- amber on its own tint
    measures 1.79:1, and darkening every accent enough to clear it would destroy
    the visual coding.

    That is acceptable only because the dots are decorative-REDUNDANT: the pill
    always prints the band name in words beside the dot, so colour is never the
    sole means of conveying the information and WCAG 1.4.11 does not apply. This
    test asserts the redundancy that the exemption depends on. An earlier
    version asserted a contrast bar against white -- the wrong background and
    the wrong requirement.
    """
    from components import risk_pill

    markup = risk_pill(name)
    assert name.upper() in markup, "the band name must be rendered as text, not colour alone"
    assert RISK_BANDS[name].text in markup
    assert RISK_BANDS[name].accent in markup


def test_unknown_is_visually_distinct_from_every_risk_level():
    """'No evidence' must not look like a risk level. Treating unknown as a weak
    signal is exactly the bug that escalated unrecognised assets in the API."""
    unknown = RISK_BANDS["unknown"]
    for name in ("healthy", "watch", "critical"):
        assert unknown.accent != RISK_BANDS[name].accent


def test_band_lookup_never_raises_on_unexpected_input():
    """Band names arrive over the wire. A bare dict subscript on one is what
    would have killed the landing page when the backend gained a new priority."""
    for value in ("", "Watch", "not-a-band", "compound priority", None):
        assert band(value) is RISK_BANDS["unknown"] or band(value) in RISK_BANDS.values()


def test_every_backend_priority_has_an_icon():
    """Keep the UI's vocabulary in step with the API's."""
    expected = {"routine", "single-signal priority", "compound priority", "insufficient data"}
    assert expected <= set(PRIORITY_ICONS)
