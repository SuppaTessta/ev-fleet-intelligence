"""Design tokens and page styling.

One place owns colour, so the UI cannot re-derive a risk band the backend
already owns.

Contrast is measured, not asserted: tests/test_dashboard_theme.py recomputes
every text-on-tint pair from these tokens and fails below WCAG AA 4.5:1. The
values here measure 5.63 / 6.36 / 6.98, leaving room to adjust a tint without
silently dropping under the line.
"""

from dataclasses import dataclass

WCAG_AA_NORMAL = 4.5
WCAG_AA_LARGE = 3.0


@dataclass(frozen=True)
class RiskBand:
    text: str      # readable on `tint`; carries the actual information
    accent: str    # dot / chart fill. DECORATIVE-REDUNDANT: risk_pill() always
                   # prints the band name in words beside the dot, so colour is
                   # never the sole carrier and WCAG 1.4.11 does not bite. The
                   # test suite asserts that redundancy rather than a contrast
                   # bar that does not apply.
    tint: str      # pill background, already blended against white


RISK_BANDS = {
    "healthy": RiskBand(text="#046C4E", accent="#00A676", tint="#E0F4EF"),
    "watch": RiskBand(text="#92400E", accent="#D97706", tint="#FDF1E0"),
    "critical": RiskBand(text="#9B1C1C", accent="#E5484D", tint="#FCE9EA"),
    # distinct from the three risk levels on purpose: "we have no evidence" is
    # not a risk level, and colouring it like one is what the compound-risk bug
    # did in the API.
    "unknown": RiskBand(text="#44403C", accent="#A8A29E", tint="#F5F5F4"),
}

PRIORITY_ICONS = {
    "compound priority": "🔴",
    "single-signal priority": "🟡",
    "routine": "🟢",
    "insufficient data": "⚪",
}

INK = "#1A2E2A"
MUTED = "#5C6F6A"
SURFACE = "#F7F9F8"
BORDER = "#E7ECEA"
BRAND = "#00A676"


def relative_luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(a: str, b: str) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def band(name: str) -> RiskBand:
    """Never raises. A band name this UI does not know renders as `unknown`
    rather than crashing the page."""
    return RISK_BANDS.get(name, RISK_BANDS["unknown"])


PAGE_CSS = f"""
<style>
  .stMetric {{
      background: {SURFACE}; border-radius: 10px; padding: 14px 18px;
      border: 1px solid {BORDER};
  }}
  div[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-weight: 700; }}
  div[data-testid="stMetricLabel"] {{
      font-weight: 500; color: {MUTED}; letter-spacing: 0.01em;
  }}
  .risk-pill {{
      display: inline-flex; align-items: center; gap: 6px;
      padding: 4px 12px 4px 9px; border-radius: 999px;
      font-weight: 600; font-size: 0.8rem; letter-spacing: 0.02em;
      white-space: nowrap;
  }}
  .risk-dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
  .provenance {{
      display: inline-block; padding: 2px 8px; border-radius: 4px;
      font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em;
      text-transform: uppercase; border: 1px solid {BORDER}; color: {MUTED};
      background: {SURFACE};
  }}
  h1 {{ border-bottom: 3px solid {BRAND}; padding-bottom: 10px; }}

  /* Streamlit's dark theme keeps our light surfaces otherwise, which is worse
     than not supporting dark mode at all. */
  @media (prefers-color-scheme: dark) {{
      .stMetric {{ background: rgba(255,255,255,0.04); border-color: rgba(255,255,255,0.12); }}
      div[data-testid="stMetricLabel"] {{ color: #9CA3AF; }}
  }}
</style>
"""
