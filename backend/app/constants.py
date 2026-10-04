"""Constants shared by more than one module.

Values used by a single module stay in that module. These are here because two
copies of a shared constant drift, and the drift is silent.
"""

# Rolling window for battery capacity features. The training pipeline and the
# serving agent must agree, or the model is fed features on a different basis
# than it was fitted on.
ROLL_WINDOW = 5

# Rs/kWh, Indian commercial EV packs, 2026. Midpoint of a consistently reported
# Rs 15,000-25,000 range (OEM service data, EV industry press).
BATTERY_COST_INR_PER_KWH = 18_000

# Single-shift local delivery duty cycle. The carbon agent annualises daily
# savings with it; the TCO model derives lifetime distance from it.
OPERATING_DAYS_PER_YEAR = 300
