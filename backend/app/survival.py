"""Censoring-aware regression for remaining useful life.

Only 5 of the 19 NASA cells are recorded until they cross their end-of-life
threshold. The other 14 stop while still healthy, so for 73.1% of rows the RUL
label is a LOWER BOUND -- "at least N cycles left", not "exactly N".

Treating that bound as an observation teaches the model that a healthy cell near
the end of its recording is nearly dead, and business_impact.py prices premature
replacements in rupees. Discarding those rows instead is honest but throws away
two thirds of the data.

The loss here encodes the inequality directly:

    observed   loss = (pred - y)^2          penalise both directions
    censored   loss = max(0, y - pred)^2    penalise only under-prediction

Predicting more life than the lower bound is not an error; the cell demonstrably
had at least that much. This is a one-sided squared hinge, the standard
reduction of right-censored regression when a full survival model is more
machinery than the problem needs -- implemented as a LightGBM objective rather
than by adding a survival dependency.

XGBoost's survival:aft and scikit-survival handle this natively and would be the
right call with more data. With 19 cells they add a distributional assumption
this dataset can neither support nor refute.

Whether it actually helps is measured, not asserted: see
evaluation/battery_censoring_study.py and ADR-0005.
"""

from __future__ import annotations

import numpy as np


def censored_l2_objective(censored: np.ndarray):
    """Build a LightGBM objective for right-censored targets.

    `censored[i]` is True when `y[i]` is a lower bound rather than an observed
    event time. Returns a callable of LightGBM's `(y_true, y_pred)` shape.

    The hessian is held at a constant 2.0 even where the hinge is inactive and
    the true second derivative is 0: a zero hessian makes the Newton step
    undefined and the split gain degenerate. Overstating curvature understates
    the step size, which is the safe direction to be wrong in.
    """
    censored = np.asarray(censored, dtype=bool)

    def objective(y_true: np.ndarray, y_pred: np.ndarray):
        residual = y_pred - y_true

        # observed rows: ordinary squared error, gradient 2 * residual
        grad = 2.0 * residual

        # censored rows: gradient only where the model predicts BELOW the known
        # lower bound. Predicting above it costs nothing -- that is the whole
        # point, and it is what stops the model being pulled down toward
        # "the recording stopped here" on cells that were still healthy.
        under = censored & (residual < 0)
        grad = np.where(censored, np.where(under, 2.0 * residual, 0.0), grad)

        return grad, np.full_like(y_pred, 2.0)

    return objective


def censored_mae(y_true: np.ndarray, y_pred: np.ndarray,
                 censored: np.ndarray) -> float:
    """MAE that respects censoring: a censored row contributes only when the
    prediction falls below its lower bound.

    Plain MAE against censored labels rewards a model for reproducing the point
    at which the experiment was switched off. Reported alongside, never instead
    of, the uncensored-only MAE.
    """
    censored = np.asarray(censored, dtype=bool)
    error = np.abs(y_pred - y_true)
    violation = np.maximum(0.0, y_true - y_pred)
    return float(np.mean(np.where(censored, violation, error)))


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray,
                      censored: np.ndarray) -> float:
    """Harrell's C-index: over all comparable pairs, how often is remaining life
    ranked in the right order?

    A pair is comparable when the ordering is known despite censoring, which
    excludes pairs whose shorter value is a censored lower bound. Rank quality is
    what a fleet manager consumes -- which truck to service first -- and unlike
    MAE it stays well defined under censoring. 0.5 is random, 1.0 perfect.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    censored = np.asarray(censored, dtype=bool)

    concordant = permissible = 0
    tied = 0
    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            # comparable only if the one with the smaller true value is observed
            if y_true[i] < y_true[j]:
                shorter, longer = i, j
            elif y_true[j] < y_true[i]:
                shorter, longer = j, i
            else:
                continue
            if censored[shorter]:
                continue  # true value could exceed the other; ordering unknown
            permissible += 1
            if y_pred[shorter] < y_pred[longer]:
                concordant += 1
            elif y_pred[shorter] == y_pred[longer]:
                tied += 1

    if permissible == 0:
        return float("nan")
    return (concordant + 0.5 * tied) / permissible
