"""
Censoring-aware loss and metrics.

Needs no artifacts, so it runs in CI.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.survival import (  # noqa: E402
    censored_l2_objective,
    censored_mae,
    concordance_index,
)

# ---------------------------------------------------------------- the objective

def test_observed_rows_behave_like_squared_error():
    obj = censored_l2_objective(np.array([False, False]))
    grad, hess = obj(np.array([10.0, 20.0]), np.array([12.0, 15.0]))
    np.testing.assert_allclose(grad, [4.0, -10.0])   # 2 * (pred - y)
    np.testing.assert_allclose(hess, [2.0, 2.0])


def test_censored_over_prediction_is_free():
    """The whole point. A censored label is a LOWER BOUND, so predicting more
    life than it states contradicts nothing and must carry no gradient."""
    obj = censored_l2_objective(np.array([True]))
    grad, _ = obj(np.array([10.0]), np.array([99.0]))
    assert grad[0] == 0.0


def test_censored_under_prediction_is_penalised():
    """Predicting less than a known lower bound does contradict the data."""
    obj = censored_l2_objective(np.array([True]))
    grad, _ = obj(np.array([10.0]), np.array([4.0]))
    assert grad[0] == pytest.approx(-12.0)   # 2 * (4 - 10), pushes the prediction up


def test_gradient_pushes_censored_predictions_upward_only():
    """Sign check across a sweep: the censored gradient is never positive, so
    the loss can only ever raise a prediction, never lower it."""
    obj = censored_l2_objective(np.full(5, True))
    y = np.full(5, 50.0)
    grad, _ = obj(y, np.array([10.0, 40.0, 50.0, 60.0, 200.0]))
    assert (grad <= 0).all()
    assert (grad[3:] == 0).all()      # at or above the bound: no penalty


def test_hessian_is_never_zero():
    """A zero hessian makes LightGBM's Newton step undefined and the split gain
    degenerate, so curvature is deliberately overstated where the hinge is
    inactive."""
    obj = censored_l2_objective(np.array([True, True, False]))
    _, hess = obj(np.array([10.0, 10.0, 10.0]), np.array([99.0, 1.0, 5.0]))
    assert (hess > 0).all()


# ---------------------------------------------------------------- the metrics

def test_censored_mae_ignores_over_prediction():
    y = np.array([10.0, 10.0])
    pred = np.array([99.0, 99.0])
    assert censored_mae(y, pred, np.array([True, True])) == 0.0
    assert censored_mae(y, pred, np.array([False, False])) == pytest.approx(89.0)


def test_concordance_is_perfect_when_ranking_is_right():
    y = np.array([10.0, 20.0, 30.0])
    assert concordance_index(y, np.array([1.0, 2.0, 3.0]),
                             np.zeros(3, dtype=bool)) == 1.0


def test_concordance_is_zero_when_ranking_is_reversed():
    y = np.array([10.0, 20.0, 30.0])
    assert concordance_index(y, np.array([3.0, 2.0, 1.0]),
                             np.zeros(3, dtype=bool)) == 0.0


def test_concordance_skips_incomparable_pairs():
    """If the SHORTER of a pair is censored, its true value could exceed the
    other's, so the ordering is unknown and the pair must not be scored."""
    y = np.array([10.0, 20.0])
    # the shorter value is censored -> no comparable pairs at all
    assert np.isnan(concordance_index(y, np.array([1.0, 2.0]),
                                      np.array([True, False])))
    # the LONGER one censored is still comparable: 10 observed < 20+
    assert concordance_index(y, np.array([1.0, 2.0]),
                             np.array([False, True])) == 1.0


def test_concordance_handles_ties_as_half_credit():
    y = np.array([10.0, 20.0])
    assert concordance_index(y, np.array([5.0, 5.0]),
                             np.zeros(2, dtype=bool)) == 0.5
