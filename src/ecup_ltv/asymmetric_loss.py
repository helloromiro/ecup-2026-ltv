"""Asymmetric L2 objective for LightGBM: penalizes underestimation more
than overestimation, aimed at the same systematic high-value
underestimation bias as sample-weighting
(experiments/highvalue_reweighting/notes.md) - but through a different
mechanism. Value-based reweighting fixed a weight per row from its target
magnitude before training even starts, and it worked (tail RMSLE dropped
15-30%) but always at a larger cost to the bulk (89% of rows are
small/zero and dominate the loss even in log1p-space), with no blend
weight small enough to net out positive.

This objective instead penalizes by the CURRENT residual's *sign* at
every boosting iteration, regardless of the row's target magnitude: a
small-target row that happens to be under-predicted right now pays the
same penalty as a large-target one, and a large-target row that is
already correctly (or over-) predicted pays nothing extra. The
regularization sweep
(experiments/highvalue_reweighting/regularization_sweep_notes.md) ruled
out tree capacity as the bottleneck - this changes the training economics
directly instead.
"""

from typing import Callable

import numpy as np


def make_asymmetric_l2_objective(under_penalty: float) -> Callable[[np.ndarray, "lgb.Dataset"], tuple]:
    """under_penalty=1.0 recovers standard L2. under_penalty>1.0 makes
    underestimation (pred < true, i.e. residual = pred-true < 0) cost
    `under_penalty`x as much (both gradient and curvature) as
    overestimation, symmetric case."""

    def objective(preds: np.ndarray, train_data) -> tuple[np.ndarray, np.ndarray]:
        y = train_data.get_label()
        residual = preds - y
        weight = np.where(residual < 0, under_penalty, 1.0)
        grad = weight * residual
        hess = weight
        return grad, hess

    return objective
