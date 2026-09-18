"""Optional training-time sample weighting, aimed at the tail-compression
bias identified in error analysis on v3: with log1p target + L2 loss and
tree regularization (min_data_in_leaf, num_leaves caps), the ~89% of rows
with small/zero target dominate leaf statistics, and high-value active
users get systematically pulled toward their (much smaller) neighbors'
predictions. Reweighting the loss toward high-value rows is a direct lever
on that mechanism - unlike features (BTYD/momentum, already tried and
neutral - see experiments/v12_btyd_features), it changes what the model is
optimizing for, not what it can see.

Weights are always renormalized to mean 1 so the effective learning rate /
gradient scale stays comparable to the unweighted config - only the
*distribution* of emphasis across rows changes, not the overall step size.
"""

from typing import Optional

import numpy as np


def compute_sample_weights(
    y_raw: np.ndarray,
    scheme: Optional[str],
    alpha: float = 0.0,
    threshold: float = 200.0,
    boost: float = 4.0,
) -> Optional[np.ndarray]:
    if not scheme or scheme == "none":
        return None
    y = np.clip(np.asarray(y_raw, dtype=np.float64), 0, None)

    if scheme == "value_pow":
        # continuous: w ~ (1+y)^alpha, no hard cutoff - a user with 2x the
        # target gets 2^alpha more weight than one right next to them.
        w = np.power(1.0 + y, alpha)
    elif scheme == "threshold_boost":
        # step: flat `boost` extra weight for rows at/above `threshold`,
        # everything else weight 1 - simplest possible test of "does the
        # tail even benefit from more gradient attention at all".
        w = np.where(y >= threshold, boost, 1.0)
    else:
        raise ValueError(f"unknown sample_weight scheme {scheme!r}")

    w = w / w.mean()
    return w.astype(np.float32)
