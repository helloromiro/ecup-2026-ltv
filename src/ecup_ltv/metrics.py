"""Competition primary metric (RMSLE) plus the tie-breaker metrics named in
the official rules (point 4.5.1: Gini over per-user predictions, RMSPE over
the total predicted GMV across all users) - kept here so local validation
reports the same numbers the jury will look at, not just the leaderboard metric.
"""

import numpy as np


def rmsle(y_true, y_pred) -> float:
    y_true = np.clip(np.asarray(y_true, dtype=np.float64), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def gini(y_true, y_pred) -> float:
    """Normalized Gini coefficient of predictions ranked by y_pred, a measure
    of how well the model orders users by value (used by the jury as a
    tie-breaker on top of the leaderboard RMSLE rank)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    def _gini_raw(actual, pred):
        order = np.argsort(-pred, kind="mergesort")
        actual_sorted = actual[order]
        cum = np.cumsum(actual_sorted)
        total = cum[-1]
        if total == 0:
            return 0.0
        lorenz = cum / total
        n = len(actual)
        return 1.0 - 2.0 * np.sum(lorenz) / n + 1.0 / n

    g_pred = _gini_raw(y_true, y_pred)
    g_perfect = _gini_raw(y_true, y_true)
    return float(g_pred / g_perfect) if g_perfect != 0 else 0.0


def rmspe_total(y_true, y_pred) -> float:
    """Relative error between the total predicted and true GMV summed across
    all users - the aggregate-level tie-breaker from the competition rules."""
    total_true = float(np.sum(y_true))
    total_pred = float(np.sum(y_pred))
    if total_true == 0:
        return 0.0
    return abs(total_pred - total_true) / total_true
