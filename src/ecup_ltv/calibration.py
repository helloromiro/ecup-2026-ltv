"""Isotonic recalibration of a model's log1p predictions.

Error analysis on v3 (see experiments/error_analysis notes) found the
single biggest driver of RMSLE was systematic under-prediction of
high-value active users (median predicted GMV among active users: 24.8,
median true: 61.4) - not the zero/non-zero split, which was roughly
balanced. An isotonic regression fit on (predicted_log1p -> true_log1p)
from out-of-fold predictions corrects this monotonically without
assuming a parametric form for the bias.

Validated out-of-time in both directions (not resampling one fold - see
experiments/prediction_calibration/notes.md):
- calibrator fit on fold_05, applied to a fold_4-held-out model: RMSLE
  1.6974 -> 1.6856
- calibrator fit on fold_4, applied to a fold_3-held-out model: RMSLE
  1.7376 -> 1.7247

This is the strongest, most rigorously out-of-time-validated effect
found in this repo's research so far.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_calibrator(pred_log1p_oof: np.ndarray, y_true_log1p: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(pred_log1p_oof, y_true_log1p)
    return calibrator


def apply_calibrator(calibrator: IsotonicRegression, pred_log1p: np.ndarray) -> np.ndarray:
    """Returns calibrated raw (non-log) predictions, clipped >= 0."""
    calibrated_log1p = calibrator.predict(pred_log1p)
    return np.clip(np.expm1(calibrated_log1p), 0, None)


def save_calibrator(calibrator: IsotonicRegression, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)


def load_calibrator(path: Path) -> IsotonicRegression:
    with open(path, "rb") as f:
        return pickle.load(f)
