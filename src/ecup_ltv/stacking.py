"""Production non-linear stacking: a small, heavily-regularized LightGBM
meta-learner combining out-of-fold predictions from several base bundles
(+ an isotonic-calibrated Tweedie prediction), validated in
experiments/nonlinear_stacking/notes.md via 6 independent split-half
checks (3 seeds x 2 directions) - every single one showed the
meta-learner beating the best individual base model (v3) by a consistent
~0.005 RMSLE, unlike the linear (NNLS) stacking attempt which flipped
sign between split directions and failed to reproduce on the public
leaderboard.

The meta-learner is fit once on ALL of fold_05's cv_only (out-of-fold)
predictions - by this point the 6-split check has already established
that this combination generalizes, so no further split is held back from
the final meta-model fit, matching how every other "production" model in
this repo pools all available CV data for its final fit.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from ecup_ltv.blend import _bundle_log1p_pred

BASE_BUNDLES = ["v2_seed_bagging", "ensemble_b_exact", "v3_top100_5seed", "catboost_diverse"]
TWEEDIE_BUNDLE = "v5_tweedie"

META_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=7,
    max_depth=3,
    min_data_in_leaf=200,
    lambda_l2=10.0,
    num_threads=4,
    verbosity=-1,
    seed=42,
)
META_ROUNDS = 150


def _feature_matrix(models_dir: Path, bundle_suffix: str, df: pl.DataFrame, tweedie_calibrator: IsotonicRegression = None):
    """bundle_suffix is '' for production bundles or '/cv_only' for the
    out-of-fold ones used to fit the meta-learner."""
    cols = []
    for name in BASE_BUNDLES:
        cols.append(_bundle_log1p_pred(models_dir / f"{name}{bundle_suffix}", df))
    tweedie_log1p = _bundle_log1p_pred(models_dir / f"{TWEEDIE_BUNDLE}{bundle_suffix}", df)
    tweedie_raw = np.expm1(tweedie_log1p)
    if tweedie_calibrator is not None:
        cols.append(tweedie_calibrator.predict(tweedie_raw))
    else:
        cols.append(np.log1p(tweedie_raw))  # placeholder, only used when fitting the calibrator itself
    return np.column_stack(cols)


def fit_stack(models_dir: Path, fold05_df: pl.DataFrame, out_dir: Path) -> dict[str, Any]:
    y_raw = fold05_df["target"].to_numpy().astype(np.float64)
    y_log = np.log1p(np.clip(y_raw, 0, None))

    tweedie_log1p = _bundle_log1p_pred(models_dir / f"{TWEEDIE_BUNDLE}/cv_only", fold05_df)
    tweedie_raw = np.expm1(tweedie_log1p)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(tweedie_raw, y_log)  # fit on all of fold_05 - final production calibrator

    X = _feature_matrix(models_dir, "/cv_only", fold05_df, calibrator)
    names = BASE_BUNDLES + ["tweedie_calibrated"]

    ds = lgb.Dataset(X, label=y_log, feature_name=names, free_raw_data=True)
    meta_model = lgb.train(META_PARAMS, ds, num_boost_round=META_ROUNDS)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_model.save_model(str(out_dir / "meta_model.txt"))
    with open(out_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    with open(out_dir / "base_bundles.json", "w") as f:
        json.dump({"lgb_bundles": BASE_BUNDLES, "tweedie_bundle": TWEEDIE_BUNDLE}, f, indent=2)

    imp = dict(zip(names, meta_model.feature_importance(importance_type="gain").tolist()))
    return {"feature_importance_gain": imp}


def load_stack(out_dir: Path) -> tuple[lgb.Booster, IsotonicRegression]:
    meta_model = lgb.Booster(model_file=str(out_dir / "meta_model.txt"))
    with open(out_dir / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    return meta_model, calibrator


def predict_stack(models_dir: Path, stack_dir: Path, df: pl.DataFrame) -> np.ndarray:
    meta_model, calibrator = load_stack(stack_dir)
    X = _feature_matrix(models_dir, "", df, calibrator)
    pred_log = meta_model.predict(X)
    return np.clip(np.expm1(pred_log), 0, None)
