"""Blend multiple already-trained model bundles.

Averaging happens in log1p-space (i.e. the blend is a weighted geometric
mean of each bundle's raw prediction), not raw-space, because the
competition metric (RMSLE) is itself computed in log-space - this matches
how within-bundle seed-bagging already averages in each bundle's own
transformed training space (see models/lgbm.py).

This only ever combines existing bundles produced by train.py; it does not
train anything itself, keeping the train/inference separation intact.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ecup_ltv.metrics import gini, rmsle, rmspe_total


def _bundle_model_type(bundle_dir: Path) -> str:
    with open(bundle_dir / "meta.json") as f:
        meta = json.load(f)
    # older bundles (trained before model_type was recorded) are all LightGBM
    return meta.get("model_type", "lightgbm")


def _bundle_log1p_pred(bundle_dir: Path, df: pl.DataFrame) -> np.ndarray:
    model_type = _bundle_model_type(bundle_dir)
    if model_type == "lightgbm":
        from ecup_ltv.models.lgbm import invert_prediction, load_bundle, predict_transformed_bagged

        models, feat_cols, target_transform = load_bundle(bundle_dir)
        X = df.select(feat_cols).to_numpy()
        pred_transformed = predict_transformed_bagged(models, X)
        pred_raw = invert_prediction(pred_transformed, target_transform)
    elif model_type == "catboost":
        from ecup_ltv.models.catboost_model import load_bundle, predict_log_bagged

        models, feat_cols = load_bundle(bundle_dir)
        X = df.select(feat_cols).to_numpy()
        pred_log = predict_log_bagged(models, X)
        pred_raw = np.clip(np.expm1(pred_log), 0, None)
    else:
        raise ValueError(f"unknown model_type {model_type!r} in {bundle_dir}")
    return np.log1p(pred_raw)


def blend_predict(models_dir: Path, bundle_names: list[str], df: pl.DataFrame, weights: list[float] = None) -> np.ndarray:
    if weights is None:
        weights = [1.0] * len(bundle_names)
    assert len(weights) == len(bundle_names)
    log1p_preds = [_bundle_log1p_pred(models_dir / name, df) for name in bundle_names]
    blended_log1p = np.average(log1p_preds, axis=0, weights=weights)
    return np.clip(np.expm1(blended_log1p), 0, None)


def evaluate_blend_on_val(
    cfg: dict[str, Any], store_dir: Path, models_dir: Path, bundle_names: list[str], weights: list[float] = None
) -> dict[str, Any]:
    """Honest blend evaluation on fold_05. Uses each experiment's cv_only/
    sub-bundle (trained on folds[0:n_folds-1], see train.py) rather than its
    production bundle - the production bundle was trained on ALL folds
    including fold_05 and would silently leak into this "held-out" score."""
    from ecup_ltv.build_features import load_fold

    n_folds = cfg["n_folds"]
    val_df = load_fold(store_dir, f"fold_{n_folds-1:02d}")
    y_val_raw = val_df["target"].to_numpy().astype(np.float64)

    cv_bundle_dirs = [models_dir / name / "cv_only" for name in bundle_names]
    log1p_preds = [_bundle_log1p_pred(d, val_df) for d in cv_bundle_dirs]
    blended_log1p = np.average(log1p_preds, axis=0, weights=weights)
    pred = np.clip(np.expm1(blended_log1p), 0, None)

    return {
        "bundles": bundle_names,
        "weights": weights or [1.0] * len(bundle_names),
        "rmsle": rmsle(y_val_raw, pred),
        "gini": gini(y_val_raw, pred),
        "rmspe_total": rmspe_total(y_val_raw, pred),
    }
