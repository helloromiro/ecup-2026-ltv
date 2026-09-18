"""Multi-fold CV evaluation - the reliability fix for the single-fold
`evaluate_cv` in train.py.

Yesterday's session ran 10+ experiments against the same single held-out
fold (fold_05) to decide which config was "better". One of those decisions
(v10) later failed on the real public leaderboard despite winning on that
fold - a textbook case of adaptive overfitting to one validation split
after many rounds of comparison against it (see
experiments/cv_vs_leaderboard_gap.md for the full writeup).

This evaluates a config against SEVERAL independent walk-forward splits
(each with strictly more training history than the last, matching the
walk-forward structure used everywhere else in this repo) and reports the
mean *and* the spread across folds. A config that wins by less than the
fold-to-fold spread is not a real win - it is noise, and should not be
used to make decisions, submissions, or ensemble-weight choices.
"""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ecup_ltv.asymmetric_loss import make_asymmetric_l2_objective
from ecup_ltv.build_features import get_feature_cols, load_fold
from ecup_ltv.metrics import gini, rmsle, rmspe_total
from ecup_ltv.models.lgbm import fit_one_seed, invert_prediction, prepare_xy, top_features_by_gain
from ecup_ltv.weighting import compute_sample_weights


def _objective_override(exp: dict[str, Any]):
    under_penalty = exp.get("asymmetric_under_penalty")
    return make_asymmetric_l2_objective(under_penalty) if under_penalty else None


def _select_features(train_df: pl.DataFrame, all_feat_cols: list[str], exp: dict[str, Any]) -> list[str]:
    top_n = exp.get("top_n_features")
    if not top_n:
        return all_feat_cols
    target_transform = exp.get("target_transform", "log1p")
    X, y, _ = prepare_xy(train_df, all_feat_cols, target_transform)
    screen_model = fit_one_seed(
        X, y, all_feat_cols, exp["lgbm_params"], exp["num_boost_round"], exp["seeds"][0],
        objective_override=_objective_override(exp),
    )
    return top_features_by_gain(screen_model, top_n)


# thresholds (raw GMV) for the segment-level RMSLE breakdown reported
# alongside the overall metric - the identified failure mode
# (experiments/highvalue_reweighting/notes.md) is specifically about the
# true>=200 segment, so every multi-fold run tracks it whether or not the
# config being tested targets it, to catch regressions there from unrelated
# changes too.
SEGMENT_THRESHOLDS = [50, 200]


def _segment_metrics(y_val_raw: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    out = {}
    for thr in SEGMENT_THRESHOLDS:
        mask = y_val_raw >= thr
        if mask.sum() == 0:
            continue
        out[f"rmsle_ge{thr}"] = rmsle(y_val_raw[mask], pred[mask])
        out[f"n_ge{thr}"] = int(mask.sum())
        out[f"median_true_ge{thr}"] = float(np.median(y_val_raw[mask]))
        out[f"median_pred_ge{thr}"] = float(np.median(pred[mask]))
    return out


def evaluate_one_split(cfg: dict[str, Any], store_dir: Path, train_idx: list[int], val_fold_idx: int) -> dict[str, Any]:
    exp = cfg["experiment"]
    target_transform = exp.get("target_transform", "log1p")

    train_df = pl.concat([load_fold(store_dir, f"fold_{i:02d}") for i in train_idx], how="vertical")
    val_df = load_fold(store_dir, f"fold_{val_fold_idx:02d}")
    all_feat_cols = get_feature_cols(train_df)
    feat_cols = _select_features(train_df, all_feat_cols, exp)

    X_train, y_train, y_train_raw = prepare_xy(train_df, feat_cols, target_transform)
    X_val, _, y_val_raw = prepare_xy(val_df, feat_cols, target_transform)

    weight_scheme = exp.get("sample_weight_scheme")
    sample_weight = compute_sample_weights(
        y_train_raw,
        weight_scheme,
        alpha=exp.get("sample_weight_alpha", 0.0),
        threshold=exp.get("sample_weight_threshold", 200.0),
        boost=exp.get("sample_weight_boost", 4.0),
    )

    obj_override = _objective_override(exp)
    preds = []
    for seed in exp["seeds"]:
        m = fit_one_seed(
            X_train, y_train, feat_cols, exp["lgbm_params"], exp["num_boost_round"], seed,
            sample_weight=sample_weight, objective_override=obj_override,
        )
        preds.append(m.predict(X_val))
    pred = invert_prediction(np.mean(preds, axis=0), target_transform)

    result = {
        "train_folds": train_idx,
        "val_fold": val_fold_idx,
        "rmsle": rmsle(y_val_raw, pred),
        "gini": gini(y_val_raw, pred),
        "rmspe_total": rmspe_total(y_val_raw, pred),
        "n_features": len(feat_cols),
    }
    result.update(_segment_metrics(y_val_raw, pred))
    return result


def evaluate_multi_fold(cfg: dict[str, Any], store_dir: Path, val_fold_indices: list[int] = (3, 4, 5)) -> dict[str, Any]:
    """Walk-forward rounds: round i trains on folds[0:val_fold_indices[i]]
    and validates on val_fold_indices[i] - e.g. with the default (3,4,5):
    train=[0,1,2] val=3, train=[0,1,2,3] val=4, train=[0,1,2,3,4] val=5.
    Each round is an independent, never-before-used-for-this-config split."""
    per_fold = [evaluate_one_split(cfg, store_dir, list(range(v)), v) for v in val_fold_indices]
    rmsles = [r["rmsle"] for r in per_fold]
    out = {
        "per_fold": per_fold,
        "rmsle_mean": float(np.mean(rmsles)),
        "rmsle_std": float(np.std(rmsles)),
        "rmsle_min": float(np.min(rmsles)),
        "rmsle_max": float(np.max(rmsles)),
        "gini_mean": float(np.mean([r["gini"] for r in per_fold])),
        "rmspe_total_mean": float(np.mean([r["rmspe_total"] for r in per_fold])),
    }
    for thr in SEGMENT_THRESHOLDS:
        key = f"rmsle_ge{thr}"
        if all(key in r for r in per_fold):
            out[f"{key}_mean"] = float(np.mean([r[key] for r in per_fold]))
    return out
