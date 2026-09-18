"""Train entrypoint: config in, model bundle out.

Two things happen here, and they are deliberately kept separate:

1. CV evaluation - walk-forward, train on all folds except the last,
   validate on the last untouched fold - to report an honest RMSLE/Gini/
   RMSPE estimate of what this config would score.
2. The production fit - trained on ALL folds pooled (more data -> better
   final model), saved as the versioned artifact. Its own held-out score is
   not known (there is no more held-out data left), which is exactly why
   step 1 exists: it is the only trustworthy performance number for this
   config, and it is stored in the bundle's metrics.json.

Inference (predict.py) never touches this file or the raw data - it only
reads the saved bundle.
"""

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ecup_ltv.build_features import get_feature_cols, load_fold
from ecup_ltv.config import load_config, resolve_path
from ecup_ltv.metrics import gini, rmsle, rmspe_total
from ecup_ltv.multi_fold_eval import _objective_override, _segment_metrics
from ecup_ltv.models.lgbm import (
    fit_one_seed,
    fit_one_seed_early_stopping,
    invert_prediction,
    prepare_xy,
    save_bundle,
    top_features_by_gain,
)
from ecup_ltv.seed import set_all_seeds
from ecup_ltv.weighting import compute_sample_weights


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


def evaluate_cv(cfg: dict[str, Any], store_dir: Path) -> tuple[dict[str, Any], list, list[str], str]:
    """Returns (metrics, cv_models, feat_cols, target_transform). cv_models
    are trained on folds[0:n_folds-1] only (fold_05 held out) - the caller
    may save them separately from the production bundle (which is trained
    on ALL folds and must never be used to "evaluate" on fold_05, since
    fold_05 is part of its own training data by then)."""
    exp = cfg["experiment"]
    n_folds = cfg["n_folds"]
    if n_folds < 2:
        raise ValueError("Need at least 2 CV folds to hold one out for validation")

    train_idx = list(range(n_folds - 1))
    val_fold_name = f"fold_{n_folds-1:02d}"
    train_df = pl.concat([load_fold(store_dir, f"fold_{i:02d}") for i in train_idx], how="vertical")
    val_df = load_fold(store_dir, val_fold_name)
    all_feat_cols = get_feature_cols(train_df)

    feat_cols = _select_features(train_df, all_feat_cols, exp)
    target_transform = exp.get("target_transform", "log1p")

    X_train, y_train, y_train_raw = prepare_xy(train_df, feat_cols, target_transform)
    X_val, y_val, y_val_raw = prepare_xy(val_df, feat_cols, target_transform)

    sample_weight = compute_sample_weights(
        y_train_raw,
        exp.get("sample_weight_scheme"),
        alpha=exp.get("sample_weight_alpha", 0.0),
        threshold=exp.get("sample_weight_threshold", 200.0),
        boost=exp.get("sample_weight_boost", 4.0),
    )

    obj_override = _objective_override(exp)
    early_stopping_rounds = exp.get("early_stopping_rounds")
    preds_transformed = []
    best_iterations = []
    cv_models = []
    for seed in exp["seeds"]:
        if early_stopping_rounds:
            m = fit_one_seed_early_stopping(
                X_train, y_train, X_val, y_val, feat_cols, exp["lgbm_params"],
                exp["num_boost_round"], early_stopping_rounds, seed,
            )
            best_iterations.append(m.best_iteration)
            preds_transformed.append(m.predict(X_val, num_iteration=m.best_iteration))
        else:
            m = fit_one_seed(
                X_train, y_train, feat_cols, exp["lgbm_params"], exp["num_boost_round"], seed,
                sample_weight=sample_weight, objective_override=obj_override,
            )
            preds_transformed.append(m.predict(X_val))
        cv_models.append(m)
    pred = invert_prediction(np.mean(preds_transformed, axis=0), target_transform)

    naive_pred = val_df["gmv_sum_30d"].to_numpy()

    result = {
        "cv_scheme": f"train=folds[0:{n_folds-1}] val={val_fold_name} (walk-forward, last fold held out)",
        "n_features_total": len(all_feat_cols),
        "n_features_used": len(feat_cols),
        "seeds": exp["seeds"],
        "rmsle": rmsle(y_val_raw, pred),
        "rmsle_naive_last30d_baseline": rmsle(y_val_raw, naive_pred),
        "gini": gini(y_val_raw, pred),
        "rmspe_total": rmspe_total(y_val_raw, pred),
    }
    result.update(_segment_metrics(y_val_raw, pred))
    if best_iterations:
        # mean best_iteration across seeds, rounded up, used as the production
        # num_boost_round since the final fit has no held-out set of its own
        result["cv_best_iterations"] = best_iterations
        result["effective_num_boost_round"] = int(np.ceil(np.mean(best_iterations)))
    return result, cv_models, feat_cols, target_transform


def train(train_config_name: str) -> Path:
    t0 = time.time()
    cfg = load_config(train_config_name)
    exp = cfg["experiment"]
    set_all_seeds(exp["seeds"][0])

    store_dir = resolve_path(cfg, "features_store_dir")
    meta_path = store_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Feature store not found at {store_dir}. Run `python scripts/build_features.py` first."
        )

    print(f"[{time.time()-t0:.1f}s] evaluating {exp['name']} via walk-forward CV")
    cv_metrics, cv_models, cv_feat_cols, target_transform = evaluate_cv(cfg, store_dir)
    print(f"[{time.time()-t0:.1f}s] CV result: {cv_metrics}")

    # saved separately from the production bundle below: these models were
    # trained on folds[0:n_folds-1] only, so they (and only they) are valid
    # for out-of-sample evaluation on fold_05, e.g. when scoring an ensemble
    # blend of several experiments (see blend.py) without leaking fold_05
    # into any blended model's own training data.
    cv_bundle_dir = resolve_path(cfg, "models_dir") / exp["name"] / "cv_only"
    save_bundle(cv_bundle_dir, cv_models, exp["seeds"], cv_feat_cols, cfg, cv_metrics, target_transform)

    n_folds = cfg["n_folds"]
    full_train_df = pl.concat([load_fold(store_dir, f"fold_{i:02d}") for i in range(n_folds)], how="vertical")
    all_feat_cols = get_feature_cols(full_train_df)
    feat_cols = _select_features(full_train_df, all_feat_cols, exp)

    # the production fit pools all folds, so there is no held-out set left to
    # early-stop against; if CV determined a data-driven round count, reuse
    # it here instead of the config's num_boost_round ceiling.
    num_boost_round = cv_metrics.get("effective_num_boost_round", exp["num_boost_round"])

    X, y, y_raw = prepare_xy(full_train_df, feat_cols, target_transform)
    sample_weight = compute_sample_weights(
        y_raw,
        exp.get("sample_weight_scheme"),
        alpha=exp.get("sample_weight_alpha", 0.0),
        threshold=exp.get("sample_weight_threshold", 200.0),
        boost=exp.get("sample_weight_boost", 4.0),
    )
    models = []
    for seed in exp["seeds"]:
        print(f"[{time.time()-t0:.1f}s] fitting production model, seed={seed}, num_boost_round={num_boost_round}")
        models.append(fit_one_seed(
            X, y, feat_cols, exp["lgbm_params"], num_boost_round, seed,
            sample_weight=sample_weight, objective_override=_objective_override(exp),
        ))

    out_dir = resolve_path(cfg, "models_dir") / exp["name"]
    save_bundle(out_dir, models, exp["seeds"], feat_cols, cfg, cv_metrics, target_transform)
    print(f"[{time.time()-t0:.1f}s] saved model bundle to {out_dir}")
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m ecup_ltv.train <train_config_name>  (e.g. train_v1_baseline)")
        sys.exit(1)
    train(sys.argv[1])
