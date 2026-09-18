"""CatBoost train entrypoint - mirrors train.py's shape (walk-forward CV
eval on folds[0:n-1]->fold_{n-1}, then a production fit on all folds) but
uses models/catboost_model.py instead of models/lgbm.py. See that module's
docstring for why CatBoost isn't folded into train.py directly.
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
from ecup_ltv.models.catboost_model import prepare_xy, save_bundle
from ecup_ltv.seed import set_all_seeds

import catboost as cb


def _fit_one_seed(X_train, y_train, X_val, y_val, params, num_boost_round, early_stopping_rounds, seed):
    p = dict(params)
    p["random_seed"] = seed
    p["iterations"] = num_boost_round
    if early_stopping_rounds and X_val is not None:
        p["early_stopping_rounds"] = early_stopping_rounds
    model = cb.CatBoostRegressor(**p)
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_cv(cfg: dict[str, Any], store_dir: Path) -> tuple[dict[str, Any], list, list[str]]:
    exp = cfg["experiment"]
    n_folds = cfg["n_folds"]
    train_idx = list(range(n_folds - 1))
    val_fold_name = f"fold_{n_folds-1:02d}"

    train_df = pl.concat([load_fold(store_dir, f"fold_{i:02d}") for i in train_idx], how="vertical")
    val_df = load_fold(store_dir, val_fold_name)
    feat_cols = get_feature_cols(train_df)

    X_train, y_train, _ = prepare_xy(train_df, feat_cols)
    X_val, y_val, y_val_raw = prepare_xy(val_df, feat_cols)

    early_stopping_rounds = exp.get("early_stopping_rounds")
    preds_log, best_iters, cv_models = [], [], []
    for seed in exp["seeds"]:
        m = _fit_one_seed(
            X_train, y_train, X_val, y_val, exp["catboost_params"], exp["num_boost_round"], early_stopping_rounds, seed
        )
        preds_log.append(m.predict(X_val))
        if early_stopping_rounds:
            best_iters.append(m.get_best_iteration())
        cv_models.append(m)

    pred = np.clip(np.expm1(np.mean(preds_log, axis=0)), 0, None)
    naive_pred = val_df["gmv_sum_30d"].to_numpy()

    result = {
        "cv_scheme": f"train=folds[0:{n_folds-1}] val={val_fold_name}",
        "n_features": len(feat_cols),
        "seeds": exp["seeds"],
        "rmsle": rmsle(y_val_raw, pred),
        "rmsle_naive_last30d_baseline": rmsle(y_val_raw, naive_pred),
        "gini": gini(y_val_raw, pred),
        "rmspe_total": rmspe_total(y_val_raw, pred),
    }
    if best_iters:
        result["cv_best_iterations"] = best_iters
        result["effective_num_boost_round"] = int(np.ceil(np.mean(best_iters)))
    return result, cv_models, feat_cols


def train(train_config_name: str) -> Path:
    t0 = time.time()
    cfg = load_config(train_config_name)
    exp = cfg["experiment"]
    set_all_seeds(exp["seeds"][0])

    store_dir = resolve_path(cfg, "features_store_dir")
    if not (store_dir / "meta.json").exists():
        raise FileNotFoundError(f"Feature store not found at {store_dir}. Run scripts/build_features.py first.")

    print(f"[{time.time()-t0:.1f}s] evaluating {exp['name']} via walk-forward CV")
    cv_metrics, cv_models, cv_feat_cols = evaluate_cv(cfg, store_dir)
    print(f"[{time.time()-t0:.1f}s] CV result: {cv_metrics}")

    cv_bundle_dir = resolve_path(cfg, "models_dir") / exp["name"] / "cv_only"
    save_bundle(cv_bundle_dir, cv_models, exp["seeds"], cv_feat_cols, cfg, cv_metrics)

    n_folds = cfg["n_folds"]
    full_train_df = pl.concat([load_fold(store_dir, f"fold_{i:02d}") for i in range(n_folds)], how="vertical")
    feat_cols = get_feature_cols(full_train_df)
    num_boost_round = cv_metrics.get("effective_num_boost_round", exp["num_boost_round"])

    X, y, _ = prepare_xy(full_train_df, feat_cols)
    models = []
    for seed in exp["seeds"]:
        print(f"[{time.time()-t0:.1f}s] fitting production model, seed={seed}, num_boost_round={num_boost_round}")
        models.append(_fit_one_seed(X, y, None, None, exp["catboost_params"], num_boost_round, None, seed))

    out_dir = resolve_path(cfg, "models_dir") / exp["name"]
    save_bundle(out_dir, models, exp["seeds"], feat_cols, cfg, cv_metrics)
    print(f"[{time.time()-t0:.1f}s] saved model bundle to {out_dir}")
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m ecup_ltv.train_catboost <train_config_name>")
        sys.exit(1)
    train(sys.argv[1])
