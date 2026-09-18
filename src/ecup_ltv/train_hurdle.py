"""Two-stage "hurdle" model: a classifier for P(target > 0) times a
regressor fit only on active (target > 0) rows.

The task's target is zero-inflated (many users buy nothing in the next 30
days) and heavily right-skewed among buyers - a hurdle model is the
standard approach for exactly this shape, and the earlier research phase's
quick attempt at it wasn't conclusively worse or better (soft-multiply
combination came close to the plain regressor). This retries it with
proper tuning and this repo's multi-fold validation instead of a single
untuned comparison.

Combination is "soft": final_pred = p_active * regressor_pred, in raw
(non-log) space, matching what the earlier attempt found worked best
among the combination strategies it tried (soft multiply vs hard
threshold).
"""

import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from ecup_ltv.build_features import get_feature_cols, load_fold
from ecup_ltv.config import load_config, resolve_path
from ecup_ltv.metrics import gini, rmsle, rmspe_total
from ecup_ltv.seed import set_all_seeds


def _fit_classifier(X, y_bin, params, num_boost_round, seed):
    p = dict(params)
    p["seed"] = seed
    ds = lgb.Dataset(X, label=y_bin, free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=num_boost_round)


def _fit_regressor(X, y_log, params, num_boost_round, seed):
    p = dict(params)
    p["seed"] = seed
    ds = lgb.Dataset(X, label=y_log, free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=num_boost_round)


def evaluate_cv(cfg: dict[str, Any], store_dir: Path) -> dict[str, Any]:
    exp = cfg["experiment"]
    n_folds = cfg["n_folds"]
    train_idx = list(range(n_folds - 1))
    val_fold_name = f"fold_{n_folds-1:02d}"

    train_df = pl.concat([load_fold(store_dir, f"fold_{i:02d}") for i in train_idx], how="vertical")
    val_df = load_fold(store_dir, val_fold_name)
    feat_cols = get_feature_cols(train_df)

    X_train = train_df.select(feat_cols).to_numpy()
    y_train_raw = train_df["target"].to_numpy().astype(np.float64)
    y_train_bin = (y_train_raw > 0).astype(np.float32)
    y_train_log = np.log1p(np.clip(y_train_raw, 0, None)).astype(np.float32)

    X_val = val_df.select(feat_cols).to_numpy()
    y_val_raw = val_df["target"].to_numpy().astype(np.float64)

    clf_seeds = exp["classifier_seeds"]
    reg_seeds = exp["regressor_seeds"]

    p_active_preds = []
    for seed in clf_seeds:
        clf = _fit_classifier(X_train, y_train_bin, exp["classifier_params"], exp["classifier_rounds"], seed)
        p_active_preds.append(clf.predict(X_val))
    p_active = np.mean(p_active_preds, axis=0)

    active_mask = y_train_raw > 0
    X_train_act, y_train_act_log = X_train[active_mask], y_train_log[active_mask]

    reg_preds_log = []
    for seed in reg_seeds:
        reg = _fit_regressor(X_train_act, y_train_act_log, exp["regressor_params"], exp["regressor_rounds"], seed)
        reg_preds_log.append(reg.predict(X_val))
    reg_pred = np.clip(np.expm1(np.mean(reg_preds_log, axis=0)), 0, None)

    pred = p_active * reg_pred

    naive_pred = val_df["gmv_sum_30d"].to_numpy()
    plain_regressor_only = reg_pred  # reference: regressor alone, no hurdle gating

    return {
        "cv_scheme": f"train=folds[0:{n_folds-1}] val={val_fold_name}",
        "n_features": len(feat_cols),
        "rmsle_hurdle": rmsle(y_val_raw, pred),
        "rmsle_regressor_only_reference": rmsle(y_val_raw, plain_regressor_only),
        "rmsle_naive_last30d_baseline": rmsle(y_val_raw, naive_pred),
        "gini_hurdle": gini(y_val_raw, pred),
        "rmspe_total_hurdle": rmspe_total(y_val_raw, pred),
        "mean_p_active": float(p_active.mean()),
        "true_active_rate": float((y_val_raw > 0).mean()),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m ecup_ltv.train_hurdle <train_config_name>")
        sys.exit(1)
    t0 = time.time()
    cfg = load_config(sys.argv[1])
    set_all_seeds(cfg["experiment"]["classifier_seeds"][0])
    store_dir = resolve_path(cfg, "features_store_dir")
    result = evaluate_cv(cfg, store_dir)
    print(f"[{time.time()-t0:.1f}s] {result}")
