"""LightGBM training + a uniform save/load bundle format.

A bundle is a directory containing one booster text file per seed (bagging
happens by averaging predictions in log-space across boosters, never inside
a single model), the exact feature column list used, and the resolved
config the bundle was produced from - enough to regenerate or audit the
run without going back to this codebase's git history.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml


def fit_one_seed(
    X: np.ndarray,
    y_log: np.ndarray,
    feat_cols: list[str],
    lgbm_params: dict[str, Any],
    num_boost_round: int,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
    objective_override: Optional[Any] = None,
) -> lgb.Booster:
    """objective_override: a custom LightGBM objective callable
    (preds, Dataset) -> (grad, hess), e.g. from asymmetric_loss.py - takes
    the place of params["objective"] when given. LightGBM 4.x's train()
    has no separate fobj argument; a custom objective is passed as the
    "objective" param value itself (verified against the installed 4.6.0)."""
    params = dict(lgbm_params)
    params["seed"] = seed
    if objective_override is not None:
        params["objective"] = objective_override
    train_set = lgb.Dataset(X, label=y_log, weight=sample_weight, feature_name=feat_cols, free_raw_data=True)
    return lgb.train(params, train_set, num_boost_round=num_boost_round)


def fit_one_seed_early_stopping(
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_val: np.ndarray,
    y_val_log: np.ndarray,
    feat_cols: list[str],
    lgbm_params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    seed: int,
) -> lgb.Booster:
    """Used only during CV evaluation to pick a data-driven round count
    (booster.best_iteration) instead of guessing num_boost_round - the
    production fit in train.py then reuses that count with no validation
    set held out, since by then all folds are pooled into training data."""
    params = dict(lgbm_params)
    params["seed"] = seed
    train_set = lgb.Dataset(X_train, label=y_train_log, feature_name=feat_cols, free_raw_data=True)
    val_set = lgb.Dataset(X_val, label=y_val_log, reference=train_set, free_raw_data=False)
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )


def predict_transformed_bagged(models: list[lgb.Booster], X: np.ndarray) -> np.ndarray:
    """Average predictions across seeds in the model's own training space
    (log1p or raw, see `target_transform`) - averaging must happen before
    inverting the transform, not after, or bagging degenerates into a
    biased average-of-exponentials for the log1p case."""
    preds = [m.predict(X) for m in models]
    return np.mean(preds, axis=0)


def invert_prediction(pred_transformed: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "log1p":
        return np.clip(np.expm1(pred_transformed), 0, None)
    if target_transform == "sqrt":
        return np.clip(pred_transformed, 0, None) ** 2
    if target_transform == "none":
        return np.clip(pred_transformed, 0, None)
    raise ValueError(f"unknown target_transform {target_transform!r}")


def top_features_by_gain(model: lgb.Booster, top_n: int) -> list[str]:
    names = model.feature_name()
    gain = model.feature_importance(importance_type="gain")
    ranked = [n for n, _ in sorted(zip(names, gain), key=lambda x: -x[1])]
    return ranked[:top_n]


def save_bundle(
    out_dir: Path,
    models: list[lgb.Booster],
    seeds: list[int],
    feat_cols: list[str],
    resolved_config: dict[str, Any],
    cv_metrics: dict[str, Any],
    target_transform: str = "log1p",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed, model in zip(seeds, models):
        model.save_model(str(out_dir / f"model_seed{seed}.txt"))
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(resolved_config, f, allow_unicode=True, sort_keys=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(cv_metrics, f, indent=2)
    meta = {
        "seeds": seeds,
        "target_transform": target_transform,
        "model_type": "lightgbm",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_bundle(bundle_dir: Path) -> tuple[list[lgb.Booster], list[str], str]:
    with open(bundle_dir / "meta.json") as f:
        meta = json.load(f)
    with open(bundle_dir / "feature_cols.json") as f:
        feat_cols = json.load(f)
    models = [lgb.Booster(model_file=str(bundle_dir / f"model_seed{s}.txt")) for s in meta["seeds"]]
    return models, feat_cols, meta.get("target_transform", "log1p")


def prepare_xy(
    df: pl.DataFrame, feat_cols: list[str], target_transform: str = "log1p"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y_transformed, y_raw). y_raw is kept for RMSLE evaluation;
    y_transformed is what the model is actually fit on:
    - "log1p": log1p(clip(y_raw, 0)) - pairs with objective=regression.
    - "none": clip(y_raw, 0) - pairs with objective=tweedie, which models the
      zero-inflated right-skewed raw GMV distribution directly and expects
      to see it on its natural scale, not pre-compressed by log1p.
    """
    X = df.select(feat_cols).to_numpy()
    y_raw = df["target"].to_numpy().astype(np.float64)
    if target_transform == "sqrt":
        # Корень вместо log1p. log1p сжимает хвост сильнее, чем нужно метрике:
        # RMSLE — это RMSE в лог-пространстве, поэтому обучение на log1p совпадает
        # с метрикой, но и наследует её геометрию целиком. Корень сжимает слабее,
        # то есть модель иначе распределяет внимание между нулями и хвостом.
        # Предсказание возвращается в лог-пространство при инференсе.
        return X, np.sqrt(np.clip(y_raw, 0, None)).astype(np.float32), y_raw
    if target_transform == "log1p":
        y_transformed = np.log1p(np.clip(y_raw, 0, None)).astype(np.float32)
    elif target_transform == "none":
        y_transformed = np.clip(y_raw, 0, None).astype(np.float32)
    else:
        raise ValueError(f"unknown target_transform {target_transform!r}")
    return X, y_transformed, y_raw
