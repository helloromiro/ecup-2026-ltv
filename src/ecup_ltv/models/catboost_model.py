"""CatBoost training + a bundle format mirroring models/lgbm.py.

Kept as its own module (not forced into the LightGBM-specific train.py)
because CatBoost's training API (Pool, symmetric/oblivious trees, ordered
boosting) is different enough that sharing train.py's fit loop would mean
either a leaky abstraction or scattering CatBoost-specific branches through
otherwise-simple LightGBM code. The bundle format (model files + feature
list + config snapshot + metrics) and the config-driven/seeded/CV-then-
production-fit shape are intentionally the same as lgbm.py, so the rest of
the repo's conventions (blend.py, multi_fold_eval.py) still apply.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost as cb
import numpy as np
import polars as pl
import yaml


def fit_one_seed(
    X: np.ndarray, y_log: np.ndarray, feat_cols: list[str], params: dict[str, Any], num_boost_round: int, seed: int
) -> cb.CatBoostRegressor:
    p = dict(params)
    p["random_seed"] = seed
    p["iterations"] = num_boost_round
    model = cb.CatBoostRegressor(**p)
    model.fit(X, y_log, verbose=False)
    return model


def predict_log_bagged(models: list[cb.CatBoostRegressor], X: np.ndarray) -> np.ndarray:
    preds = [m.predict(X) for m in models]
    return np.mean(preds, axis=0)


def save_bundle(
    out_dir: Path,
    models: list[cb.CatBoostRegressor],
    seeds: list[int],
    feat_cols: list[str],
    resolved_config: dict[str, Any],
    cv_metrics: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed, model in zip(seeds, models):
        model.save_model(str(out_dir / f"model_seed{seed}.cbm"))
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(resolved_config, f, allow_unicode=True, sort_keys=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(cv_metrics, f, indent=2)
    meta = {
        "seeds": seeds,
        "target_transform": "log1p",
        "model_type": "catboost",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_bundle(bundle_dir: Path) -> tuple[list[cb.CatBoostRegressor], list[str]]:
    with open(bundle_dir / "meta.json") as f:
        meta = json.load(f)
    with open(bundle_dir / "feature_cols.json") as f:
        feat_cols = json.load(f)
    models = []
    for s in meta["seeds"]:
        m = cb.CatBoostRegressor()
        m.load_model(str(bundle_dir / f"model_seed{s}.cbm"))
        models.append(m)
    return models, feat_cols


def prepare_xy(df: pl.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df.select(feat_cols).to_numpy()
    y_raw = df["target"].to_numpy().astype(np.float64)
    y_log = np.log1p(np.clip(y_raw, 0, None)).astype(np.float32)
    return X, y_log, y_raw
