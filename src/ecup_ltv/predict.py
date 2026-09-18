"""Inference entrypoint: a saved model bundle + the cached `fold_end` features
in, a submission.csv out. Never trains anything - if a bundle is missing,
that is a signal to go run train.py, not to fall back to retraining here.
"""

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from ecup_ltv.build_features import load_fold
from ecup_ltv.config import load_config, resolve_path
from ecup_ltv.models.lgbm import invert_prediction, load_bundle, predict_transformed_bagged


def predict(train_config_name: str) -> Path:
    t0 = time.time()
    cfg = load_config(train_config_name)
    exp_name = cfg["experiment"]["name"]

    store_dir = resolve_path(cfg, "features_store_dir")
    bundle_dir = resolve_path(cfg, "models_dir") / exp_name
    if not (bundle_dir / "meta.json").exists():
        raise FileNotFoundError(f"No trained model bundle at {bundle_dir}. Run `python -m ecup_ltv.train {train_config_name}` first.")

    models, feat_cols, target_transform = load_bundle(bundle_dir)
    fold_end = load_fold(store_dir, "fold_end")
    X = fold_end.select(feat_cols).to_numpy()
    print(f"[{time.time()-t0:.1f}s] loaded {len(models)}-seed bundle ({target_transform=}) and fold_end features {fold_end.shape}")

    pred_transformed = predict_transformed_bagged(models, X)
    pred = invert_prediction(pred_transformed, target_transform)

    submit = pl.DataFrame({"user_id": fold_end["user_id"], "predict": pred}).sort("user_id")

    sample_sub = pl.read_csv(resolve_path(cfg, "sample_submission_path"))
    assert submit.columns == sample_sub.columns, f"column mismatch: {submit.columns} vs {sample_sub.columns}"
    assert set(submit["user_id"].to_list()) == set(sample_sub["user_id"].to_list()), "user_id set mismatch vs sample_submit.csv"

    out_dir = resolve_path(cfg, "submissions_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{exp_name}.csv"
    submit.write_csv(out_path)

    s = submit["predict"]
    print(
        "stats: mean=%.3f median=%.3f std=%.3f max=%.2f n_neg=%d n_null=%d"
        % (s.mean(), s.median(), s.std(), s.max(), (s < 0).sum(), s.null_count())
    )
    print(f"[{time.time()-t0:.1f}s] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m ecup_ltv.predict <train_config_name>  (e.g. train_v1_baseline)")
        sys.exit(1)
    predict(sys.argv[1])
