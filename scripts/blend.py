#!/usr/bin/env python
"""Evaluate and/or write a blend of already-trained model bundles.

Blending averages predictions in log1p-space (see src/ecup_ltv/blend.py).
CV evaluation uses each bundle's cv_only/ sub-bundle (fold_05 held out);
the final submission uses the full production bundles.

Usage:
    python scripts/blend.py v2_seed_bagging v3_top100_5seed
    python scripts/blend.py v2_seed_bagging v3_top100_5seed --weights 1 2 --submit
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from ecup_ltv.blend import blend_predict, evaluate_blend_on_val
from ecup_ltv.build_features import load_fold
from ecup_ltv.config import load_shared_config, resolve_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_names", nargs="+")
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--submit", action="store_true", help="also write a submission CSV from the blend")
    parser.add_argument("--out-name", default=None, help="submission filename stem (default: joined bundle names)")
    args = parser.parse_args()

    cfg = load_shared_config()
    store_dir = resolve_path(cfg, "features_store_dir")
    models_dir = resolve_path(cfg, "models_dir")

    result = evaluate_blend_on_val(cfg, store_dir, models_dir, args.bundle_names, args.weights)
    print("CV blend result:", result)

    if args.submit:
        fold_end = load_fold(store_dir, "fold_end")
        pred = blend_predict(models_dir, args.bundle_names, fold_end, args.weights)
        submit = pl.DataFrame({"user_id": fold_end["user_id"], "predict": pred}).sort("user_id")

        sample_sub = pl.read_csv(resolve_path(cfg, "sample_submission_path"))
        assert submit.columns == sample_sub.columns
        assert set(submit["user_id"].to_list()) == set(sample_sub["user_id"].to_list())

        out_name = args.out_name or "blend_" + "_".join(args.bundle_names)
        out_dir = resolve_path(cfg, "submissions_dir")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{out_name}.csv"
        submit.write_csv(out_path)
        print(f"wrote {out_path}")
