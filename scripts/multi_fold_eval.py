#!/usr/bin/env python
"""Robust multi-fold CV evaluation for a train config - use this instead of
trusting a single fold_05 number (see src/ecup_ltv/multi_fold_eval.py for
why). Does not train or save a production bundle; purely diagnostic.

Usage:
    python scripts/multi_fold_eval.py train_v3_top100_5seed
    python scripts/multi_fold_eval.py train_v3_top100_5seed --val-folds 2 3 4 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecup_ltv.config import load_config, resolve_path
from ecup_ltv.multi_fold_eval import evaluate_multi_fold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_config_name")
    parser.add_argument("--val-folds", nargs="+", type=int, default=[3, 4, 5])
    args = parser.parse_args()

    cfg = load_config(args.train_config_name)
    store_dir = resolve_path(cfg, "features_store_dir")
    result = evaluate_multi_fold(cfg, store_dir, args.val_folds)

    print(json.dumps(result, indent=2))
    print(f"\nRMSLE: {result['rmsle_mean']:.4f} +/- {result['rmsle_std']:.4f}  "
          f"(min={result['rmsle_min']:.4f} max={result['rmsle_max']:.4f})")
