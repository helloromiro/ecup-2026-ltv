#!/usr/bin/env python
"""Build the anchor-date feature store (CV folds + final prediction anchor)
from data/raw/train.parquet, cached to artifacts/features_store/.

Expensive (walks ~30M raw rows for 7 anchors); re-run only when
configs/features.yaml or configs/cv.yaml change.

Usage:
    python scripts/build_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecup_ltv.build_features import build_feature_store
from ecup_ltv.config import load_shared_config, resolve_path

if __name__ == "__main__":
    cfg = load_shared_config()
    build_feature_store(cfg, resolve_path(cfg, "features_store_dir"))
