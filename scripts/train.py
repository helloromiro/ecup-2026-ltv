#!/usr/bin/env python
"""Train a model bundle from a configs/train_*.yaml experiment config.
Requires the feature store to already exist (see scripts/build_features.py).

Usage:
    python scripts/train.py train_v1_baseline
    python scripts/train.py train_v2_seed_bagging
    python scripts/train.py train_v3_top100_5seed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecup_ltv.train import train

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    train(sys.argv[1])
