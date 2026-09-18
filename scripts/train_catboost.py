#!/usr/bin/env python
"""Train a CatBoost model bundle from a configs/train_*.yaml config with a
`catboost_params` section. See src/ecup_ltv/train_catboost.py.

Usage:
    python scripts/train_catboost.py train_catboost_diverse
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecup_ltv.train_catboost import train

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    train(sys.argv[1])
