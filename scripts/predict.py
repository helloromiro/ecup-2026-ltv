#!/usr/bin/env python
"""Run inference from an already-trained model bundle (see scripts/train.py)
and write a submission CSV to artifacts/submissions/.

Usage:
    python scripts/predict.py train_v3_top100_5seed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecup_ltv.predict import predict

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    predict(sys.argv[1])
