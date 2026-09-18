"""Precompute the feature store: one set of anchor-date features per CV fold
plus a final `fold_end` anchor (the last available date, used for the actual
submission). This is the expensive step (~30M raw rows -> per-user window
aggregates for 7 anchors), so it is cached to parquet under
artifacts/features_store/ and only re-run when configs/features.yaml or
configs/cv.yaml change - train.py and predict.py both read from this cache.
"""

import json
import time
from pathlib import Path
from typing import Any

import polars as pl

from ecup_ltv.btyd import compute_btyd_summary, fit_and_score_btyd
from ecup_ltv.data.loading import load_raw_events, load_user_ids
from ecup_ltv.features import compute_platform_daily, generate_cv_anchor_dates, generate_features, generate_targets
from ecup_ltv.holiday_affinity import compute_holiday_affinity


def build_feature_store(
    cfg: dict[str, Any],
    out_dir: Path,
    batch_size: int = 50_000,
    include_btyd: bool = True,
    include_holiday_affinity: bool = True,
    only_folds: set[int] | None = None,
    include_final: bool = True,
    anchors_override: list | None = None,
) -> None:
    """`only_folds` / `include_final` let a caller build just part of a store -
    used by scripts/build_features_extended.py, which reuses the six folds the
    default store already has instead of recomputing them."""
    t0 = time.time()
    raw_path = Path(cfg["raw_path"])
    if not raw_path.is_absolute():
        from ecup_ltv.config import REPO_ROOT

        raw_path = REPO_ROOT / raw_path
    data = load_raw_events(raw_path)
    print(f"[{time.time()-t0:.1f}s] loaded raw events {data.shape}")

    from ecup_ltv.config import REPO_ROOT

    sample_sub_path = REPO_ROOT / cfg["sample_submission_path"]
    users = load_user_ids(sample_sub_path)
    print(f"n_users={len(users)}")

    platform_daily = compute_platform_daily(data)

    # `anchors_override` lets a caller lay the anchors out to match a specific
    # validation protocol rather than the fixed stride - deployment-matched CV
    # needs the last training anchor exactly `horizon` days before each
    # validation anchor, which a 14-day stride can never place when the horizon
    # is 30.
    anchors_cv = anchors_override or generate_cv_anchor_dates(
        data,
        prediction_horizon_days=cfg["horizon_days"],
        stride_days=cfg["stride_days"],
        min_history_days=cfg["min_history_days"],
        n_folds=cfg["n_folds"],
    )
    anchor_final = data["event_date"].max()
    print("CV anchors:", anchors_cv)
    print("Final anchor:", anchor_final)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "cv_anchors": [str(a) for a in anchors_cv],
        "final_anchor": str(anchor_final),
        "batch_size": batch_size,
        "n_users": len(users),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    n_batches = (len(users) + batch_size - 1) // batch_size
    value_cols, aggs, windows = cfg["value_cols"], cfg["aggs"], cfg["windows"]

    # not anchor-dependent (fixed 2025 calendar dates), computed once for the
    # whole population and joined into every fold/fold_end batch below.
    holiday_affinity = compute_holiday_affinity(data, users) if include_holiday_affinity else None

    def btyd_for_anchor(anchor) -> pl.DataFrame | None:
        # BG/NBD + Gamma-Gamma fit population-level parameters, so this must
        # run once over ALL users for this anchor (not per-batch like the
        # window features above, which are independent per user).
        if not include_btyd:
            return None
        summary = compute_btyd_summary(data, anchor, users)
        return fit_and_score_btyd(summary, horizon_days=cfg["horizon_days"])

    for fold_idx, anchor in enumerate(anchors_cv):
        if only_folds is not None and fold_idx not in only_folds:
            continue
        fold_dir = out_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        tf0 = time.time()
        btyd_scored = btyd_for_anchor(anchor)
        for b in range(n_batches):
            batch_users = users[b * batch_size : (b + 1) * batch_size]
            feats = generate_features(
                data, [anchor], value_cols, aggs, windows, user_ids=batch_users, platform_daily=platform_daily
            )
            tgt = generate_targets(data, [anchor], user_ids=batch_users, horizon_days=cfg["horizon_days"])
            out = feats.join(tgt, on=["anchor_date", "user_id"], how="left")
            if btyd_scored is not None:
                out = out.join(btyd_scored, on="user_id", how="left")
            if holiday_affinity is not None:
                out = out.join(holiday_affinity, on="user_id", how="left")
            out.write_parquet(fold_dir / f"batch_{b:04d}.parquet")
        print(f"[{time.time()-t0:.1f}s] fold_{fold_idx:02d} anchor={anchor} done in {time.time()-tf0:.1f}s")

    if not include_final:
        print(f"TOTAL {time.time()-t0:.1f}s (fold_end skipped)")
        return

    fold_dir = out_dir / "fold_end"
    fold_dir.mkdir(parents=True, exist_ok=True)
    tf0 = time.time()
    btyd_scored = btyd_for_anchor(anchor_final)
    for b in range(n_batches):
        batch_users = users[b * batch_size : (b + 1) * batch_size]
        feats = generate_features(
            data, [anchor_final], value_cols, aggs, windows, user_ids=batch_users, platform_daily=platform_daily
        )
        if btyd_scored is not None:
            feats = feats.join(btyd_scored, on="user_id", how="left")
        if holiday_affinity is not None:
            feats = feats.join(holiday_affinity, on="user_id", how="left")
        feats.write_parquet(fold_dir / f"batch_{b:04d}.parquet")
    print(f"[{time.time()-t0:.1f}s] fold_end anchor={anchor_final} done in {time.time()-tf0:.1f}s")
    print(f"TOTAL {time.time()-t0:.1f}s")


NON_FEATURE_COLS = {"anchor_date", "user_id", "target"}


def load_fold(store_dir: Path, fold_name: str) -> pl.DataFrame:
    df = pl.read_parquet(str(store_dir / fold_name / "batch_*.parquet"))
    cast_exprs = [
        pl.col(c).cast(pl.Float32)
        for c, t in zip(df.columns, df.dtypes)
        if c not in ("anchor_date", "user_id") and t in (pl.Float64, pl.Int64, pl.Int32, pl.UInt32)
    ]
    return df.with_columns(cast_exprs) if cast_exprs else df


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


if __name__ == "__main__":
    from ecup_ltv.config import load_shared_config, resolve_path

    cfg = load_shared_config()
    build_feature_store(cfg, resolve_path(cfg, "features_store_dir"))
