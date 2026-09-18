"""Anchor-date feature engineering for the LTV/GMV task.

For a given "anchor" date, every feature is computed strictly from events on
or before that anchor, and the target is the sum of `gmv` over the 30 days
strictly after it. Walking the anchor back through history (see
`generate_cv_anchor_dates`) turns a single wide user log into many
point-in-time train/validation snapshots without ever looking into a
snapshot's own future.
"""

import math
from datetime import date, timedelta
from typing import Optional

import polars as pl

EPS = 1.0


def _window_agg_exprs(anchor_val: date, value_cols: list[str], aggs: list[str], windows: list[dict]) -> list[pl.Expr]:
    exprs = []
    for w in windows:
        w_name, start_off, end_off = w["name"], w["start_offset"], w["end_offset"]
        w_start = anchor_val - timedelta(days=start_off)
        w_end = anchor_val - timedelta(days=end_off)
        mask = pl.col("event_date").is_between(w_start, w_end)
        exprs.append(pl.when(mask).then(1).otherwise(0).sum().alias(f"active_days_{w_name}"))
        for col in value_cols:
            for agg in aggs:
                if agg == "sum":
                    e = pl.when(mask).then(pl.col(col)).otherwise(0.0).sum()
                elif agg == "mean":
                    e = pl.when(mask).then(pl.col(col)).otherwise(None).mean()
                elif agg == "std":
                    e = pl.when(mask).then(pl.col(col)).otherwise(None).std()
                elif agg == "max":
                    e = pl.when(mask).then(pl.col(col)).otherwise(None).max()
                else:
                    raise ValueError(f"unknown agg {agg!r}")
                exprs.append(e.alias(f"{col}_{agg}_{w_name}"))
    return exprs


def _exp_decay_exprs(anchor_val: date, value_cols: list[str], halflives: list[int], window_days: int) -> list[pl.Expr]:
    """Smooth recency-weighted sums: weight = exp(-ln2/halflife * age_days),
    vs. the flat sum/mean windows above which weight every day in-window
    equally and only compare DISCRETE non-overlapping windows (the
    `*_momentum_*` features) to sense acceleration. This is untested -
    see experiments/exp_decay_recency/notes.md before trusting it."""
    w_start = anchor_val - timedelta(days=window_days - 1)
    mask = pl.col("event_date").is_between(w_start, anchor_val)
    age_days = (pl.lit(anchor_val) - pl.col("event_date")).dt.total_days().cast(pl.Float64)
    exprs = []
    for hl in halflives:
        decay = (-age_days * (math.log(2) / hl)).exp()
        for col in value_cols:
            e = pl.when(mask).then(pl.col(col) * decay).otherwise(0.0).sum()
            exprs.append(e.alias(f"{col}_expdecay_hl{hl}"))
    return exprs


def generate_exp_decay_features(
    data: pl.DataFrame,
    anchor_dates: list[date],
    value_cols: list[str],
    halflives: list[int],
    window_days: int = 90,
    user_ids: Optional[list[int]] = None,
) -> pl.DataFrame:
    """Standalone, opt-in companion to `generate_features` - computes only the
    exponential-decay recency features so an experiment can join them onto an
    already-built fold without recomputing the full feature set. Not wired
    into `build_feature_store`; see experiments/exp_decay_recency/notes.md."""
    if user_ids is None:
        user_ids = data["user_id"].unique().sort().to_list()

    max_anchor = max(anchor_dates)
    data_f = data.filter(pl.col("user_id").is_in(user_ids) & (pl.col("event_date") <= max_anchor))

    parts = []
    for a in anchor_dates:
        w_start = a - timedelta(days=window_days - 1)
        ad = data_f.filter(pl.col("event_date").is_between(w_start, a))
        if len(ad) > 0:
            exprs = _exp_decay_exprs(a, value_cols, halflives, window_days)
            feats = ad.group_by("user_id").agg(exprs).with_columns(anchor_date=pl.lit(a))
            parts.append(feats)

    features_df = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    index_df = pl.DataFrame({"anchor_date": anchor_dates}).join(pl.DataFrame({"user_id": user_ids}), how="cross")
    result = index_df.join(features_df, on=["anchor_date", "user_id"], how="left")
    fill_cols = [c for c in result.columns if c not in ("anchor_date", "user_id")]
    return result.with_columns([pl.col(c).fill_null(0.0) for c in fill_cols])


def _recency_tenure_exprs(anchor_val: date) -> list[pl.Expr]:
    return [
        (pl.lit(anchor_val) - pl.col("event_date").min()).dt.total_days().alias("tenure_days"),
        (pl.lit(anchor_val) - pl.col("event_date").max()).dt.total_days().alias("days_since_last_active"),
        (pl.lit(anchor_val) - pl.col("event_date").filter(pl.col("to_ord") > 0).max())
        .dt.total_days()
        .alias("days_since_last_order"),
        (pl.lit(anchor_val) - pl.col("event_date").filter(pl.col("gmv") > 0).max())
        .dt.total_days()
        .alias("days_since_last_gmv"),
        (pl.lit(anchor_val) - pl.col("event_date").filter(pl.col("searches") > 0).max())
        .dt.total_days()
        .alias("days_since_last_search"),
    ]


def _calendar_features(anchor_dates: list[date], data_min_date: date) -> pl.DataFrame:
    rows = []
    for a in anchor_dates:
        target_start = a + timedelta(days=1)
        target_mid = a + timedelta(days=15)
        rows.append(
            {
                "anchor_date": a,
                "anchor_month": a.month,
                "anchor_day_of_week": a.weekday(),
                "anchor_day_of_month": a.day,
                "target_start_month": target_start.month,
                "target_mid_doy": target_mid.timetuple().tm_yday,
                "elapsed_days_since_start": (a - data_min_date).days,
                "yoy_available": 1 if (target_start - timedelta(days=365)) >= data_min_date else 0,
            }
        )
    return pl.DataFrame(rows)


def compute_platform_daily(data: pl.DataFrame) -> pl.DataFrame:
    """Platform-wide (all-user) daily activity, used to normalize a user's
    intensity against the overall trend so tree models can extrapolate past
    the training anchors despite platform-wide growth over the period."""
    return (
        data.group_by("event_date")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("gmv").sum().alias("gmv_sum"),
            pl.col("searches").sum().alias("searches_sum"),
        )
        .sort("event_date")
    )


def _platform_trend_features(platform_daily: pl.DataFrame, anchor_dates: list[date]) -> pl.DataFrame:
    rows = []
    for a in anchor_dates:
        row = {"anchor_date": a}
        for w_name, start_off in [("30d", 29), ("90d", 89)]:
            w_start = a - timedelta(days=start_off)
            sub = platform_daily.filter(pl.col("event_date").is_between(w_start, a))
            n_rows = sub["n_rows"].sum() or 0
            gmv_sum = sub["gmv_sum"].sum() or 0.0
            searches_sum = sub["searches_sum"].sum() or 0.0
            row[f"platform_gmv_per_active_day_{w_name}"] = gmv_sum / (n_rows + EPS)
            row[f"platform_searches_per_active_day_{w_name}"] = searches_sum / (n_rows + EPS)
        rows.append(row)
    return pl.DataFrame(rows)


def generate_features(
    data: pl.DataFrame,
    anchor_dates: list[date],
    value_cols: list[str],
    aggs: list[str],
    windows: list[dict],
    user_ids: Optional[list[int]] = None,
    platform_daily: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    if user_ids is None:
        user_ids = data["user_id"].unique().sort().to_list()

    data_min_date = data["event_date"].min()
    max_anchor = max(anchor_dates)

    data_f = data.filter(pl.col("user_id").is_in(user_ids) & (pl.col("event_date") <= max_anchor))

    parts = []
    for a in anchor_dates:
        ad = data_f.filter(pl.col("event_date") <= a)
        if len(ad) > 0:
            exprs = _window_agg_exprs(a, value_cols, aggs, windows) + _recency_tenure_exprs(a)
            feats = ad.group_by("user_id").agg(exprs).with_columns(anchor_date=pl.lit(a))
            parts.append(feats)

    features_df = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()

    index_df = pl.DataFrame({"anchor_date": anchor_dates}).join(pl.DataFrame({"user_id": user_ids}), how="cross")
    result = index_df.join(features_df, on=["anchor_date", "user_id"], how="left")

    sum_active_cols = [c for c in result.columns if ("_sum_" in c or c.startswith("active_days_"))]
    result = result.with_columns([pl.col(c).fill_null(0.0) for c in sum_active_cols])

    for c in ["days_since_last_active", "days_since_last_order", "days_since_last_gmv", "days_since_last_search"]:
        result = result.with_columns(pl.col(c).fill_null(9999))
    result = result.with_columns(pl.col("tenure_days").fill_null(0))

    other_cols = [
        c
        for c in result.columns
        if c
        not in [
            "anchor_date",
            "user_id",
            "tenure_days",
            "days_since_last_active",
            "days_since_last_order",
            "days_since_last_gmv",
            "days_since_last_search",
        ]
        and c not in sum_active_cols
    ]
    result = result.with_columns([pl.col(c).fill_null(0.0) for c in other_cols])

    # derived ratio / rate features, keyed off the fixed 7/30/90/all window names
    derived = []
    for col in ["gmv", "searches", "to_ord"]:
        s7 = pl.col(f"{col}_sum_07d")
        s30 = pl.col(f"{col}_sum_30d")
        s90 = pl.col(f"{col}_sum_90d")
        sall = pl.col(f"{col}_sum_all")
        derived.append((s7 / (s30 + EPS)).alias(f"{col}_ratio_7_30"))
        derived.append((s30 / (s90 + EPS)).alias(f"{col}_ratio_30_90"))
        derived.append((s90 / (sall + EPS)).alias(f"{col}_ratio_90_all"))
    derived.append((pl.col("to_ord_sum_30d") / (pl.col("to_cart_sum_30d") + EPS)).alias("conv_cart_to_ord_30d"))
    derived.append((pl.col("to_cart_sum_30d") / (pl.col("searches_sum_30d") + EPS)).alias("conv_search_to_cart_30d"))
    derived.append((pl.col("gmv_sum_30d") / (pl.col("to_ord_sum_30d") + EPS)).alias("avg_order_value_30d"))
    derived.append((pl.col("gmv_sum_all") / (pl.col("to_ord_sum_all") + EPS)).alias("avg_order_value_all"))
    derived.append((pl.col("gmv_search_sum_30d") / (pl.col("gmv_sum_30d") + EPS)).alias("search_gmv_share_30d"))
    derived.append((pl.col("active_days_30d") / 30.0).alias("activity_rate_30d"))
    derived.append((pl.col("active_days_all") / (pl.col("tenure_days") + EPS)).alias("activity_rate_all"))
    # spend volatility: coefficient of variation (std/mean) of per-active-day
    # gmv, separating steady-spend users from bursty/one-off buyers with the
    # same mean spend.
    derived.append((pl.col("gmv_std_90d") / (pl.col("gmv_mean_90d") + EPS)).alias("gmv_cv_90d"))
    derived.append((pl.col("gmv_std_all") / (pl.col("gmv_mean_all") + EPS)).alias("gmv_cv_all"))
    # RFM-style composite: recent + frequent spenders score highest; decays
    # smoothly with days since last purchase rather than using a hard window cutoff.
    derived.append(
        (pl.col("active_days_90d") / (pl.col("days_since_last_gmv") + 1.0)).alias("rfm_score_90d")
    )
    # momentum: growth rate between adjacent, NON-overlapping periods (e.g.
    # "last 7 days" vs "the 7 days before that"). The existing *_ratio_7_30
    # etc. features compare cumulative windows of different lengths and
    # can't distinguish "steady" from "accelerating/decelerating" - these can.
    for col, (short, long_) in [("gmv", (7, 14)), ("gmv", (30, 60)), ("gmv", (90, 180)), ("searches", (7, 14)), ("to_ord", (30, 60))]:
        recent = pl.col(f"{col}_sum_{short:02d}d")
        prior = pl.col(f"{col}_sum_{long_}d") - recent
        derived.append(((recent - prior) / (prior + EPS)).alias(f"{col}_momentum_{short}_{long_}"))
    result = result.with_columns(derived)

    cal = _calendar_features(anchor_dates, data_min_date)
    result = result.join(cal, on="anchor_date", how="left")

    if platform_daily is not None:
        trend = _platform_trend_features(platform_daily, anchor_dates)
        result = result.join(trend, on="anchor_date", how="left")
        result = result.with_columns(
            [
                (pl.col("gmv_mean_30d") / (pl.col("platform_gmv_per_active_day_30d") + EPS)).alias(
                    "user_gmv_intensity_vs_platform_30d"
                ),
                (pl.col("gmv_mean_90d") / (pl.col("platform_gmv_per_active_day_90d") + EPS)).alias(
                    "user_gmv_intensity_vs_platform_90d"
                ),
                (pl.col("searches_mean_30d") / (pl.col("platform_searches_per_active_day_30d") + EPS)).alias(
                    "user_search_intensity_vs_platform_30d"
                ),
            ]
        )

    return result


def generate_targets(
    data: pl.DataFrame,
    anchor_dates: list[date],
    user_ids: Optional[list[int]] = None,
    horizon_days: int = 30,
    target_col: str = "gmv",
) -> pl.DataFrame:
    if user_ids is None:
        user_ids = data["user_id"].unique().sort().to_list()

    index_df = pl.DataFrame({"anchor_date": anchor_dates}).join(pl.DataFrame({"user_id": user_ids}), how="cross")

    parts = []
    for a in anchor_dates:
        t_start = a + timedelta(days=1)
        t_end = a + timedelta(days=horizon_days)
        tgt = (
            data.filter(pl.col("user_id").is_in(user_ids) & pl.col("event_date").is_between(t_start, t_end))
            .group_by("user_id")
            .agg(pl.col(target_col).sum().alias("target"))
            .with_columns(anchor_date=pl.lit(a))
        )
        parts.append(tgt)

    tgt_df = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    targets = index_df.join(tgt_df, on=["anchor_date", "user_id"], how="left")
    return targets.with_columns(pl.col("target").fill_null(0.0))


def generate_cv_anchor_dates(
    data: pl.DataFrame,
    prediction_horizon_days: int = 30,
    stride_days: int = 14,
    min_history_days: int = 120,
    n_folds: Optional[int] = None,
) -> list[date]:
    """Walk-forward anchors: each anchor must leave `horizon_days` of future
    data available (to compute its target) and `min_history_days` of past
    data (for the widest feature window to be non-degenerate)."""
    min_date = data["event_date"].min()
    max_date = data["event_date"].max()

    latest_anchor = max_date - timedelta(days=prediction_horizon_days)
    earliest_anchor = min_date + timedelta(days=min_history_days - 1)

    if latest_anchor < earliest_anchor:
        raise ValueError("Not enough history to place a single valid CV anchor")

    n_steps = (latest_anchor - earliest_anchor).days // stride_days
    all_anchors = sorted(latest_anchor - timedelta(days=i * stride_days) for i in range(n_steps + 1))

    if n_folds is not None:
        return all_anchors[-n_folds:]
    return all_anchors
