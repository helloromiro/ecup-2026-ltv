"""Second wave of history-shape features.

Wave one (`gap_features.py`) added inter-purchase gaps, daily-amount
quantiles and the last three purchases, all for `gmv`, and was worth -0.0025
on honest multi-fold and -0.00082 on the leaderboard. It left obvious room:

* **activity gaps, not just purchase gaps.** 46% of users have no purchase in
  the target window and many have almost none in history, so every wave-one
  gap feature is the 9999 sentinel for them - a large slice of the population
  gets no shape signal at all. Gaps between days with *any* event separate a
  browsing regular from someone who showed up twice.
* **gap trend.** `overdue_ratio` says how late the user is right now;
  it cannot say whether their rhythm is decaying. The last three gaps against
  the lifetime mean is a direct churn signal and is what a 30-day-ahead
  target turns on.
* **shape of the order count**, not only of the money: the same families for
  `to_ord`.
* **lifecycle position.** Spend in the user's first 30 days against their
  recent 30 - a maturing new user and a decaying old one can have identical
  window aggregates.
* **OLS slope** over the daily series. `*_momentum_*` compares two adjacent
  buckets, which is a two-point slope with all the noise that implies.

Deliberately NOT included: day-of-month concentration. The intuition was
payday cycles, but the target window is exactly 30 days, so it contains every
day of the month once and exactly one cycle - knowing a user buys on the 20th
does not change whether they fall inside the window. No mechanism.
"""

from datetime import date, timedelta

import polars as pl

EPS = 1.0


def _activity_gap_exprs(anchor: date) -> list[pl.Expr]:
    """Same shape statistics as wave one, but over days with ANY event - the
    only signal available for users who rarely or never purchase."""
    exprs: list[pl.Expr] = []
    for name, off in [("90d", 89), ("all", 2000)]:
        start = anchor - timedelta(days=off)
        days = pl.col("event_date").filter(pl.col("event_date") >= start).unique().sort()
        gaps = days.diff().dt.total_days()
        exprs += [
            gaps.mean().alias(f"act_gap_mean_{name}"),
            gaps.std().alias(f"act_gap_std_{name}"),
            gaps.max().alias(f"act_gap_max_{name}"),
            (gaps.std() / (gaps.mean() + EPS)).alias(f"act_gap_cv_{name}"),
            ((pl.lit(anchor) - days.max()).dt.total_days() / (gaps.mean() + EPS)).alias(f"act_overdue_ratio_{name}"),
        ]
    return exprs


def _gap_trend_exprs(anchor: date) -> list[pl.Expr]:
    """Are the gaps stretching out (decaying) or tightening (accelerating)?

    Only for gmv: `to_ord > 0` and `gmv > 0` mark exactly the same days in
    this dataset (verified over all 30.6M rows), so an order-gap version
    would be a bit-identical duplicate."""
    days = pl.col("event_date").filter(pl.col("gmv") > 0).sort()
    gaps = days.diff().dt.total_days()
    recent = gaps.tail(3).mean()
    return [
        recent.alias("buy_gap_recent3"),
        (recent / (gaps.mean() + EPS)).alias("buy_gap_trend"),
    ]


def _to_ord_shape_exprs(anchor: date) -> list[pl.Expr]:
    """Wave-one families applied to the order count instead of the money."""
    exprs: list[pl.Expr] = []
    for name, off in [("90d", 89), ("all", 2000)]:
        start = anchor - timedelta(days=off)
        daily = pl.col("to_ord").filter((pl.col("to_ord") > 0) & (pl.col("event_date") >= start))
        exprs += [
            daily.quantile(0.5).alias(f"to_ord_q50_{name}"),
            daily.quantile(0.9).alias(f"to_ord_q90_{name}"),
            (daily.max() / (daily.quantile(0.5) + EPS)).alias(f"to_ord_max_over_med_{name}"),
            # n_order_days_all would duplicate wave one's n_purchase_days_all
            *([daily.len().alias(f"n_order_days_{name}")] if name != "all" else []),
        ]
    return exprs


def _lifecycle_exprs(anchor: date) -> list[pl.Expr]:
    """Where the user is in their own life, not in calendar time."""
    first = pl.col("event_date").min()
    early = pl.col("gmv").filter(pl.col("event_date") <= first + pl.duration(days=29)).sum()
    recent = pl.col("gmv").filter(pl.col("event_date") >= pl.lit(anchor) - pl.duration(days=29)).sum()
    return [
        early.alias("gmv_first30d"),
        (recent / (early + EPS)).alias("gmv_recent_over_first30d"),
        # events per day of tenure: a dense short life vs a sparse long one
        (pl.len() / ((pl.lit(anchor) - first).dt.total_days() + EPS)).alias("events_per_tenure_day"),
    ]


def _slope_exprs(anchor: date) -> list[pl.Expr]:
    """OLS slope of daily gmv against day index - the momentum features are a
    two-bucket approximation of this and carry the noise of both buckets."""
    exprs: list[pl.Expr] = []
    for name, off in [("30d", 29), ("90d", 89), ("180d", 179)]:
        start = anchor - timedelta(days=off)
        inw = pl.col("event_date") >= start
        x = pl.when(inw).then((pl.col("event_date") - pl.lit(start)).dt.total_days()).otherwise(None)
        y = pl.when(inw).then(pl.col("gmv")).otherwise(None)
        cov = (x * y).mean() - x.mean() * y.mean()
        var = (x * x).mean() - x.mean() * x.mean()
        exprs.append((cov / (var + EPS)).alias(f"gmv_slope_{name}"))
    return exprs


def compute_gap_features2(data: pl.DataFrame, anchor: date, user_ids: list[int]) -> pl.DataFrame:
    ad = data.filter(pl.col("user_id").is_in(user_ids) & (pl.col("event_date") <= anchor))
    exprs = (
        _activity_gap_exprs(anchor)
        + _gap_trend_exprs(anchor)
        + _to_ord_shape_exprs(anchor)
        + _lifecycle_exprs(anchor)
        + _slope_exprs(anchor)
    )
    feats = ad.group_by("user_id").agg(exprs)
    out = pl.DataFrame({"user_id": user_ids}).join(feats, on="user_id", how="left")

    def _fill(col: str) -> float:
        # same convention as wave one: "no history to measure" sits beyond
        # every real value rather than colliding with a genuine zero gap
        return 9999.0 if "gap" in col or "overdue" in col else 0.0

    return out.with_columns([pl.col(c).fill_null(_fill(c)) for c in out.columns if c != "user_id"])
