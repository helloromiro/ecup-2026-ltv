"""Feature families the window aggregates structurally cannot express.

`configs/features.yaml` builds 442 features as value_cols x {sum, mean, std,
max} x windows. Four aggregations over fixed windows describe *how much* a
user spent, but say almost nothing about the *shape* of their purchase
history:

* **inter-purchase gaps.** "Bought 6 times in 180 days" is the same feature
  value whether the user buys like clockwork every 30 days or bought six
  times in one week and vanished. BTYD's recency/frequency summarises the
  same two numbers the windows already carry; the *dispersion* of the gaps
  is new, and for a 30-day-ahead target it is close to the whole question.
* **quantiles of the daily amount.** With sum/mean/std/max only, a user with
  one 50k order and thirty 100-rouble days is hard to tell from a steady
  spender with the same mean. The median and p75/p90 of active-day spend
  separate them directly, and the target is log-scaled, so the typical day
  matters more than the total.
* **the last few purchases.** The most recent 1-3 order values, and how long
  ago each was, are the plain sequence tail. `gmv_sum_07d`/`14d` blur them
  into windows that most users' purchases do not line up with.
* **weekday concentration.** Share of spend on weekends - cheap, and the
  target window is dominated by two dated gifting peaks.

All of it is computed strictly from events on or before the anchor, same as
everything else in features.py.
"""

from datetime import date, timedelta

import polars as pl

EPS = 1.0
GAP_WINDOWS = [("180d", 179), ("all", 2000)]
QUANTILE_WINDOWS = [("30d", 29), ("90d", 89), ("all", 2000)]
N_LAST = 3


def _gap_exprs(anchor: date) -> list[pl.Expr]:
    """Statistics of the spacing between consecutive purchase days."""
    exprs: list[pl.Expr] = []
    for name, off in GAP_WINDOWS:
        start = anchor - timedelta(days=off)
        buy = pl.col("event_date").filter((pl.col("gmv") > 0) & (pl.col("event_date") >= start)).sort()
        gaps = buy.diff().dt.total_days()
        exprs += [
            gaps.mean().alias(f"gap_mean_{name}"),
            gaps.std().alias(f"gap_std_{name}"),
            gaps.max().alias(f"gap_max_{name}"),
            gaps.min().alias(f"gap_min_{name}"),
            # regularity: a clockwork buyer has CV near 0, a bursty one >> 1
            (gaps.std() / (gaps.mean() + EPS)).alias(f"gap_cv_{name}"),
            buy.len().alias(f"n_purchase_days_{name}"),
            # how the latest silence compares to this user's usual spacing -
            # "overdue" is a per-user notion, not the global days_since_last_gmv
            ((pl.lit(anchor) - buy.max()).dt.total_days() / (gaps.mean() + EPS)).alias(f"overdue_ratio_{name}"),
        ]
    return exprs


def _quantile_exprs(anchor: date) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for name, off in QUANTILE_WINDOWS:
        start = anchor - timedelta(days=off)
        daily = pl.col("gmv").filter((pl.col("gmv") > 0) & (pl.col("event_date") >= start))
        for q in (0.25, 0.5, 0.75, 0.9):
            exprs.append(daily.quantile(q).alias(f"gmv_q{int(q * 100):02d}_{name}"))
        # how top-heavy the spend is: max over median, and the share the single
        # biggest day takes of the window total
        exprs.append((daily.max() / (daily.quantile(0.5) + EPS)).alias(f"gmv_max_over_med_{name}"))
        exprs.append((daily.max() / (daily.sum() + EPS)).alias(f"gmv_top_day_share_{name}"))
    return exprs


def _last_purchase_exprs(anchor: date) -> list[pl.Expr]:
    """The plain sequence tail: the last N purchase amounts and their dates."""
    exprs: list[pl.Expr] = []
    amt = pl.col("gmv").filter(pl.col("gmv") > 0)
    day = pl.col("event_date").filter(pl.col("gmv") > 0)
    order = pl.col("event_date").filter(pl.col("gmv") > 0).arg_sort(descending=True)
    for k in range(N_LAST):
        exprs.append(amt.gather(order.get(k, null_on_oob=True)).first().alias(f"last{k + 1}_gmv"))
        exprs.append(
            (pl.lit(anchor) - day.gather(order.get(k, null_on_oob=True)).first())
            .dt.total_days()
            .alias(f"last{k + 1}_days_ago")
        )
    return exprs


def _weekend_exprs(anchor: date) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for name, off in [("90d", 89), ("all", 2000)]:
        start = anchor - timedelta(days=off)
        inw = pl.col("event_date") >= start
        wknd = inw & (pl.col("event_date").dt.weekday() >= 6)
        tot = pl.when(inw).then(pl.col("gmv")).otherwise(0.0).sum()
        exprs.append((pl.when(wknd).then(pl.col("gmv")).otherwise(0.0).sum() / (tot + EPS)).alias(f"weekend_gmv_share_{name}"))
    return exprs


def compute_gap_features(data: pl.DataFrame, anchor: date, user_ids: list[int]) -> pl.DataFrame:
    """One row per user in `user_ids`, all columns derived from events <= anchor."""
    ad = data.filter(pl.col("user_id").is_in(user_ids) & (pl.col("event_date") <= anchor))
    exprs = _gap_exprs(anchor) + _quantile_exprs(anchor) + _last_purchase_exprs(anchor) + _weekend_exprs(anchor)
    feats = ad.group_by("user_id").agg(exprs)

    out = pl.DataFrame({"user_id": user_ids}).join(feats, on="user_id", how="left")
    # A user with fewer than two purchases has no gap at all. Filling those
    # with 0 would put them next to the most frequent buyers; 9999 keeps
    # "never / not enough history" on the far side of every real value, the
    # same sentinel features.py already uses for days_since_last_*.
    def _fill(col: str) -> float:
        if col.startswith(("gap_", "overdue_")) or col.endswith("_days_ago"):
            return 9999.0
        return 0.0

    return out.with_columns([pl.col(c).fill_null(_fill(c)) for c in out.columns if c != "user_id"])
