"""Two signals the audit found missing, built as one side-feature set.

**Channel counts.** `configs/features.yaml` aggregates 12 of the 16 raw
signals. `search_to_cart`, `search_to_ord`, `cat_to_cart` and `cat_to_ord`
are loaded and then never used, so FINDINGS.md's claim that all 16 are in
play was wrong. The totals are recoverable - `to_cart = search_to_cart +
cat_to_cart` and likewise for orders - and the `has_*` flags say whether each
channel fired at all, but neither recovers HOW the conversions split between
Search and Catalog. Over the 90 days before the final anchor both channels
appear for 40.6% of users on cart and 17.4% on orders, so the split is real
for a large minority.

**Zero-only rows.** 4,549,734 rows (14.9% of the log) have every one of the
16 signals at zero, and today they count as activity: `active_days_*` counts
rows, `days_since_last_active` takes the max event_date without looking at
the signals, and wave-two's activity gaps were built over all rows. So a
user last seen 8 days ago with nothing recorded looks identical to one who
actually did something. That is a plausible reason wave two measured exactly
nothing despite a sensible construction - it was timing gaps between log
rows, not between actions.

Deleting those rows would be wrong: a row's existence may itself encode an
observation, and 384 users have nothing else. So both semantics are provided
side by side and the model can pick.
"""

from datetime import date, timedelta

import polars as pl

EPS = 1.0
CHANNEL_WINDOWS = [("30d", 29), ("90d", 89), ("180d", 179), ("all", 2000)]
ZERO_WINDOWS = [("30d", 29), ("90d", 89), ("all", 2000)]
CHANNEL_COLS = ["search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord"]


def _channel_exprs(anchor: date) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for name, off in CHANNEL_WINDOWS:
        start = anchor - timedelta(days=off)
        inw = pl.col("event_date") >= start
        sums = {c: pl.when(inw).then(pl.col(c)).otherwise(0.0).sum() for c in CHANNEL_COLS}
        for c, e in sums.items():
            exprs.append(e.alias(f"{c}_sum_{name}"))
        # the split itself, which the totals cannot express
        exprs += [
            (sums["search_to_ord"] / (sums["search_to_ord"] + sums["cat_to_ord"] + EPS)).alias(
                f"search_share_ord_{name}"
            ),
            (sums["search_to_cart"] / (sums["search_to_cart"] + sums["cat_to_cart"] + EPS)).alias(
                f"search_share_cart_{name}"
            ),
            # per-channel funnel: does Search convert better than Catalog for
            # this user? a different question from the overall conversion rate
            (sums["search_to_ord"] / (sums["search_to_cart"] + EPS)).alias(f"search_cart2ord_{name}"),
            (sums["cat_to_ord"] / (sums["cat_to_cart"] + EPS)).alias(f"cat_cart2ord_{name}"),
        ]
    # is the channel mix drifting? a user moving from browsing catalog to
    # searching is behaving differently even at constant volume
    exprs.append(
        (pl.when(pl.col("event_date") >= anchor - timedelta(days=29))
         .then(pl.col("search_to_ord")).otherwise(0.0).sum()
         / (pl.when(pl.col("event_date") >= anchor - timedelta(days=29))
            .then(pl.col("to_ord")).otherwise(0.0).sum() + EPS)
         - pl.when(pl.col("event_date") >= anchor - timedelta(days=179))
         .then(pl.col("search_to_ord")).otherwise(0.0).sum()
         / (pl.when(pl.col("event_date") >= anchor - timedelta(days=179))
            .then(pl.col("to_ord")).otherwise(0.0).sum() + EPS)
         ).alias("search_share_drift_30_180")
    )
    return exprs


def _zero_row_exprs(anchor: date, signal_cols: list[str]) -> list[pl.Expr]:
    nonzero = pl.any_horizontal([pl.col(c) != 0 for c in signal_cols])
    exprs: list[pl.Expr] = []
    for name, off in ZERO_WINDOWS:
        start = anchor - timedelta(days=off)
        inw = pl.col("event_date") >= start
        rows = pl.when(inw).then(1).otherwise(0).sum()
        real = pl.when(inw & nonzero).then(1).otherwise(0).sum()
        exprs += [
            real.alias(f"nonzero_signal_days_{name}"),
            (rows - real).alias(f"zero_only_days_{name}"),
            ((rows - real) / (rows + EPS)).alias(f"zero_only_share_{name}"),
        ]
    exprs += [
        # the recency that features.py currently gets wrong: it takes the last
        # row, which may carry no signal at all
        (pl.lit(anchor) - pl.col("event_date").filter(nonzero).max()).dt.total_days().alias(
            "days_since_last_nonzero"
        ),
        # how long the user has been showing up with nothing to show
        (pl.col("event_date").max() - pl.col("event_date").filter(nonzero).max()).dt.total_days().alias(
            "zero_only_streak_days"
        ),
        # gaps between real actions, not between log rows - wave two measured
        # the latter and found nothing
        pl.col("event_date").filter(nonzero).sort().diff().dt.total_days().mean().alias("nonzero_gap_mean"),
        pl.col("event_date").filter(nonzero).sort().diff().dt.total_days().std().alias("nonzero_gap_std"),
    ]
    return exprs


def compute_channel_zero_features(data: pl.DataFrame, anchor: date, user_ids: list[int]) -> pl.DataFrame:
    signal_cols = [c for c in data.columns if c not in ("event_date", "user_id")]
    ad = data.filter(pl.col("user_id").is_in(user_ids) & (pl.col("event_date") <= anchor))
    feats = ad.group_by("user_id").agg(_channel_exprs(anchor) + _zero_row_exprs(anchor, signal_cols))
    out = pl.DataFrame({"user_id": user_ids}).join(feats, on="user_id", how="left")

    def _fill(col: str) -> float:
        # "no such day ever" sits beyond every real value, as elsewhere
        return 9999.0 if col.startswith("days_since") or "gap" in col else 0.0

    return out.with_columns(
        [pl.col(c).fill_null(_fill(c)).fill_nan(_fill(c)) for c in out.columns if c != "user_id"]
    )
