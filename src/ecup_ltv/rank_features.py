"""Percentile ranks: the level-invariant view of the same features.

The largest correction in this project is a constant shift, because the
platform's level in the test window differs from every training fold and a
tree cannot extrapolate a calendar value it has never seen. That is handled
after the fact, on the prediction. This handles it before the fact, on the
inputs.

`gmv_sum_30d = 500` means something different in November and in February.
"this user is at the 87th percentile of the population this month" means the
same thing in both. A rank is invariant to any monotone change in the level,
not only to scaling - which is more than the existing
`user_gmv_intensity_vs_platform_30d` gets by dividing through the platform
average.

Ranks are computed within each anchor separately, which is the whole point:
they are relative to the population as it was at that moment.

The obvious risk: if the model's remaining errors are about absolute amounts
rather than about position, ranks throw away exactly the information that
matters and add nothing. That is what the leaderboard is for.
"""

from datetime import date

import polars as pl

# the families where position in the population is a natural way to read the
# number; deliberately not everything, since a rank of a ratio is rarely
# more meaningful than the ratio
RANK_COLS = [
    "gmv_sum_07d", "gmv_sum_30d", "gmv_sum_90d", "gmv_sum_180d", "gmv_sum_all",
    "to_ord_sum_30d", "to_ord_sum_90d", "to_ord_sum_all",
    "searches_sum_30d", "searches_sum_90d",
    "active_days_30d", "active_days_90d", "active_days_all",
    "days_since_last_gmv", "days_since_last_order", "tenure_days",
    "gmv_mean_90d", "gmv_max_90d", "gmv_std_90d",
]


def compute_rank_features(fold_df: pl.DataFrame, anchor: date) -> pl.DataFrame:
    """One row per user: the percentile of each listed column within this
    anchor's population, plus a couple of within-user comparisons that only
    make sense once everything is on the same 0-1 scale."""
    present = [c for c in RANK_COLS if c in fold_df.columns]
    out = fold_df.select(
        "user_id",
        *[(pl.col(c).rank("average") / pl.len()).alias(f"rank_{c}") for c in present],
    )
    # how a user's short-window standing compares to their long-window one:
    # rising and falling users can share both raw values but not this
    if "rank_gmv_sum_30d" in out.columns and "rank_gmv_sum_all" in out.columns:
        out = out.with_columns(
            (pl.col("rank_gmv_sum_30d") - pl.col("rank_gmv_sum_all")).alias("rank_shift_30_all"),
            (pl.col("rank_gmv_sum_07d") - pl.col("rank_gmv_sum_90d")).alias("rank_shift_07_90"),
        )
    return out
