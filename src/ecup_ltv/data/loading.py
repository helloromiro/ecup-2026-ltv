from pathlib import Path

import polars as pl

# Raw event columns arrive as int64/float64; the value ranges are small
# (daily per-user counts and RUB amounts), so downcasting roughly halves
# memory footprint with no precision loss for counts and negligible loss
# for float32 GMV sums.
_DOWNCAST_INT8 = ["search", "cat", "has_search_to_cart", "has_search_to_ord", "has_cat_to_cart", "has_cat_to_ord"]
_DOWNCAST_INT16 = ["search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord", "to_cart", "to_ord", "searches"]
_DOWNCAST_INT32 = ["user_id"]
_DOWNCAST_FLOAT32 = ["gmv", "gmv_search", "gmv_cat"]


def load_raw_events(path: Path) -> pl.DataFrame:
    """Load the daily user-activity log and downcast dtypes for memory efficiency.

    The competition warns that fully densifying these sparse per-user-day
    records (filling in zero-activity days) can blow the data up to ~100M
    rows, so callers must keep working with the sparse form and let
    feature aggregation (see features.py) fill in zeros only for the
    aggregated window statistics, never for the raw event log itself.
    """
    df = pl.read_parquet(path)
    cast_exprs = (
        [pl.col(c).cast(pl.Int8) for c in _DOWNCAST_INT8]
        + [pl.col(c).cast(pl.Int16) for c in _DOWNCAST_INT16]
        + [pl.col(c).cast(pl.Int32) for c in _DOWNCAST_INT32]
        + [pl.col(c).cast(pl.Float32) for c in _DOWNCAST_FLOAT32]
    )
    return df.with_columns(cast_exprs)


def load_user_ids(sample_submission_path: Path) -> list[int]:
    """The set of 250k users to score, taken from the official sample submission
    rather than the training log, so we always predict for exactly the
    required population even if some users have zero rows in the log."""
    return pl.read_csv(sample_submission_path)["user_id"].sort().to_list()
