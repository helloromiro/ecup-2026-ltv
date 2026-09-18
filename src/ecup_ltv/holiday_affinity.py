"""Per-user responsiveness to Russian gift-giving holidays (23 February,
8 March), computed once from their actual 2025 occurrence.

Motivation: the competition's real test target window (2026-02-14 to
2026-03-15) contains both dates - the two biggest gift-giving peaks in RU
e-commerce (see experiments/level_drift/notes.md). A direct measurement
(experiments/holiday_affinity/notes.md) found the resulting spend uplift
is NOT spread evenly across users: only ~16% of users show any uplift at
all, and the top 1% of users by uplift capture ~45% of the total. The
level-shift correction already applied (see calibration/level_drift work)
moves every user's prediction up by the same scalar, which is correct on
average but misallocates this concentrated effect at the individual level
- this feature lets the model learn who is actually holiday-responsive
instead of assuming everyone is.

Unlike every other feature in this repo, this one is NOT anchor-relative:
it always references the fixed 2025 occurrence of these two dates, so it
is only meaningful for anchors after ~2025-03-10 (true for all 19 anchors
this repo uses) and is computed once for the whole population, not
per-anchor like BTYD.

Caveat (see experiments/holiday_affinity/notes.md): no historical fold's
own target window contains an analogous major holiday, so this feature's
true predictive value cannot be backtested the way everything else in
this repo has been - it is a domain-reasoning bet, not an empirically
walk-forward-validated one.
"""

from datetime import date, timedelta

import polars as pl

_HOLIDAYS_2025 = [date(2025, 2, 23), date(2025, 3, 8)]
_WINDOW_RADIUS_DAYS = 2
_BASELINE_START, _BASELINE_END = date(2025, 1, 20), date(2025, 2, 15)

EPS = 1.0


def compute_holiday_affinity(data: pl.DataFrame, user_ids: list[int]) -> pl.DataFrame:
    holiday_days = []
    for h in _HOLIDAYS_2025:
        holiday_days += [h + timedelta(days=d) for d in range(-_WINDOW_RADIUS_DAYS, _WINDOW_RADIUS_DAYS + 1)]
    n_holiday_days = len(holiday_days)
    n_baseline_days = (_BASELINE_END - _BASELINE_START).days + 1

    baseline = (
        data.filter(pl.col("user_id").is_in(user_ids) & pl.col("event_date").is_between(_BASELINE_START, _BASELINE_END))
        .group_by("user_id")
        .agg(pl.col("gmv").sum().alias("_baseline_gmv"))
    )
    holiday = (
        data.filter(pl.col("user_id").is_in(user_ids) & pl.col("event_date").is_in(holiday_days))
        .group_by("user_id")
        .agg(pl.col("gmv").sum().alias("_holiday_gmv"))
    )

    index_df = pl.DataFrame({"user_id": user_ids})
    result = (
        index_df.join(baseline, on="user_id", how="left")
        .join(holiday, on="user_id", how="left")
        .with_columns([pl.col("_baseline_gmv").fill_null(0.0), pl.col("_holiday_gmv").fill_null(0.0)])
        .with_columns(
            (pl.col("_baseline_gmv") / n_baseline_days).alias("holiday_baseline_daily_gmv"),
            (pl.col("_holiday_gmv") / n_holiday_days).alias("holiday_window_daily_gmv"),
        )
        .with_columns(
            (pl.col("holiday_window_daily_gmv") - pl.col("holiday_baseline_daily_gmv")).alias("holiday_uplift_gmv"),
            (
                (pl.col("holiday_window_daily_gmv") - pl.col("holiday_baseline_daily_gmv"))
                / (pl.col("holiday_baseline_daily_gmv") + EPS)
            ).alias("holiday_uplift_ratio"),
        )
        .select("user_id", "holiday_uplift_gmv", "holiday_uplift_ratio", "holiday_baseline_daily_gmv")
    )
    return result


_HOLIDAY_LABELS = {date(2025, 2, 23): "feb23", date(2025, 3, 8): "mar8"}


def compute_holiday_affinity_split(data: pl.DataFrame, user_ids: list[int]) -> pl.DataFrame:
    """Same idea as compute_holiday_affinity, but Feb 23 (Defender of the
    Fatherland Day - skews male-gift) and Mar 8 (International Women's Day -
    skews female-gift) get SEPARATE uplift columns instead of one lumped
    window. A user responsive to one occasion is not necessarily responsive
    to the other; the combined feature cannot distinguish them. Untested
    like the combined version - see experiments/holiday_affinity_split/notes.md
    for the same backtestability caveat."""
    n_baseline_days = (_BASELINE_END - _BASELINE_START).days + 1
    baseline = (
        data.filter(pl.col("user_id").is_in(user_ids) & pl.col("event_date").is_between(_BASELINE_START, _BASELINE_END))
        .group_by("user_id")
        .agg(pl.col("gmv").sum().alias("_baseline_gmv"))
    )
    index_df = pl.DataFrame({"user_id": user_ids})
    result = index_df.join(baseline, on="user_id", how="left").with_columns(pl.col("_baseline_gmv").fill_null(0.0))
    result = result.with_columns((pl.col("_baseline_gmv") / n_baseline_days).alias("holiday_baseline_daily_gmv"))

    for h, label in _HOLIDAY_LABELS.items():
        h_days = [h + timedelta(days=d) for d in range(-_WINDOW_RADIUS_DAYS, _WINDOW_RADIUS_DAYS + 1)]
        n_h_days = len(h_days)
        h_gmv = (
            data.filter(pl.col("user_id").is_in(user_ids) & pl.col("event_date").is_in(h_days))
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias(f"_{label}_gmv"))
        )
        result = result.join(h_gmv, on="user_id", how="left").with_columns(pl.col(f"_{label}_gmv").fill_null(0.0))
        window_daily = pl.col(f"_{label}_gmv") / n_h_days
        result = result.with_columns(
            (window_daily - pl.col("holiday_baseline_daily_gmv")).alias(f"holiday_uplift_gmv_{label}"),
            (
                (window_daily - pl.col("holiday_baseline_daily_gmv"))
                / (pl.col("holiday_baseline_daily_gmv") + EPS)
            ).alias(f"holiday_uplift_ratio_{label}"),
        )

    keep = ["user_id", "holiday_baseline_daily_gmv"] + [
        f"holiday_uplift_{kind}_{label}" for label in _HOLIDAY_LABELS.values() for kind in ("gmv", "ratio")
    ]
    return result.select(keep)
