"""Cross-user structure: what a user looks like relative to the population.

Every one of the 482 existing features is computed from one user's own log in
isolation. Two things are therefore missing entirely:

* **which days**, not just how many. Two users with identical
  `gmv_sum_30d` and `active_days_30d` are indistinguishable to the model even
  if one bought on three promo days that half the platform also bought on and
  the other bought on three quiet Tuesdays. Co-activity on the same calendar
  days is a real signal in e-commerce (promotions, paydays, holidays) and no
  window aggregate can express it.
* **borrowed strength.** A user with four purchases has a very noisy profile.
  Projecting them onto components fitted across all 250k users replaces that
  noise with the population's shared structure - the sparse-data case where
  per-user aggregates are weakest is exactly where this helps most.

Both come out of a truncated SVD of the user x time matrix.

The construction detail that decides whether this works at all: the basis is
fitted ONCE on all anchors stacked, not per anchor. Component sign and order
are arbitrary in any single SVD, so a per-anchor fit would make "component 3"
mean something different in every fold and the tree could not use it. Stacking
gives one basis and coordinates that are comparable across folds.

No target is involved anywhere, and every matrix cell comes from events on or
before that row's anchor, so there is nothing to leak.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

PROMO_WINDOWS = [("90d", 89), ("180d", 179)]
TOP_DAY_SHARE = 0.05  # the platform's busiest 5% of days

DAY_WINDOW = 120  # relative days back from the anchor for the gmv matrix
WEEK_WINDOW = 26  # relative weeks back for the activity matrix
K_GMV = 16
K_ACT = 8
SEED = 42


def _stacked_matrix(
    data: pl.DataFrame, anchors: list[date], users: list[int], value: str, window: int, weekly: bool
) -> csr_matrix:
    """Rows are (anchor, user) pairs in the order anchors x users; columns are
    time buckets counted back from that row's own anchor."""
    n_users = len(users)
    uidx = pl.DataFrame({"user_id": users, "_u": np.arange(n_users, dtype=np.int64)})

    rows, cols, vals = [], [], []
    for a_i, anchor in enumerate(anchors):
        span = window * (7 if weekly else 1)
        start = anchor - timedelta(days=span - 1)
        sub = (
            data.filter((pl.col("event_date") >= start) & (pl.col("event_date") <= anchor) & (pl.col(value) > 0))
            .join(uidx, on="user_id", how="inner")
            .with_columns(((pl.lit(anchor) - pl.col("event_date")).dt.total_days()).alias("_back"))
        )
        if weekly:
            sub = sub.with_columns((pl.col("_back") // 7).alias("_bucket"))
        else:
            sub = sub.with_columns(pl.col("_back").alias("_bucket"))
        agg = sub.group_by(["_u", "_bucket"]).agg(pl.col(value).sum().alias("_v"))
        rows.append(agg["_u"].to_numpy() + a_i * n_users)
        cols.append(agg["_bucket"].to_numpy())
        vals.append(np.log1p(agg["_v"].to_numpy()))

    return csr_matrix(
        (np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(anchors) * n_users, window),
    )


def compute_cross_user_factors(
    data: pl.DataFrame, anchors: list[date], users: list[int]
) -> dict[date, pl.DataFrame]:
    """One DataFrame of SVD coordinates per anchor, keyed by anchor date."""
    out: dict[date, list[np.ndarray]] = {a: [] for a in anchors}
    names: list[str] = []

    for value, window, weekly, k, tag in [
        ("gmv", DAY_WINDOW, False, K_GMV, "gmvday"),
        ("searches", WEEK_WINDOW, True, K_ACT, "actweek"),
    ]:
        m = _stacked_matrix(data, anchors, users, value, window, weekly)
        svd = TruncatedSVD(n_components=k, random_state=SEED)
        f = svd.fit_transform(m).astype(np.float32)
        names += [f"svd_{tag}_{i:02d}" for i in range(k)]
        n_users = len(users)
        for a_i, a in enumerate(anchors):
            out[a].append(f[a_i * n_users : (a_i + 1) * n_users])

    frames = {}
    for a in anchors:
        block = np.hstack(out[a])
        frames[a] = pl.DataFrame({"user_id": users}).with_columns(
            [pl.Series(n, block[:, i]) for i, n in enumerate(names)]
        )
    return frames


def compute_promo_response(
    data: pl.DataFrame, anchor: date, users: list[int], platform_daily: pl.DataFrame
) -> pl.DataFrame:
    """Does this user shop when the whole platform shops?

    The SVD above cannot answer that: indexing days relative to each anchor
    aligns every user to their own timeline and throws away which CALENDAR day
    a purchase fell on, which is the entire promo signal. These features ask it
    directly, are identically defined at every anchor, and are absent from the
    482-feature pool in any form.
    """
    exprs: list[pl.Expr] = []
    for name, off in PROMO_WINDOWS:
        start = anchor - timedelta(days=off)
        win = platform_daily.filter(pl.col("event_date").is_between(start, anchor)).sort("gmv_sum", descending=True)
        top_days = win.head(max(1, int(len(win) * TOP_DAY_SHARE)))["event_date"].to_list()

        inw = pl.col("event_date").is_between(start, anchor)
        on_top = inw & pl.col("event_date").is_in(top_days)
        tot = pl.when(inw).then(pl.col("gmv")).otherwise(0.0).sum()
        exprs += [
            # share of the user's spend that lands on the platform's busiest days
            (pl.when(on_top).then(pl.col("gmv")).otherwise(0.0).sum() / (tot + 1.0)).alias(f"promo_gmv_share_{name}"),
            (pl.when(on_top & (pl.col("gmv") > 0)).then(1).otherwise(0).sum()
             / (pl.when(inw & (pl.col("gmv") > 0)).then(1).otherwise(0).sum() + 1.0)
             ).alias(f"promo_day_share_{name}"),
            # how much busier than usual the platform was on the days this
            # user chose: a ratio, not a correlation - pl.corr over a masked
            # column returns NaN for everyone whose window has a constant
            # side, which here is most of the population
            (pl.when(inw & (pl.col("gmv") > 0)).then(pl.col("_platform_gmv")).otherwise(None).mean()
             / (pl.when(inw).then(pl.col("_platform_gmv")).otherwise(None).mean() + 1.0)
             ).alias(f"promo_lift_gmv_{name}"),
            (pl.when(inw & (pl.col("searches") > 0)).then(pl.col("_platform_searches")).otherwise(None).mean()
             / (pl.when(inw).then(pl.col("_platform_searches")).otherwise(None).mean() + 1.0)
             ).alias(f"promo_lift_act_{name}"),
        ]

    ad = (
        data.filter(pl.col("user_id").is_in(users) & (pl.col("event_date") <= anchor))
        .join(
            platform_daily.select(
                "event_date",
                pl.col("gmv_sum").alias("_platform_gmv"),
                pl.col("searches_sum").alias("_platform_searches"),
            ),
            on="event_date",
            how="left",
        )
    )
    feats = ad.group_by("user_id").agg(exprs)
    out = pl.DataFrame({"user_id": users}).join(feats, on="user_id", how="left")
    return out.with_columns(
        [pl.col(c).fill_null(0.0).fill_nan(0.0) for c in out.columns if c != "user_id"]
    )
