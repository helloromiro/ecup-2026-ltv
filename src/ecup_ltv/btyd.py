"""BTYD (Buy-Till-You-Die) modeling - explicitly suggested by the
competition's own task description as one of the approaches worth trying,
alongside classical ML and time-series methods.

Unlike every other model in this repo (all LightGBM variants on the same
anchor-window features), BG/NBD + Gamma-Gamma is a genuinely different
paradigm: a parametric statistical model of purchase timing (a Beta-
Geometric mixture over each customer's Poisson purchase rate and dropout
probability) fit on the whole population, then scored per user. It
produces predictions in the same units as the competition target (GMV
over a horizon), so besides being usable as extra LightGBM input features,
its raw prediction is directly blendable with the tree-model ensemble -
useful diversity, since its error structure has essentially nothing to do
with gradient boosting on window aggregates.

Each user's frequency/recency/T/monetary_value must be computed using only
activity up to the anchor date, same no-leakage discipline as
features.py - see compute_btyd_summary.
"""

from datetime import date

import numpy as np
import polars as pl
from lifetimes import BetaGeoFitter, GammaGammaFitter


def compute_btyd_summary(data: pl.DataFrame, anchor_date: date, user_ids: list[int]) -> pl.DataFrame:
    """Per-user (frequency, recency, T, monetary_value) as of anchor_date,
    in BTYD convention: frequency = number of REPEAT purchase-days (first
    purchase doesn't count), recency = age in days at the last purchase,
    T = age in days at the anchor (observation time), monetary_value =
    mean GMV per purchase-day among repeat purchases only.
    """
    hist = data.filter(pl.col("user_id").is_in(user_ids) & (pl.col("event_date") <= anchor_date) & (pl.col("to_ord") > 0))

    per_user_days = (
        hist.group_by("user_id")
        .agg(
            pl.col("event_date").min().alias("first_purchase_date"),
            pl.col("event_date").max().alias("last_purchase_date"),
            pl.col("event_date").n_unique().alias("n_purchase_days"),
            pl.col("gmv").sum().alias("total_gmv"),
        )
        .with_columns(
            frequency=(pl.col("n_purchase_days") - 1).clip(lower_bound=0),
            recency=(pl.col("last_purchase_date") - pl.col("first_purchase_date")).dt.total_days(),
            T=(pl.lit(anchor_date) - pl.col("first_purchase_date")).dt.total_days(),
        )
        .with_columns(
            # average GMV per purchase-day (we only have daily aggregates, not
            # per-order amounts, so "per transaction" here means "per purchase-day")
            monetary_value=pl.when(pl.col("frequency") > 0)
            .then(pl.col("total_gmv") / pl.col("n_purchase_days"))
            .otherwise(0.0)
        )
    )

    index_df = pl.DataFrame({"user_id": user_ids})
    result = index_df.join(
        per_user_days.select("user_id", "frequency", "recency", "T", "monetary_value"), on="user_id", how="left"
    )
    # never purchased: frequency/recency/monetary=0, T = full tenure (no purchase to anchor from) ->
    # BG/NBD requires T > 0; use tenure_days-equivalent (days since first ever activity, not just purchase)
    tenure = (
        data.filter(pl.col("user_id").is_in(user_ids) & (pl.col("event_date") <= anchor_date))
        .group_by("user_id")
        .agg((pl.lit(anchor_date) - pl.col("event_date").min()).dt.total_days().alias("tenure_days"))
    )
    result = result.join(tenure, on="user_id", how="left")
    result = result.with_columns(
        frequency=pl.col("frequency").fill_null(0.0),
        recency=pl.col("recency").fill_null(0.0),
        T=pl.coalesce([pl.col("T"), pl.col("tenure_days")]).fill_null(1.0).clip(lower_bound=1.0),
        monetary_value=pl.col("monetary_value").fill_null(0.0),
    ).drop("tenure_days")
    return result


def fit_and_score_btyd(summary_df: pl.DataFrame, horizon_days: int = 30, penalizer: float = 0.01) -> pl.DataFrame:
    """Fits BG/NBD + Gamma-Gamma on this anchor's population and scores
    every user. Returns [user_id, btyd_pred_purchases, btyd_prob_alive,
    btyd_expected_value, btyd_clv]. btyd_clv (predicted purchases x
    expected value per purchase) is on the same scale as the competition
    target and is the candidate for direct ensembling."""
    frequency = summary_df["frequency"].to_numpy().astype(np.float64)
    recency = summary_df["recency"].to_numpy().astype(np.float64)
    T = summary_df["T"].to_numpy().astype(np.float64)
    monetary_value = summary_df["monetary_value"].to_numpy().astype(np.float64)

    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(frequency, recency, T)

    pred_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(horizon_days, frequency, recency, T)
    prob_alive = bgf.conditional_probability_alive(frequency, recency, T)
    # lifetimes' BG/NBD conditional-expectation formula (a 2F1 hypergeometric
    # series) is numerically unstable for a small minority of edge cases -
    # empirically, near-zero-tenure users with frequency=0. Those users have
    # essentially no purchase signal anyway, so 0 is a reasonable fallback
    # rather than propagating NaN through the whole pipeline.
    pred_purchases = np.nan_to_num(pred_purchases, nan=0.0)
    prob_alive = np.nan_to_num(prob_alive, nan=0.0)

    repeat_mask = frequency > 0
    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    ggf.fit(frequency[repeat_mask], monetary_value[repeat_mask])
    expected_value = ggf.conditional_expected_average_profit(frequency, monetary_value)
    # for zero-frequency users, conditional_expected_average_profit returns the
    # population unconditional mean - a reasonable prior for "some value if they buy"
    expected_value = np.nan_to_num(expected_value, nan=0.0)

    clv = pred_purchases * expected_value

    return pl.DataFrame(
        {
            "user_id": summary_df["user_id"].to_numpy(),
            "btyd_pred_purchases": pred_purchases.astype(np.float32),
            "btyd_prob_alive": prob_alive.astype(np.float32),
            "btyd_expected_value": expected_value.astype(np.float32),
            "btyd_clv": clv.astype(np.float32),
        }
    )
