from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_monthly_prices(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert daily adjusted prices into month-end prices
    and monthly returns.
    """

    data = prices[
        [
            "Date",
            "Adj Close",
        ]
    ].copy()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .dropna()
        .sort_values("Date")
        .drop_duplicates("Date")
    )

    data["Period"] = (
        data["Date"].dt.to_period("M")
    )

    monthly = (
        data
        .groupby(
            "Period",
            as_index=False,
        )
        .agg(
            Price=("Adj Close", "last"),
            Date=("Date", "last"),
        )
    )

    monthly["Return"] = (
        monthly["Price"].pct_change()
    )

    return monthly


def build_stock_features(
    stock_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build monthly predictive features.

    Every feature at month t uses only information
    available BEFORE month t begins.

    Target:
        stock return during month t
        minus SPY return during month t
    """

    stock = prepare_monthly_prices(
        stock_prices
    )

    benchmark = prepare_monthly_prices(
        benchmark_prices
    ).rename(
        columns={
            "Return": "Benchmark Return",
        }
    )

    data = stock.merge(
        benchmark[
            [
                "Period",
                "Benchmark Return",
            ]
        ],
        on="Period",
        how="inner",
    )

    data["Excess Return"] = (
        data["Return"]
        - data["Benchmark Return"]
    )
    #
    # ROLLING MARKET BETA
    #
    # Everything is shifted by one month first,
    # so the beta for month t uses only information
    # available before month t.
    #

    lagged_stock_return = (
        data["Return"].shift(1)
    )

    lagged_market_return = (
        data["Benchmark Return"].shift(1)
    )


    rolling_covariance = (
        lagged_stock_return
        .rolling(24)
        .cov(lagged_market_return)
    )


    rolling_market_variance = (
        lagged_market_return
        .rolling(24)
        .var()
    )


    data["Market Beta 24M"] = (
        rolling_covariance
        / rolling_market_variance
    )


    #
    #       REALIZED MARKET-RESIDUAL RETURN
    #
    # Beta was known before the target month.
    # The current month's return is the label.
    #

    data["Residual Return"] = (
        data["Return"]
        - (
            data["Market Beta 24M"]
            * data["Benchmark Return"]
        )
    )
    #
    # TARGET CALENDAR MONTH
    #

    data["Calendar Month"] = (
        data["Period"].dt.month
    )

    #
    # ORDINARY MOMENTUM
    #
    # shift(1) is critical:
    # current month's return cannot enter
    # current month's prediction.
    #

    lagged_return = (
        data["Return"].shift(1)
    )

    data["Momentum 3M"] = (
        (
            1 + lagged_return
        )
        .rolling(3)
        .apply(
            np.prod,
            raw=True,
        )
        - 1
    )

    data["Momentum 6M"] = (
        (
            1 + lagged_return
        )
        .rolling(6)
        .apply(
            np.prod,
            raw=True,
        )
        - 1
    )

    data["Momentum 12M"] = (
        (
            1 + lagged_return
        )
        .rolling(12)
        .apply(
            np.prod,
            raw=True,
        )
        - 1
    )

    #
    # SHORT-TERM REVERSAL
    #

    data["Last Month Return"] = (
        data["Return"].shift(1)
    )

    data["Last Month Excess"] = (
        data["Excess Return"].shift(1)
    )

    #
    # VOLATILITY
    #

    data["Volatility 6M"] = (
        lagged_return
        .rolling(6)
        .std()
    )

    data["Volatility 12M"] = (
        lagged_return
        .rolling(12)
        .std()
    )

    #
    # SAME-CALENDAR-MONTH SEASONALITY
    #
    # For September 2025, for example,
    # this uses previous Septembers only.
    #

    seasonal_mean = []
    seasonal_recent = []
    seasonal_median = []
    seasonal_beat = []
    seasonal_q25 = []
    seasonal_samples = []

    for index, row in data.iterrows():

        month = row["Calendar Month"]

        history = data.iloc[:index]

        history = history[
            history["Calendar Month"]
            == month
        ]

        excess = history[
            "Excess Return"
        ].dropna()

        seasonal_samples.append(
            len(excess)
        )

        if len(excess) == 0:

            seasonal_mean.append(np.nan)
            seasonal_recent.append(np.nan)
            seasonal_median.append(np.nan)
            seasonal_beat.append(np.nan)
            seasonal_q25.append(np.nan)

            continue

        seasonal_mean.append(
            float(excess.mean())
        )

        seasonal_median.append(
            float(excess.median())
        )

        seasonal_beat.append(
            float(
                (excess > 0).mean()
            )
        )

        seasonal_q25.append(
            float(
                excess.quantile(0.25)
            )
        )

        seasonal_recent.append(
            float(
                excess.tail(5).mean()
            )
        )

    data["Season Mean"] = seasonal_mean
    data["Season Recent"] = seasonal_recent
    data["Season Median"] = seasonal_median
    data["Season Beat Rate"] = seasonal_beat
    data["Season Q25"] = seasonal_q25
    data["Season Samples"] = seasonal_samples

    #
    # MARKET REGIME FEATURES
    #

    benchmark_lag = (
        data[
            "Benchmark Return"
        ].shift(1)
    )

    data["Market Momentum 3M"] = (
        (
            1 + benchmark_lag
        )
        .rolling(3)
        .apply(
            np.prod,
            raw=True,
        )
        - 1
    )

    data["Market Momentum 6M"] = (
        (
            1 + benchmark_lag
        )
        .rolling(6)
        .apply(
            np.prod,
            raw=True,
        )
        - 1
    )

    data["Market Volatility 6M"] = (
        benchmark_lag
        .rolling(6)
        .std()
    )

    return data