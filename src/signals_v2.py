from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def prepare_prices(
    prices: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "Date",
        "Adj Close",
    }

    missing = required.difference(
        prices.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

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
        .reset_index(drop=True)
    )

    return data


def monthly_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert daily adjusted prices into
    calendar-month total returns.
    """

    data = prepare_prices(prices)

    data["Month"] = (
        data["Date"].dt.to_period("M")
    )

    monthly = (
        data
        .groupby("Month", as_index=False)
        .agg(
            Date=("Date", "last"),
            Price=("Adj Close", "last"),
        )
    )

    monthly["Return"] = (
        monthly["Price"].pct_change()
    )

    return monthly.dropna(
        subset=["Return"]
    ).reset_index(drop=True)


def same_month_seasonality(
    stock_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    target_month: int,
    as_of_date: date,
    lookback_years: int = 15,
    recent_years: int = 5,
) -> dict:
    """
    Historical relative performance during the
    same calendar month.

    Example:
        target_month = 9

    asks how the stock historically performed
    during Septembers relative to SPY.
    """

    stock = monthly_returns(
        stock_prices
    ).rename(
        columns={
            "Return": "Stock Return"
        }
    )

    benchmark = monthly_returns(
        benchmark_prices
    ).rename(
        columns={
            "Return": "Benchmark Return"
        }
    )

    merged = stock[
        [
            "Month",
            "Stock Return",
        ]
    ].merge(
        benchmark[
            [
                "Month",
                "Benchmark Return",
            ]
        ],
        on="Month",
        how="inner",
    )

    cutoff = pd.Period(
        as_of_date,
        freq="M",
    )

    # Absolutely no current/future month leakage.
    merged = merged[
        merged["Month"] < cutoff
    ].copy()

    merged = merged[
        merged["Month"].dt.month
        == target_month
    ].copy()

    merged["Excess Return"] = (
        merged["Stock Return"]
        - merged["Benchmark Return"]
    )

    merged = (
        merged
        .sort_values("Month")
        .tail(lookback_years)
        .reset_index(drop=True)
    )

    if merged.empty:

        return {
            "Season Samples": 0,
            "Season Mean Excess": np.nan,
            "Season Median Excess": np.nan,
            "Season Win Rate": np.nan,
            "Season Beat Rate": np.nan,
            "Season Q25 Excess": np.nan,
            "Season Std": np.nan,
            "Recent Mean Excess": np.nan,
            "Recent Median Excess": np.nan,
        }

    excess = merged[
        "Excess Return"
    ]

    recent = merged.tail(
        recent_years
    )

    return {
        "Season Samples":
            len(merged),

        "Season Mean Excess":
            float(excess.mean()),

        "Season Median Excess":
            float(excess.median()),

        "Season Win Rate":
            float(
                (
                    merged["Stock Return"] > 0
                ).mean()
            ),

        "Season Beat Rate":
            float(
                (
                    merged["Excess Return"] > 0
                ).mean()
            ),

        "Season Q25 Excess":
            float(
                excess.quantile(0.25)
            ),

        "Season Std":
            float(
                excess.std(ddof=1)
            )
            if len(excess) > 1
            else 0.0,

        "Recent Mean Excess":
            float(
                recent[
                    "Excess Return"
                ].mean()
            ),

        "Recent Median Excess":
            float(
                recent[
                    "Excess Return"
                ].median()
            ),
    }


def momentum_12_1(
    prices: pd.DataFrame,
    as_of_date: date,
) -> float:
    """
    Approximate 12-1 momentum.

    Measures performance from roughly
    12 months ago to 1 month ago.

    The most recent ~21 trading sessions
    are excluded.
    """

    data = prepare_prices(prices)

    cutoff = pd.Timestamp(
        as_of_date
    )

    data = data[
        data["Date"] < cutoff
    ].reset_index(drop=True)

    if len(data) < 253:
        return np.nan

    start_price = float(
        data["Adj Close"].iloc[-253]
    )

    end_price = float(
        data["Adj Close"].iloc[-22]
    )

    if start_price <= 0:
        return np.nan

    return (
        end_price / start_price
    ) - 1