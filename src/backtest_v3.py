from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_v3 import (
    FEATURE_COLUMNS,
    predict_cross_section,
    train_model,
)


def run_walk_forward_v3(
    panel: pd.DataFrame,
    start_year: int = 2016,
    end_year: int = 2025,
    top_n: int = 10,
    training_years: int = 10,
    alpha: float = 10.0,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    True monthly walk-forward simulation.

    Every month:

        Train using ONLY earlier months.
        Predict current-month excess returns.
        Buy top N predictions.
        Reveal actual returns.

    No current/future month enters training.
    """

    data = panel.copy()

    data["Period"] = (
        data["Period"]
        .astype("period[M]")
    )

    test_periods = sorted(
        data[
            (
                data["Period"].dt.year
                >= start_year
            )
            & (
                data["Period"].dt.year
                <= end_year
            )
        ]["Period"].unique()
    )

    trades = []
    monthly_results = []

    for number, period in enumerate(
        test_periods,
        start=1,
    ):

        print(
            f"[{number}/{len(test_periods)}] "
            f"{period}"
        )

        test = data[
            data["Period"] == period
        ].copy()

        if test.empty:
            continue

        training_end = period - 1

        training_start = (
            period
            - training_years * 12
        )

        train = data[
            (
                data["Period"]
                >= training_start
            )
            & (
                data["Period"]
                <= training_end
            )
        ].copy()

        train = train.dropna(
            subset=(
                FEATURE_COLUMNS
                + [
                    "Excess Return",
                ]
            )
        )

        test = test.dropna(
            subset=FEATURE_COLUMNS
        )

        if (
            train.empty
            or len(test) < top_n
        ):
            continue

        model = train_model(
            training_data=train,
            alpha=alpha,
        )

        predicted = (
            predict_cross_section(
                model=model,
                cross_section=test,
            )
        )

        selected = (
            predicted
            .head(top_n)
            .copy()
        )

        selected[
            "Test Period"
        ] = period

        trades.append(
            selected
        )

        portfolio_return = float(
            selected["Return"].mean()
        )

        benchmark_return = float(
            selected[
                "Benchmark Return"
            ].iloc[0]
        )

        excess_return = (
            portfolio_return
            - benchmark_return
        )

        monthly_results.append(
            {
                "Period": period,

                "Stocks":
                    len(selected),

                "Portfolio Return":
                    portfolio_return,

                "Benchmark Return":
                    benchmark_return,

                "Excess Return":
                    excess_return,

                "Beat Benchmark":
                    excess_return > 0,

                "Positive Month":
                    portfolio_return > 0,
            }
        )

    if trades:

        trade_df = pd.concat(
            trades,
            ignore_index=True,
        )

    else:

        trade_df = pd.DataFrame()

    monthly_df = pd.DataFrame(
        monthly_results
    )

    return (
        trade_df,
        monthly_df,
    )


def summarize_v3(
    monthly_results: pd.DataFrame,
) -> dict:

    if monthly_results.empty:

        return {}

    excess = (
        monthly_results[
            "Excess Return"
        ]
    )

    returns = (
        monthly_results[
            "Portfolio Return"
        ]
    )

    return {
        "Months":
            len(monthly_results),

        "Positive Month Rate":
            float(
                monthly_results[
                    "Positive Month"
                ].mean()
            ),

        "Beat SPY Rate":
            float(
                monthly_results[
                    "Beat Benchmark"
                ].mean()
            ),

        "Average Monthly Return":
            float(
                returns.mean()
            ),

        "Median Monthly Return":
            float(
                returns.median()
            ),

        "Average Monthly Excess":
            float(
                excess.mean()
            ),

        "Median Monthly Excess":
            float(
                excess.median()
            ),

        "Excess Volatility":
            float(
                excess.std(ddof=1)
            ),

        "Worst Month":
            float(
                returns.min()
            ),

        "Best Month":
            float(
                returns.max()
            ),
    }