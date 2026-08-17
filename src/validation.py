from __future__ import annotations

import pandas as pd

from src.metrics import (
    calculate_metrics,
    calculate_relative_metrics,
)


def chronological_split(
    observations: pd.DataFrame,
    test_years: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split seasonal observations chronologically.

    Example with 15 years:
        first 10 years -> training
        last 5 years   -> out-of-sample
    """

    if observations.empty:
        raise ValueError("Observations are empty")

    if "Year" not in observations.columns:
        raise ValueError("Observations must contain a Year column")

    data = (
        observations
        .sort_values("Year")
        .reset_index(drop=True)
    )

    years = sorted(
        data["Year"]
        .dropna()
        .unique()
    )

    if len(years) <= test_years:
        raise ValueError(
            "Not enough years for train/test validation"
        )

    test_year_set = set(
        years[-test_years:]
    )

    train = data[
        ~data["Year"].isin(test_year_set)
    ].copy()

    test = data[
        data["Year"].isin(test_year_set)
    ].copy()

    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def calculate_validation_metrics(
    observations: pd.DataFrame,
    benchmark_name: str = "SPY",
    test_years: int = 5,
) -> dict:
    """
    Calculate training and out-of-sample metrics.
    """

    train, test = chronological_split(
        observations=observations,
        test_years=test_years,
    )

    train_absolute = calculate_metrics(train)

    train_relative = calculate_relative_metrics(
        train,
        benchmark_name=benchmark_name,
    )

    test_absolute = calculate_metrics(test)

    test_relative = calculate_relative_metrics(
        test,
        benchmark_name=benchmark_name,
    )

    return {
        # Training
        "Train Sample Size":
            train_absolute["sample_size"],

        "Train Win Rate":
            train_absolute["win_rate"],

        "Train Median Return":
            train_absolute["median_return"],

        "Train Wilson":
            train_absolute["wilson_lower_bound"],

        "Train Beat SPY Rate":
            train_relative["beat_benchmark_rate"],

        "Train Median Excess":
            train_relative["median_excess_return"],

        # Out-of-sample
        "OOS Sample Size":
            test_absolute["sample_size"],

        "OOS Win Rate":
            test_absolute["win_rate"],

        "OOS Median Return":
            test_absolute["median_return"],

        "OOS Beat SPY Rate":
            test_relative["beat_benchmark_rate"],

        "OOS Median Excess":
            test_relative["median_excess_return"],
    }