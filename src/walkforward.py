from __future__ import annotations

import pandas as pd

from src.metrics import (
    calculate_metrics,
    calculate_relative_metrics,
)


def walk_forward_validate(
    observations: pd.DataFrame,
    benchmark_name: str = "SPY",
    minimum_training_years: int = 8,
    minimum_win_rate: float = 0.70,
    minimum_median_return: float = 0.0,
    minimum_beat_benchmark_rate: float = 0.60,
    minimum_median_excess_return: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Expanding-window walk-forward validation.

    For every historical test year:
        1. Use only earlier years as training data.
        2. Calculate seasonal statistics.
        3. Determine whether the opportunity would have passed
           the screener at that time.
        4. Record what actually happened in the next year.

    Important:
    This validates an already-selected seasonal window.
    It does not re-discover the best window separately in every fold.
    """

    if observations.empty:
        raise ValueError("Observations are empty")

    required_columns = {
        "Year",
        "Return",
        f"{benchmark_name} Return",
        "Excess Return",
        f"Beat {benchmark_name}",
    }

    missing = required_columns.difference(
        observations.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    data = (
        observations
        .sort_values("Year")
        .reset_index(drop=True)
    )

    if len(data) <= minimum_training_years:
        raise ValueError(
            "Not enough observations for walk-forward validation"
        )

    folds = []

    for test_index in range(
        minimum_training_years,
        len(data),
    ):

        train = data.iloc[:test_index].copy()

        test_row = data.iloc[test_index]

        absolute = calculate_metrics(train)

        relative = calculate_relative_metrics(
            train,
            benchmark_name=benchmark_name,
        )

        qualifies = (
            absolute["sample_size"]
            >= minimum_training_years
            and absolute["win_rate"]
            >= minimum_win_rate
            and absolute["median_return"]
            >= minimum_median_return
            and relative["beat_benchmark_rate"]
            >= minimum_beat_benchmark_rate
            and relative["median_excess_return"]
            >= minimum_median_excess_return
        )

        folds.append(
            {
                "Test Year":
                    int(test_row["Year"]),

                "Training Years":
                    absolute["sample_size"],

                "Prior Win Rate":
                    absolute["win_rate"],

                "Prior Wilson":
                    absolute["wilson_lower_bound"],

                "Prior Median Return":
                    absolute["median_return"],

                "Prior Beat SPY Rate":
                    relative[
                        "beat_benchmark_rate"
                    ],

                "Prior Median Excess":
                    relative[
                        "median_excess_return"
                    ],

                "Qualified":
                    bool(qualifies),

                "Realized Return":
                    float(test_row["Return"]),

                "Realized Excess Return":
                    float(
                        test_row["Excess Return"]
                    ),

                "Realized Win":
                    bool(
                        test_row["Return"] > 0
                    ),

                "Realized Beat SPY":
                    bool(
                        test_row[
                            f"Beat {benchmark_name}"
                        ]
                    ),
            }
        )

    folds_df = pd.DataFrame(folds)

    qualified = folds_df[
        folds_df["Qualified"]
    ].copy()

    if qualified.empty:

        summary = {
            "WF Folds": len(folds_df),
            "WF Qualified Folds": 0,
            "WF Selection Rate": 0.0,
            "WF Win Rate": 0.0,
            "WF Beat SPY Rate": 0.0,
            "WF Median Return": 0.0,
            "WF Median Excess": 0.0,
        }

        return folds_df, summary

    summary = {
        "WF Folds":
            len(folds_df),

        "WF Qualified Folds":
            len(qualified),

        "WF Selection Rate":
            len(qualified) / len(folds_df),

        "WF Win Rate":
            float(
                qualified["Realized Win"].mean()
            ),

        "WF Beat SPY Rate":
            float(
                qualified[
                    "Realized Beat SPY"
                ].mean()
            ),

        "WF Median Return":
            float(
                qualified[
                    "Realized Return"
                ].median()
            ),

        "WF Median Excess":
            float(
                qualified[
                    "Realized Excess Return"
                ].median()
            ),
    }

    return folds_df, summary