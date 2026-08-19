from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest_v3 import (
    hac_mean_t_stat,
)


def evaluate_model(
    filename: str,
    prediction_column: str,
    model_name: str,
) -> pd.DataFrame:

    path = Path(
        filename
    )

    if not path.exists():

        print(
            f"{model_name}: "
            f"{filename} not found"
        )

        return pd.DataFrame()


    data = pd.read_parquet(
        path
    )


    period_column = (
        "Test Period"
        if "Test Period" in data.columns
        else "Period"
    )


    data[
        period_column
    ] = pd.PeriodIndex(
        data[
            period_column
        ],
        freq="M",
    )


    required = [
        prediction_column,
        "Return",
        "Market Beta 24M",
    ]


    data = data.dropna(
        subset=required
    ).copy()


    monthly_rows = []


    for period, group in data.groupby(
        period_column
    ):

        group = group.copy()


        if len(group) < 100:
            continue


        #
        # --------------------------------------
        # RECONSTRUCT THE ACTUAL V7 TARGET
        # --------------------------------------
        #
        # Cross-sectional regression each month:
        #
        # Return_i =
        #     intercept
        #     + beta_coefficient * MarketBeta_i
        #     + residual_i
        #
        # The residual is the raw beta-neutral
        # realized stock-selection return.
        #

        beta = (
            group[
                "Market Beta 24M"
            ]
            .astype(float)
            .to_numpy()
        )


        realized_return = (
            group[
                "Return"
            ]
            .astype(float)
            .to_numpy()
        )


        design = np.column_stack(
            [
                np.ones(
                    len(group)
                ),
                beta,
            ]
        )


        coefficients, *_ = (
            np.linalg.lstsq(
                design,
                realized_return,
                rcond=None,
            )
        )


        fitted = (
            design
            @ coefficients
        )


        group[
            "True Beta Neutral Return"
        ] = (
            realized_return
            - fitted
        )


        #
        # --------------------------------------
        # CORRECT IC
        # --------------------------------------
        #

        true_ic = (
            group[
                prediction_column
            ]
            .corr(
                group[
                    "True Beta Neutral Return"
                ],
                method="spearman",
            )
        )


        #
        # If the original standardized target was
        # saved in predictions, this should produce
        # essentially the same Spearman IC.
        #

        stored_target_ic = np.nan


        if (
            "Beta Neutral Target"
            in group.columns
        ):

            stored_target_ic = (
                group[
                    prediction_column
                ]
                .corr(
                    group[
                        "Beta Neutral Target"
                    ],
                    method="spearman",
                )
            )


        prediction_beta_corr = (
            group[
                prediction_column
            ]
            .corr(
                group[
                    "Market Beta 24M"
                ],
                method="spearman",
            )
        )


        #
        # --------------------------------------
        # DECILES
        # --------------------------------------
        #
        # 1 = predicted worst
        # 10 = predicted best
        #

        group[
            "Decile"
        ] = pd.qcut(
            group[
                prediction_column
            ].rank(
                method="first"
            ),
            q=10,
            labels=range(
                1,
                11,
            ),
        ).astype(int)


        top = group[
            group[
                "Decile"
            ] == 10
        ]


        bottom = group[
            group[
                "Decile"
            ] == 1
        ]


        top_return = float(
            top[
                "True Beta Neutral Return"
            ].mean()
        )


        bottom_return = float(
            bottom[
                "True Beta Neutral Return"
            ].mean()
        )


        monthly_rows.append(
            {
                "Period":
                    period,

                "IC":
                    true_ic,

                "Stored Target IC":
                    stored_target_ic,

                "Prediction Beta Correlation":
                    prediction_beta_corr,

                "Top Beta Neutral Return":
                    top_return,

                "Bottom Beta Neutral Return":
                    bottom_return,

                "Long Short Beta Neutral":
                    (
                        top_return
                        - bottom_return
                    ),
            }
        )


    monthly = pd.DataFrame(
        monthly_rows
    )


    print()
    print(
        "======================================"
    )

    print(
        f"{model_name} CORRECT TARGET CHECK"
    )

    print(
        "======================================"
    )


    ic = monthly[
        "IC"
    ].dropna()


    print(
        f"Months: "
        f"{len(monthly)}"
    )

    print(
        f"Average True IC: "
        f"{ic.mean():.4f}"
    )

    print(
        f"Median True IC: "
        f"{ic.median():.4f}"
    )

    print(
        f"Positive IC Rate: "
        f"{(ic > 0).mean():.2%}"
    )

    print(
        f"IC HAC T-Stat: "
        f"{hac_mean_t_stat(ic, max_lag=3):.3f}"
    )


    if (
        monthly[
            "Stored Target IC"
        ].notna().any()
    ):

        stored = (
            monthly[
                "Stored Target IC"
            ].dropna()
        )

        print(
            f"Stored Target IC: "
            f"{stored.mean():.4f}"
        )


    beta_corr = (
        monthly[
            "Prediction Beta Correlation"
        ].dropna()
    )


    print(
        f"Prediction Beta Correlation: "
        f"{beta_corr.mean():.2%}"
    )


    top = monthly[
        "Top Beta Neutral Return"
    ]


    bottom = monthly[
        "Bottom Beta Neutral Return"
    ]


    spread = monthly[
        "Long Short Beta Neutral"
    ]


    print(
        f"Top Decile Beta-Neutral Return: "
        f"{top.mean():.2%}"
    )

    print(
        f"Bottom Decile Beta-Neutral Return: "
        f"{bottom.mean():.2%}"
    )

    print(
        f"Long Short Beta-Neutral Return: "
        f"{spread.mean():.2%}"
    )

    print(
        f"Long Short HAC T-Stat: "
        f"{hac_mean_t_stat(spread, max_lag=3):.3f}"
    )


    #
    # --------------------------------------
    # HALF-PERIOD CHECK
    # --------------------------------------
    #

    monthly[
        "Year"
    ] = (
        monthly[
            "Period"
        ].dt.year
    )


    for name, subset in [

        (
            "2016-2020",
            monthly[
                monthly[
                    "Year"
                ] <= 2020
            ],
        ),

        (
            "2021-2025",
            monthly[
                monthly[
                    "Year"
                ] >= 2021
            ],
        ),

    ]:

        sub_ic = (
            subset[
                "IC"
            ].dropna()
        )

        sub_spread = (
            subset[
                "Long Short Beta Neutral"
            ].dropna()
        )


        print()
        print(name)

        print(
            f"  IC: "
            f"{sub_ic.mean():.4f} "
            f"(HAC "
            f"{hac_mean_t_stat(sub_ic, max_lag=3):.2f})"
        )

        print(
            f"  Long Short: "
            f"{sub_spread.mean():.2%} "
            f"(HAC "
            f"{hac_mean_t_stat(sub_spread, max_lag=3):.2f})"
        )


    return monthly


v7 = evaluate_model(
    filename=(
        "data/cache/"
        "v7_predictions.parquet"
    ),
    prediction_column=(
        "V7 Prediction"
    ),
    model_name="V7",
)


v8 = evaluate_model(
    filename=(
        "data/cache/"
        "v8_predictions.parquet"
    ),
    prediction_column=(
        "V8 Prediction"
    ),
    model_name="V8A",
)


if not v7.empty:

    v7.to_csv(
        "data/cache/"
        "v7_correct_target_check.csv",
        index=False,
    )


if not v8.empty:

    v8.to_csv(
        "data/cache/"
        "v8_correct_target_check.csv",
        index=False,
    )