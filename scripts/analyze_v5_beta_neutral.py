from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest_v3 import (
    hac_mean_t_stat,
)


predictions = pd.read_parquet(
    "data/cache/v5_predictions.parquet"
)


period_column = (
    "Test Period"
    if "Test Period" in predictions.columns
    else "Period"
)


predictions[
    period_column
] = pd.PeriodIndex(
    predictions[
        period_column
    ],
    freq="M",
)


required = [
    "Ticker",
    "V5 Prediction",
    "Market Beta 24M",
    "Return",
    "Residual Return",
    "Benchmark Return",
]


data = predictions.dropna(
    subset=required
).copy()


adjusted_frames = []
monthly_signal_rows = []
monthly_decile_rows = []


for period, group in data.groupby(
    period_column
):

    group = group.copy()

    if len(group) < 100:
        continue


    #
    # -----------------------------------------
    # REMOVE CROSS-SECTIONAL BETA EXPOSURE
    # -----------------------------------------
    #
    # Regress:
    #
    #     V5 prediction
    #
    # on:
    #
    #     intercept + known 24M beta
    #
    # using only the current month's cross-section.
    #
    # No realized return enters this regression.
    #

    beta = (
        group[
            "Market Beta 24M"
        ]
        .astype(float)
        .to_numpy()
    )


    prediction = (
        group[
            "V5 Prediction"
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
            prediction,
            rcond=None,
        )
    )


    fitted_prediction = (
        design
        @ coefficients
    )


    group[
        "Beta Neutral Prediction"
    ] = (
        prediction
        - fitted_prediction
    )


    #
    # -----------------------------------------
    # SIGNAL DIAGNOSTICS
    # -----------------------------------------
    #

    raw_residual_ic = (
        group[
            "V5 Prediction"
        ]
        .corr(
            group[
                "Residual Return"
            ],
            method="spearman",
        )
    )


    neutral_residual_ic = (
        group[
            "Beta Neutral Prediction"
        ]
        .corr(
            group[
                "Residual Return"
            ],
            method="spearman",
        )
    )


    raw_excess_ic = (
        group[
            "V5 Prediction"
        ]
        .corr(
            (
                group[
                    "Return"
                ]
                - group[
                    "Benchmark Return"
                ]
            ),
            method="spearman",
        )
    )


    neutral_excess_ic = (
        group[
            "Beta Neutral Prediction"
        ]
        .corr(
            (
                group[
                    "Return"
                ]
                - group[
                    "Benchmark Return"
                ]
            ),
            method="spearman",
        )
    )


    raw_beta_corr = (
        group[
            "V5 Prediction"
        ]
        .corr(
            group[
                "Market Beta 24M"
            ],
            method="spearman",
        )
    )


    neutral_beta_corr = (
        group[
            "Beta Neutral Prediction"
        ]
        .corr(
            group[
                "Market Beta 24M"
            ],
            method="spearman",
        )
    )


    monthly_signal_rows.append(
        {
            "Period":
                period,

            "Raw Residual IC":
                raw_residual_ic,

            "Neutral Residual IC":
                neutral_residual_ic,

            "Raw Excess IC":
                raw_excess_ic,

            "Neutral Excess IC":
                neutral_excess_ic,

            "Raw Beta Correlation":
                raw_beta_corr,

            "Neutral Beta Correlation":
                neutral_beta_corr,
        }
    )


    #
    # -----------------------------------------
    # DECILE ANALYSIS
    # -----------------------------------------
    #
    # 1 = predicted worst
    # 10 = predicted best
    #

    group[
        "Neutral Decile"
    ] = pd.qcut(
        group[
            "Beta Neutral Prediction"
        ].rank(
            method="first"
        ),
        q=10,
        labels=range(
            1,
            11,
        ),
    ).astype(int)


    benchmark_return = float(
        group[
            "Benchmark Return"
        ].iloc[0]
    )


    for decile, bucket in group.groupby(
        "Neutral Decile"
    ):

        raw_return = float(
            bucket[
                "Return"
            ].mean()
        )


        monthly_decile_rows.append(
            {
                "Period":
                    period,

                "Decile":
                    int(decile),

                "Return":
                    raw_return,

                "Excess Return":
                    (
                        raw_return
                        - benchmark_return
                    ),

                "Residual Return":
                    float(
                        bucket[
                            "Residual Return"
                        ].mean()
                    ),

                "Beta":
                    float(
                        bucket[
                            "Market Beta 24M"
                        ].mean()
                    ),

                "Stocks":
                    len(bucket),
            }
        )


    adjusted_frames.append(
        group
    )


signal_monthly = pd.DataFrame(
    monthly_signal_rows
)


decile_monthly = pd.DataFrame(
    monthly_decile_rows
)


adjusted_predictions = pd.concat(
    adjusted_frames,
    ignore_index=True,
)


#
# -----------------------------------------
# SIGNAL SUMMARY
# -----------------------------------------
#

print()
print(
    "======================================"
)

print(
    "BETA-NEUTRAL V5 SIGNAL"
)

print(
    "======================================"
)


signal_columns = [
    "Raw Residual IC",
    "Neutral Residual IC",
    "Raw Excess IC",
    "Neutral Excess IC",
    "Raw Beta Correlation",
    "Neutral Beta Correlation",
]


for column in signal_columns:

    values = (
        signal_monthly[
            column
        ].dropna()
    )

    print()
    print(column)

    print(
        f"  Mean: "
        f"{values.mean():.3f}"
    )

    print(
        f"  Median: "
        f"{values.median():.3f}"
    )

    print(
        f"  Positive Rate: "
        f"{(values > 0).mean():.2%}"
    )

    print(
        f"  HAC T-Stat: "
        f"{hac_mean_t_stat(values, max_lag=3):.3f}"
    )


#
# -----------------------------------------
# BETA-NEUTRAL DECILES
# -----------------------------------------
#

summary_rows = []


for decile, group in (
    decile_monthly.groupby(
        "Decile"
    )
):

    residual = (
        group[
            "Residual Return"
        ]
    )


    summary_rows.append(
        {
            "Decile":
                decile,

            "Average Return":
                group[
                    "Return"
                ].mean(),

            "Average Excess":
                group[
                    "Excess Return"
                ].mean(),

            "Average Residual":
                residual.mean(),

            "Residual HAC T":
                hac_mean_t_stat(
                    residual,
                    max_lag=3,
                ),

            "Average Beta":
                group[
                    "Beta"
                ].mean(),
        }
    )


summary = pd.DataFrame(
    summary_rows
).sort_values(
    "Decile"
)


print()
print(
    "======================================"
)

print(
    "BETA-NEUTRAL DECILE ATTRIBUTION"
)

print(
    "1 = WORST"
)

print(
    "10 = BEST"
)

print(
    "======================================"
)


display = summary.copy()


for column in [
    "Average Return",
    "Average Excess",
    "Average Residual",
]:

    display[
        column
    ] = (
        display[
            column
        ]
        * 100
    ).round(2)


display[
    "Residual HAC T"
] = (
    display[
        "Residual HAC T"
    ].round(2)
)


display[
    "Average Beta"
] = (
    display[
        "Average Beta"
    ].round(2)
)


print(
    display.to_string(
        index=False
    )
)


#
# -----------------------------------------
# HALF-PERIOD ROBUSTNESS
# -----------------------------------------
#

signal_monthly[
    "Year"
] = (
    signal_monthly[
        "Period"
    ].dt.year
)


for name, subset in [

    (
        "2016-2020",
        signal_monthly[
            signal_monthly[
                "Year"
            ] <= 2020
        ],
    ),

    (
        "2021-2025",
        signal_monthly[
            signal_monthly[
                "Year"
            ] >= 2021
        ],
    ),

]:

    print()
    print(
        "======================================"
    )

    print(name)

    print(
        "======================================"
    )

    for column in [
        "Raw Residual IC",
        "Neutral Residual IC",
        "Raw Excess IC",
        "Neutral Excess IC",
    ]:

        values = subset[
            column
        ].dropna()

        print(
            f"{column}: "
            f"{values.mean():.3f} "
            f"(HAC "
            f"{hac_mean_t_stat(values, max_lag=3):.2f})"
        )


#
# Save everything.
#

signal_monthly.to_csv(
    "data/cache/v5_beta_neutral_signal.csv",
    index=False,
)


decile_monthly.to_csv(
    "data/cache/v5_beta_neutral_deciles.csv",
    index=False,
)


adjusted_predictions.to_parquet(
    "data/cache/v5_beta_neutral_predictions.parquet",
    index=False,
)