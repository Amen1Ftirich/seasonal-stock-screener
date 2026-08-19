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
    "V5 Prediction",
    "Return",
    "Residual Return",
    "Benchmark Return",
    "Market Beta 24M",
]


data = predictions.dropna(
    subset=required
).copy()


monthly_deciles = []


for period, group in data.groupby(
    period_column
):

    group = group.copy()

    if len(group) < 100:
        continue

    #
    # qcut labels:
    #
    # 1 = predicted worst
    # 10 = predicted best
    #

    group[
        "Decile"
    ] = pd.qcut(
        group[
            "V5 Prediction"
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
        "Decile"
    ):

        raw_return = float(
            bucket[
                "Return"
            ].mean()
        )

        monthly_deciles.append(
            {
                "Period":
                    period,

                "Decile":
                    int(decile),

                "Stocks":
                    len(bucket),

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
            }
        )


monthly = pd.DataFrame(
    monthly_deciles
)


#
# -----------------------------------------
# FULL-PERIOD DECILE SUMMARY
# -----------------------------------------
#

rows = []


for decile, group in monthly.groupby(
    "Decile"
):

    residual = group[
        "Residual Return"
    ]

    excess = group[
        "Excess Return"
    ]

    rows.append(
        {
            "Decile":
                decile,

            "Average Return":
                group[
                    "Return"
                ].mean(),

            "Average Excess":
                excess.mean(),

            "Average Residual":
                residual.mean(),

            "Residual HAC T":
                hac_mean_t_stat(
                    residual,
                    max_lag=3,
                ),

            "Positive Residual Rate":
                (
                    residual > 0
                ).mean(),

            "Average Beta":
                group[
                    "Beta"
                ].mean(),
        }
    )


summary = pd.DataFrame(
    rows
).sort_values(
    "Decile"
)


print()
print(
    "======================================"
)

print(
    "V5 DECILE ATTRIBUTION"
)

print(
    "1 = WORST PREDICTION"
)

print(
    "10 = BEST PREDICTION"
)

print(
    "======================================"
)


display = summary.copy()


for column in [
    "Average Return",
    "Average Excess",
    "Average Residual",
    "Positive Residual Rate",
]:

    display[
        column
    ] = (
        display[column]
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
# TOP VERSUS BOTTOM HALF-PERIOD STABILITY
# -----------------------------------------
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

    print()
    print(
        "======================================"
    )

    print(name)

    print(
        "======================================"
    )

    period_summary = (
        subset
        .groupby(
            "Decile"
        )
        .agg(
            Residual=(
                "Residual Return",
                "mean",
            ),

            Excess=(
                "Excess Return",
                "mean",
            ),

            Beta=(
                "Beta",
                "mean",
            ),
        )
    )


    period_summary[
        "Residual"
    ] *= 100

    period_summary[
        "Excess"
    ] *= 100


    print(
        period_summary
        .round(2)
        .to_string()
    )


#
# Save for later research.
#

monthly.to_csv(
    "data/cache/v5_decile_monthly.csv",
    index=False,
)

summary.to_csv(
    "data/cache/v5_decile_summary.csv",
    index=False,
)