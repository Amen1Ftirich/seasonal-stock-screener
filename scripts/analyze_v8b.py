from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest_v3 import (
    hac_mean_t_stat,
)


MONTHLY_FILE = (
    "data/cache/v8b_monthly.csv"
)

PREDICTIONS_FILE = (
    "data/cache/v8b_predictions.parquet"
)


monthly = pd.read_csv(
    MONTHLY_FILE
)


predictions = pd.read_parquet(
    PREDICTIONS_FILE
)


# -----------------------------------------
# PERIOD CLEANUP
# -----------------------------------------

monthly[
    "Period"
] = pd.PeriodIndex(
    monthly[
        "Period"
    ],
    freq="M",
)


prediction_period_column = (
    "Test Period"
    if "Test Period"
    in predictions.columns
    else "Period"
)


predictions[
    prediction_period_column
] = pd.PeriodIndex(
    predictions[
        prediction_period_column
    ],
    freq="M",
)


# -----------------------------------------
# VERIFY REQUIRED COLUMNS
# -----------------------------------------

required_prediction_columns = [
    "V8B Prediction",
    "Beta Neutral Return",
    "Return",
    "Market Beta 24M",
]


missing = [
    column

    for column
    in required_prediction_columns

    if column
    not in predictions.columns
]


if missing:

    raise ValueError(
        "Missing prediction columns: "
        + str(missing)
    )


# =========================================
# OVERALL ROBUSTNESS
# =========================================

print()
print(
    "======================================"
)
print(
    "V8B ROBUSTNESS"
)
print(
    "======================================"
)


ic = monthly[
    "IC"
].dropna()


spread = monthly[
    "Long Short Residual"
].dropna()


beta_corr = monthly[
    "Prediction Beta Correlation"
].dropna()


print(
    f"Months: "
    f"{len(monthly)}"
)

print(
    f"Average IC: "
    f"{ic.mean():.4f}"
)

print(
    f"Median IC: "
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

print()

print(
    f"Average Long-Short "
    f"Beta-Neutral: "
    f"{spread.mean():.2%}"
)

print(
    f"Median Long-Short "
    f"Beta-Neutral: "
    f"{spread.median():.2%}"
)

print(
    f"Positive Spread Rate: "
    f"{(spread > 0).mean():.2%}"
)

print(
    f"Spread HAC T-Stat: "
    f"{hac_mean_t_stat(spread, max_lag=3):.3f}"
)

print()

print(
    f"Average Prediction-Beta "
    f"Correlation: "
    f"{beta_corr.mean():.2%}"
)


# =========================================
# SUBPERIOD ROBUSTNESS
# =========================================

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

    sub_ic = subset[
        "IC"
    ].dropna()


    sub_spread = subset[
        "Long Short Residual"
    ].dropna()


    sub_beta = subset[
        "Prediction Beta Correlation"
    ].dropna()


    print()
    print(
        "======================================"
    )
    print(name)
    print(
        "======================================"
    )


    print(
        f"Average IC: "
        f"{sub_ic.mean():.4f}"
    )

    print(
        f"Median IC: "
        f"{sub_ic.median():.4f}"
    )

    print(
        f"Positive IC: "
        f"{(sub_ic > 0).mean():.2%}"
    )

    print(
        f"IC HAC T: "
        f"{hac_mean_t_stat(sub_ic, max_lag=3):.3f}"
    )

    print(
        f"Long-Short: "
        f"{sub_spread.mean():.2%}"
    )

    print(
        f"Spread HAC T: "
        f"{hac_mean_t_stat(sub_spread, max_lag=3):.3f}"
    )

    print(
        f"Beta Correlation: "
        f"{sub_beta.mean():.2%}"
    )


# =========================================
# YEAR-BY-YEAR
# =========================================

print()
print(
    "======================================"
)
print(
    "YEAR-BY-YEAR"
)
print(
    "======================================"
)


year_rows = []


for year, group in monthly.groupby(
    "Year"
):

    year_ic = group[
        "IC"
    ].dropna()


    year_spread = group[
        "Long Short Residual"
    ].dropna()


    year_rows.append(
        {
            "Year":
                year,

            "Average IC":
                year_ic.mean(),

            "Positive IC Rate":
                (
                    year_ic > 0
                ).mean(),

            "Long Short":
                year_spread.mean(),

            "Prediction Beta Corr":
                group[
                    "Prediction Beta Correlation"
                ].mean(),
        }
    )


year_summary = pd.DataFrame(
    year_rows
)


year_display = (
    year_summary.copy()
)


year_display[
    "Average IC"
] = (
    year_display[
        "Average IC"
    ].round(3)
)


year_display[
    "Positive IC Rate"
] = (
    year_display[
        "Positive IC Rate"
    ]
    * 100
).round(1)


year_display[
    "Long Short"
] = (
    year_display[
        "Long Short"
    ]
    * 100
).round(2)


year_display[
    "Prediction Beta Corr"
] = (
    year_display[
        "Prediction Beta Corr"
    ]
    * 100
).round(2)


print(
    year_display.to_string(
        index=False
    )
)


# =========================================
# DECILE ATTRIBUTION
# =========================================

decile_rows = []


for period, group in predictions.groupby(
    prediction_period_column
):

    group = group.dropna(
        subset=[
            "V8B Prediction",
            "Beta Neutral Return",
            "Return",
            "Market Beta 24M",
        ]
    ).copy()


    if len(group) < 100:
        continue


    group[
        "Decile"
    ] = pd.qcut(
        group[
            "V8B Prediction"
        ].rank(
            method="first"
        ),
        q=10,
        labels=range(
            1,
            11,
        ),
    ).astype(int)


    for decile, bucket in group.groupby(
        "Decile"
    ):

        decile_rows.append(
            {
                "Period":
                    period,

                "Decile":
                    int(decile),

                "Beta Neutral Return":
                    bucket[
                        "Beta Neutral Return"
                    ].mean(),

                "Raw Return":
                    bucket[
                        "Return"
                    ].mean(),

                "Beta":
                    bucket[
                        "Market Beta 24M"
                    ].mean(),

                "Stocks":
                    len(bucket),
            }
        )


deciles = pd.DataFrame(
    decile_rows
)


summary_rows = []


for decile, group in deciles.groupby(
    "Decile"
):

    returns = group[
        "Beta Neutral Return"
    ]


    summary_rows.append(
        {
            "Decile":
                decile,

            "Beta Neutral Return":
                returns.mean(),

            "HAC T":
                hac_mean_t_stat(
                    returns,
                    max_lag=3,
                ),

            "Positive Rate":
                (
                    returns > 0
                ).mean(),

            "Raw Return":
                group[
                    "Raw Return"
                ].mean(),

            "Beta":
                group[
                    "Beta"
                ].mean(),
        }
    )


decile_summary = (
    pd.DataFrame(
        summary_rows
    )
    .sort_values(
        "Decile"
    )
)


display = (
    decile_summary.copy()
)


for column in [
    "Beta Neutral Return",
    "Positive Rate",
    "Raw Return",
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
    "HAC T"
] = (
    display[
        "HAC T"
    ].round(2)
)


display[
    "Beta"
] = (
    display[
        "Beta"
    ].round(2)
)


print()
print(
    "======================================"
)
print(
    "V8B DECILE ATTRIBUTION"
)
print(
    "1 = WORST, 10 = BEST"
)
print(
    "======================================"
)


print(
    display.to_string(
        index=False
    )
)


# =========================================
# SAVE
# =========================================

year_summary.to_csv(
    "data/cache/"
    "v8b_year_summary.csv",
    index=False,
)


decile_summary.to_csv(
    "data/cache/"
    "v8b_decile_summary.csv",
    index=False,
)


deciles.to_csv(
    "data/cache/"
    "v8b_decile_monthly.csv",
    index=False,
)