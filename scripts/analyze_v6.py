import pandas as pd

from src.backtest_v3 import (
    hac_mean_t_stat,
)


monthly = pd.read_csv(
    "data/cache/v6_monthly.csv",
)


monthly["Period"] = pd.PeriodIndex(
    monthly["Period"],
    freq="M",
)


def print_series_stats(
    name: str,
    values: pd.Series,
):

    values = values.dropna()

    print()
    print(name)

    print(
        f"  Mean:          "
        f"{values.mean():.2%}"
    )

    print(
        f"  Median:        "
        f"{values.median():.2%}"
    )

    print(
        f"  Positive Rate: "
        f"{(values > 0).mean():.2%}"
    )

    print(
        f"  HAC T-Stat:    "
        f"{hac_mean_t_stat(values, max_lag=3):.3f}"
    )


print(
    "======================================"
)

print(
    "V6 ROBUSTNESS ANALYSIS"
)

print(
    "======================================"
)


#
# Full-period tests
#

print_series_stats(
    "INFORMATION COEFFICIENT",
    monthly["IC"],
)


print_series_stats(
    "TOP DECILE RESIDUAL",
    monthly["Top Decile Residual"],
)


print_series_stats(
    "BOTTOM DECILE RESIDUAL",
    monthly["Bottom Decile Residual"],
)


print_series_stats(
    "BETA-HEDGED TOP-BOTTOM",
    monthly["Long Short Residual"],
)


#
# ------------------------------------
# SUBPERIOD STABILITY
# ------------------------------------
#

first_half = monthly[
    monthly["Period"]
    <= pd.Period(
        "2020-12",
        freq="M",
    )
]


second_half = monthly[
    monthly["Period"]
    >= pd.Period(
        "2021-01",
        freq="M",
    )
]


for name, data in [
    (
        "2016-2020",
        first_half,
    ),
    (
        "2021-2025",
        second_half,
    ),
]:

    print()
    print(
        "======================================"
    )

    print(
        name
    )

    print(
        "======================================"
    )

    print(
        f"Months: "
        f"{len(data)}"
    )

    print(
        f"Average IC: "
        f"{data['IC'].mean():.3f}"
    )

    print(
        f"Median IC: "
        f"{data['IC'].median():.3f}"
    )

    print(
        f"Positive IC: "
        f"{(data['IC'] > 0).mean():.2%}"
    )

    print(
        f"IC HAC T-Stat: "
        f"{hac_mean_t_stat(data['IC'], max_lag=3):.3f}"
    )

    print(
        f"Top Residual: "
        f"{data['Top Decile Residual'].mean():.2%}"
    )

    print(
        f"Bottom Residual: "
        f"{data['Bottom Decile Residual'].mean():.2%}"
    )

    print(
        f"Residual Spread: "
        f"{data['Long Short Residual'].mean():.2%}"
    )


#
# ------------------------------------
# CALENDAR-YEAR DIAGNOSTIC
# ------------------------------------
#

yearly = (
    monthly
    .assign(
        Year=monthly[
            "Period"
        ].dt.year
    )
    .groupby(
        "Year"
    )
    .agg(
        Average_IC=(
            "IC",
            "mean",
        ),

        Residual_Spread=(
            "Long Short Residual",
            "mean",
        ),

        Top_Residual=(
            "Top Decile Residual",
            "mean",
        ),

        Bottom_Residual=(
            "Bottom Decile Residual",
            "mean",
        ),
    )
)


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


display = yearly.copy()

display[
    "Average_IC"
] = (
    display[
        "Average_IC"
    ]
    .round(3)
)


for column in [
    "Residual_Spread",
    "Top_Residual",
    "Bottom_Residual",
]:

    display[column] = (
        display[column]
        * 100
    ).round(2)


print(
    display.to_string()
)