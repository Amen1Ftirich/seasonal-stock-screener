from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


PREDICTIONS_FILE = (
    "data/cache/v8c_predictions.parquet"
)

OUTPUT_FILE = (
    "data/cache/"
    "v8c_2022_top_decile_vs_spy.png"
)


# =========================================
# LOAD SAVED OOS V8C PREDICTIONS
# =========================================

data = pd.read_parquet(
    PREDICTIONS_FILE
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


# =========================================
# KEEP 2022 ONLY
# =========================================

data = data[
    data[
        period_column
    ].dt.year == 2022
].copy()


data = data.dropna(
    subset=[
        "Ticker",
        "V8C Prediction",
        "Return",
        "Benchmark Return",
    ]
)


monthly_rows = []

holding_rows = []


# =========================================
# MONTHLY TOP-DECILE PORTFOLIO
# =========================================

for period, cross_section in data.groupby(
    period_column
):

    cross_section = (
        cross_section
        .drop_duplicates(
            subset=["Ticker"]
        )
        .copy()
    )


    #
    # Convert V8C predictions into a
    # cross-sectional percentile.
    #
    # Higher prediction = higher percentile.
    #

    cross_section[
        "V8C Percentile"
    ] = (
        cross_section[
            "V8C Prediction"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )


    #
    # Top 10% of the monthly universe.
    #

    selected = cross_section[
        cross_section[
            "V8C Percentile"
        ] > 0.90
    ].copy()


    #
    # Equal-weight portfolio.
    #

    portfolio_return = float(
        selected[
            "Return"
        ].mean()
    )


    spy_return = float(
        cross_section[
            "Benchmark Return"
        ].iloc[0]
    )


    monthly_rows.append(
        {
            "Period":
                period,

            "Holdings":
                len(selected),

            "Portfolio Return":
                portfolio_return,

            "SPY Return":
                spy_return,

            "Excess Return":
                (
                    portfolio_return
                    - spy_return
                ),
        }
    )


    #
    # Save actual monthly holdings.
    #

    selected = (
        selected
        .sort_values(
            "V8C Prediction",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    for rank, row in selected.iterrows():

        holding_rows.append(
            {
                "Period":
                    period,

                "Rank":
                    rank + 1,

                "Ticker":
                    row[
                        "Ticker"
                    ],

                "V8C Prediction":
                    row[
                        "V8C Prediction"
                    ],

                "V8C Percentile":
                    row[
                        "V8C Percentile"
                    ],

                "Realized Return":
                    row[
                        "Return"
                    ],
            }
        )


# =========================================
# MONTHLY RESULTS
# =========================================

monthly = (
    pd.DataFrame(
        monthly_rows
    )
    .sort_values(
        "Period"
    )
    .reset_index(
        drop=True
    )
)


# =========================================
# GROWTH OF $100
# =========================================

monthly[
    "V8C Portfolio"
] = (
    100
    * (
        1
        + monthly[
            "Portfolio Return"
        ]
    ).cumprod()
)


monthly[
    "SPY"
] = (
    100
    * (
        1
        + monthly[
            "SPY Return"
        ]
    ).cumprod()
)


monthly[
    "Date"
] = (
    monthly[
        "Period"
    ].dt.to_timestamp()
)


# =========================================
# ADD TRUE $100 STARTING POINT
# =========================================

start_row = pd.DataFrame(
    {
        "Period": [
            pd.Period(
                "2021-12",
                freq="M",
            )
        ],

        "Holdings": [
            0
        ],

        "Portfolio Return": [
            0.0
        ],

        "SPY Return": [
            0.0
        ],

        "Excess Return": [
            0.0
        ],

        "V8C Portfolio": [
            100.0
        ],

        "SPY": [
            100.0
        ],

        "Date": [
            pd.Timestamp(
                "2021-12-31"
            )
        ],
    }
)


plot_data = pd.concat(
    [
        start_row,
        monthly,
    ],
    ignore_index=True,
)


# =========================================
# SUMMARY
# =========================================

portfolio_total_return = (
    monthly[
        "V8C Portfolio"
    ].iloc[-1]
    / 100
    - 1
)


spy_total_return = (
    monthly[
        "SPY"
    ].iloc[-1]
    / 100
    - 1
)


excess_total_return = (
    portfolio_total_return
    - spy_total_return
)


print()
print(
    "======================================"
)

print(
    "V8C 2022 TOP-DECILE PORTFOLIO"
)

print(
    "======================================"
)


print(
    f"Portfolio Return: "
    f"{portfolio_total_return:.2%}"
)


print(
    f"SPY Return: "
    f"{spy_total_return:.2%}"
)


print(
    f"Excess Return: "
    f"{excess_total_return:.2%}"
)


print(
    f"Average Holdings: "
    f"{monthly['Holdings'].mean():.1f}"
)


print()
print(
    "MONTHLY RESULTS"
)


display = monthly[
    [
        "Period",
        "Holdings",
        "Portfolio Return",
        "SPY Return",
        "Excess Return",
    ]
].copy()


for column in [
    "Portfolio Return",
    "SPY Return",
    "Excess Return",
]:

    display[
        column
    ] = (
        display[
            column
        ]
        * 100
    ).round(
        2
    )


print(
    display.to_string(
        index=False
    )
)


# =========================================
# PLOT
# =========================================

fig, ax = plt.subplots(
    figsize=(
        11,
        6,
    )
)


ax.plot(
    plot_data[
        "Date"
    ],

    plot_data[
        "V8C Portfolio"
    ],

    marker="o",

    label=(
        "V8C Top Decile"
    ),
)


ax.plot(
    plot_data[
        "Date"
    ],

    plot_data[
        "SPY"
    ],

    marker="o",

    label="SPY",
)


ax.axhline(
    100,
    linewidth=1,
)


ax.set_title(
    "V8C Top-Decile Portfolio vs SPY - 2022"
)


ax.set_xlabel(
    "Month"
)


ax.set_ylabel(
    "Growth of $100"
)


ax.legend()


ax.grid(
    alpha=0.25
)


fig.tight_layout()


fig.savefig(
    OUTPUT_FILE,
    dpi=160,
)


plt.show()


# =========================================
# SAVE RESULTS
# =========================================

monthly.to_csv(
    "data/cache/"
    "v8c_2022_top_decile_monthly.csv",
    index=False,
)


holdings = pd.DataFrame(
    holding_rows
)


holdings.to_csv(
    "data/cache/"
    "v8c_2022_top_decile_holdings.csv",
    index=False,
)


print()
print(
    f"Chart saved to: "
    f"{OUTPUT_FILE}"
)