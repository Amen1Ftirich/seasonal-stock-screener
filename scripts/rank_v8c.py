from __future__ import annotations

import pandas as pd


PREDICTIONS_FILE = (
    "data/cache/v8c_predictions.parquet"
)


predictions = pd.read_parquet(
    PREDICTIONS_FILE
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


latest_period = (
    predictions[
        period_column
    ].max()
)


latest = predictions[
    predictions[
        period_column
    ] == latest_period
].copy()


latest = latest.dropna(
    subset=[
        "Ticker",
        "V8C Prediction",
    ]
)


latest[
    "V8C Rank"
] = (
    latest[
        "V8C Prediction"
    ]
    .rank(
        ascending=False,
        method="first",
    )
    .astype(int)
)


latest[
    "V8C Percentile"
] = (
    latest[
        "V8C Prediction"
    ]
    .rank(
        pct=True,
        method="average",
    )
)


latest = latest.sort_values(
    "V8C Rank"
)


columns = [
    "V8C Rank",
    "Ticker",
    "V8C Prediction",
    "V8C Percentile",
]


optional_columns = [
    "Market Beta 24M",
    "Return",
    "Beta Neutral Return",
]


for column in optional_columns:

    if column in latest.columns:

        columns.append(
            column
        )


ranking = latest[
    columns
].copy()


ranking.to_csv(
    "data/cache/"
    "v8c_latest_ranking.csv",
    index=False,
)


print()
print(
    "======================================"
)
print(
    "V8C STOCK RANKING"
)
print(
    "======================================"
)

print(
    f"Period: {latest_period}"
)

print(
    f"Stocks ranked: {len(ranking)}"
)


print()
print(
    "TOP 25"
)
print(
    "--------------------------------------"
)


print(
    ranking
    .head(25)
    .to_string(
        index=False
    )
)


print()
print(
    "BOTTOM 25"
)
print(
    "--------------------------------------"
)


print(
    ranking
    .tail(25)
    .to_string(
        index=False
    )
)