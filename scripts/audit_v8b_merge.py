from __future__ import annotations

import pandas as pd

from src.fundamental_features_v8b import (
    add_cross_sectional_fundamental_features,
    attach_point_in_time_fundamentals,
)

from src.sec_point_in_time import (
    FEATURE_COLUMNS_SEC,
)


print(
    "Loading OOS stock-month sample..."
)


stocks = pd.read_parquet(
    "data/cache/v8_predictions.parquet"
)


#
# The saved OOS file may call it Test Period.
#

if (
    "Period"
    not in stocks.columns
    and
    "Test Period"
    in stocks.columns
):

    stocks[
        "Period"
    ] = stocks[
        "Test Period"
    ]


resolution = pd.read_csv(
    "data/cache/"
    "sec_identifier_resolution.csv"
)


timeline = pd.read_parquet(
    "data/cache/"
    "sec_fundamental_timeline.parquet"
)


print(
    f"Stock-month rows: "
    f"{len(stocks):,}"
)


merged = (
    attach_point_in_time_fundamentals(
        data=stocks,
        resolution=resolution,
        timeline=timeline,
    )
)


print()
print(
    "======================================"
)

print(
    "V8B POINT-IN-TIME MERGE AUDIT"
)

print(
    "======================================"
)


print(
    f"Rows: "
    f"{len(merged):,}"
)


print(
    f"CIK coverage: "
    f"{merged['CIK'].notna().mean():.2%}"
)


print(
    f"Matched SEC filing: "
    f"{merged['Filed'].notna().mean():.2%}"
)


matched_age = merged[
    "Fundamental Age Days"
].dropna()


print(
    f"Median filing age: "
    f"{matched_age.median():.0f} days"
)


print(
    f"Average filing age: "
    f"{matched_age.mean():.0f} days"
)


print()
print(
    "Raw feature coverage:"
)


for column in FEATURE_COLUMNS_SEC:

    print(
        f"{column:<30} "
        f"{merged[column].notna().mean():>7.2%}"
    )


#
# -----------------------------------------
# LOOK-AHEAD AUDIT
# -----------------------------------------
#

month_start = (
    merged[
        "Period"
    ].dt.start_time
)


violations = merged[
    merged[
        "Filed"
    ].notna()
    &
    (
        merged[
            "Filed"
        ]
        >= month_start
    )
]


print()
print(
    f"Look-ahead violations: "
    f"{len(violations)}"
)


#
# -----------------------------------------
# ADD CROSS-SECTIONAL MODEL FEATURES
# -----------------------------------------
#

processed = (
    add_cross_sectional_fundamental_features(
        merged
    )
)


cs_columns = [
    f"CS {column}"
    for column in FEATURE_COLUMNS_SEC
]


print()
print(
    "Cross-sectional feature means:"
)


for column in cs_columns:

    print(
        f"{column:<33} "
        f"{processed[column].mean():>8.4f}"
    )


#
# -----------------------------------------
# APPLE SANITY CHECK
# -----------------------------------------
#

apple = processed[
    processed[
        "Ticker"
    ] == "AAPL"
].copy()


if not apple.empty:

    apple[
        "Period"
    ] = pd.PeriodIndex(
        apple[
            "Period"
        ],
        freq="M",
    )


    sample_periods = [
        pd.Period(
            "2021-01",
            freq="M",
        ),
        pd.Period(
            "2021-11",
            freq="M",
        ),
        pd.Period(
            "2022-11",
            freq="M",
        ),
        pd.Period(
            "2024-11",
            freq="M",
        ),
    ]


    apple = apple[
        apple[
            "Period"
        ].isin(
            sample_periods
        )
    ]


    print()
    print(
        "Apple PIT sanity check:"
    )


    print(
        apple[
            [
                "Period",
                "Filed",
                "Fiscal End",
                "Revenue Growth YoY",
                "ROA",
                "CFO To Assets",
                "Asset Growth YoY",
            ]
        ]
        .sort_values(
            "Period"
        )
        .to_string(
            index=False
        )
    )


processed.to_parquet(
    "data/cache/"
    "v8b_merge_audit.parquet",
    index=False,
)