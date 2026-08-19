from __future__ import annotations

import pandas as pd

from src.fundamental_features_v8c import (
    add_cross_sectional_quarterly_features,
    attach_point_in_time_quarterly,
)

from src.sec_quarterly import (
    FEATURE_COLUMNS_SEC_QUARTERLY,
)


stocks = pd.read_parquet(
    "data/cache/v8_predictions.parquet"
)


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
    "sec_quarterly_timeline.parquet"
)


merged = (
    attach_point_in_time_quarterly(
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
    "V8C QUARTERLY PIT MERGE AUDIT"
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
    f"Matched quarterly filing: "
    f"{merged['Filed'].notna().mean():.2%}"
)


age = merged[
    "Quarterly Fundamental Age Days"
].dropna()


print(
    f"Median filing age: "
    f"{age.median():.0f} days"
)


print(
    f"Average filing age: "
    f"{age.mean():.0f} days"
)


print()
print(
    "Feature coverage:"
)


for column in (
    FEATURE_COLUMNS_SEC_QUARTERLY
):

    print(
        f"{column:<40} "
        f"{merged[column].notna().mean():>7.2%}"
    )


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


processed = (
    add_cross_sectional_quarterly_features(
        merged
    )
)


print()
print(
    "Cross-sectional means:"
)


for column in (
    FEATURE_COLUMNS_SEC_QUARTERLY
):

    cs = (
        f"CS {column}"
    )

    print(
        f"{cs:<43} "
        f"{processed[cs].mean():>8.4f}"
    )


#
# Apple timing sanity check.
#

apple = processed[
    processed[
        "Ticker"
    ] == "AAPL"
].copy()


if not apple.empty:

    print()
    print(
        "Apple quarterly PIT check:"
    )


    apple[
        "Period"
    ] = pd.PeriodIndex(
        apple[
            "Period"
        ],
        freq="M",
    )


    periods = [
        pd.Period(
            "2024-05",
            freq="M",
        ),
        pd.Period(
            "2024-06",
            freq="M",
        ),
        pd.Period(
            "2024-11",
            freq="M",
        ),
        pd.Period(
            "2024-12",
            freq="M",
        ),
        pd.Period(
            "2025-02",
            freq="M",
        ),
    ]


    print(
        apple[
            apple[
                "Period"
            ].isin(
                periods
            )
        ][
            [
                "Period",
                "Filed",
                "Quarter End",
                "Quarterly Revenue Growth YoY",
                "Quarterly Net Margin",
                "Quarterly Asset Growth YoY",
                "Quarterly Diluted Share Growth YoY",
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
    "v8c_merge_audit.parquet",
    index=False,
)