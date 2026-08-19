from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.sec_data import (
    get_companyfacts_by_cik,
    normalize_ticker,
)


PREDICTIONS_FILE = (
    Path(
        "data/cache/v8_predictions.parquet"
    )
)

RESOLUTION_FILE = (
    Path(
        "data/cache/sec_identifier_resolution.csv"
    )
)


predictions = pd.read_parquet(
    PREDICTIONS_FILE
)


resolution = pd.read_csv(
    RESOLUTION_FILE
)


predictions[
    "Ticker"
] = (
    predictions[
        "Ticker"
    ]
    .astype(str)
    .apply(
        normalize_ticker
    )
)


resolution[
    "Ticker"
] = (
    resolution[
        "Ticker"
    ]
    .astype(str)
    .apply(
        normalize_ticker
    )
)


#
# Only companies that actually appeared in
# our 2016-2025 OOS research universe.
#

tickers = sorted(
    predictions[
        "Ticker"
    ].unique()
)


universe = (
    pd.DataFrame(
        {
            "Ticker":
                tickers,
        }
    )
    .merge(
        resolution[
            [
                "Ticker",
                "CIK",
                "Resolution",
            ]
        ],
        on="Ticker",
        how="left",
    )
)


universe = universe[
    universe[
        "CIK"
    ].notna()
].copy()


universe[
    "CIK"
] = (
    universe[
        "CIK"
    ].astype(int)
)


#
# One CIK may theoretically have more than
# one historical ticker.
#
# Download the company only once.
#

unique_ciks = (
    universe[
        [
            "CIK",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "CIK"
    )
)


print(
    "======================================"
)

print(
    "SEC COMPANY FACTS DOWNLOAD"
)

print(
    "======================================"
)

print(
    f"OOS tickers: "
    f"{len(tickers)}"
)

print(
    f"Resolved tickers: "
    f"{len(universe)}"
)

print(
    f"Unique CIKs: "
    f"{len(unique_ciks)}"
)

print()


rows = []


total = len(
    unique_ciks
)


for number, row in enumerate(
    unique_ciks.itertuples(
        index=False
    ),
    start=1,
):

    cik = int(
        row.CIK
    )


    print(
        f"[{number}/{total}] "
        f"CIK {cik}"
    )


    try:

        facts = (
            get_companyfacts_by_cik(
                cik=cik,
            )
        )


        rows.append(
            {
                "CIK":
                    cik,

                "Success":
                    True,

                "Entity":
                    facts.get(
                        "entityName"
                    ),

                "Error":
                    None,
            }
        )


    except Exception as exc:

        print(
            f"    FAILED: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )


        rows.append(
            {
                "CIK":
                    cik,

                "Success":
                    False,

                "Entity":
                    None,

                "Error":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
            }
        )


status = pd.DataFrame(
    rows
)


status.to_csv(
    "data/cache/"
    "sec_companyfacts_download_status.csv",
    index=False,
)


successes = int(
    status[
        "Success"
    ].sum()
)


print()
print(
    "======================================"
)

print(
    "DOWNLOAD COMPLETE"
)

print(
    "======================================"
)

print(
    f"Success: "
    f"{successes}"
)

print(
    f"Failed: "
    f"{len(status) - successes}"
)

print(
    f"Coverage: "
    f"{successes / len(status):.2%}"
)