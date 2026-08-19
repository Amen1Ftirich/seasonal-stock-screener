from __future__ import annotations

import pandas as pd

from src.sec_data import (
    get_companyfacts_by_cik,
    normalize_ticker,
)

from src.universe import (
    get_sp500_historical_union,
)


RESOLUTION_FILE = (
    "data/cache/sec_identifier_resolution.csv"
)


print(
    "Loading 2006+ historical S&P universe..."
)


historical_tickers = {
    normalize_ticker(ticker)

    for ticker in get_sp500_historical_union(
        start_date="2006-01-01"
    )
}


resolution = pd.read_csv(
    RESOLUTION_FILE
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


universe = resolution[
    resolution[
        "Ticker"
    ].isin(
        historical_tickers
    )
].copy()


resolved = universe[
    universe[
        "CIK"
    ].notna()
].copy()


resolved[
    "CIK"
] = (
    resolved[
        "CIK"
    ].astype(int)
)


unique_ciks = (
    resolved[
        [
            "CIK",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "CIK"
    )
)


print()
print(
    "======================================"
)

print(
    "FULL HISTORICAL SEC DOWNLOAD"
)

print(
    "======================================"
)

print(
    f"Historical tickers: "
    f"{len(historical_tickers)}"
)

print(
    f"Resolved historical tickers: "
    f"{len(resolved)}"
)

print(
    f"Unique CIKs required: "
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
    "sec_companyfacts_full_status.csv",
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
    "FULL DOWNLOAD COMPLETE"
)

print(
    "======================================"
)

print(
    f"Success: {successes}"
)

print(
    f"Failed: "
    f"{len(status) - successes}"
)

print(
    f"Coverage: "
    f"{successes / len(status):.2%}"
)