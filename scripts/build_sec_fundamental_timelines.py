from __future__ import annotations

import json

from pathlib import Path

import pandas as pd

from src.sec_point_in_time import (
    FEATURE_COLUMNS_SEC,
    build_company_fundamental_timeline,
)


STATUS_FILE = Path(
    "data/cache/"
    "sec_companyfacts_full_status.csv"
)


SEC_CACHE = Path(
    "data/cache/sec"
)


OUTPUT_FILE = Path(
    "data/cache/"
    "sec_fundamental_timeline.parquet"
)


status = pd.read_csv(
    STATUS_FILE
)


status = status[
    status[
        "Success"
    ] == True
].copy()


frames = []


total = len(
    status
)


for number, row in enumerate(
    status.itertuples(
        index=False
    ),
    start=1,
):

    cik = int(
        row.CIK
    )


    if (
        number == 1
        or number % 50 == 0
        or number == total
    ):

        print(
            f"[{number}/{total}] "
            f"Building CIK {cik}"
        )


    filename = (
        SEC_CACHE
        / (
            f"companyfacts_"
            f"{cik:010d}.json"
        )
    )


    if not filename.exists():
        continue


    try:

        with filename.open(
            "r",
            encoding="utf-8",
        ) as file:

            companyfacts = (
                json.load(
                    file
                )
            )


        timeline = (
            build_company_fundamental_timeline(
                companyfacts
            )
        )


        if not timeline.empty:

            frames.append(
                timeline
            )


    except Exception as exc:

        print(
            f"FAILED CIK {cik}: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )


if not frames:

    raise RuntimeError(
        "No fundamental timelines built"
    )


data = pd.concat(
    frames,
    ignore_index=True,
)


data.to_parquet(
    OUTPUT_FILE,
    index=False,
)


print()
print(
    "======================================"
)

print(
    "POINT-IN-TIME FUNDAMENTAL DATASET"
)

print(
    "======================================"
)

print(
    f"Rows: "
    f"{len(data):,}"
)

print(
    f"Companies: "
    f"{data['CIK'].nunique():,}"
)

print(
    f"First filing: "
    f"{data['Filed'].min()}"
)

print(
    f"Last filing: "
    f"{data['Filed'].max()}"
)

print()


print(
    "Feature availability:"
)


for column in FEATURE_COLUMNS_SEC:

    count = int(
        data[
            column
        ].notna().sum()
    )

    rate = (
        data[
            column
        ].notna().mean()
    )


    print(
        f"{column:<30} "
        f"{count:>6,} "
        f"{rate:>7.2%}"
    )


print()
print(
    "Latest Apple snapshots:"
)


apple = data[
    data[
        "CIK"
    ] == 320193
]


if not apple.empty:

    columns = [
        "Filed",
        "Fiscal End",
        "Revenue Growth YoY",
        "ROA",
        "CFO To Assets",
        "Accruals To Assets",
        "Asset Growth YoY",
        "Diluted Share Growth YoY",
    ]


    print(
        apple[
            columns
        ]
        .tail(5)
        .to_string(
            index=False
        )
    )