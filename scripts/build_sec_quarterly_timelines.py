from __future__ import annotations

import json

from pathlib import Path

import pandas as pd

from src.sec_quarterly import (
    FEATURE_COLUMNS_SEC_QUARTERLY,
    build_company_quarterly_timeline,
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
    "sec_quarterly_timeline.parquet"
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
            f"Building quarterly CIK {cik}"
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
            build_company_quarterly_timeline(
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
        "No quarterly timelines built"
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
    "POINT-IN-TIME QUARTERLY DATASET"
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


for column in (
    FEATURE_COLUMNS_SEC_QUARTERLY
):

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
        f"{column:<38} "
        f"{count:>7,} "
        f"{rate:>7.2%}"
    )


print()
print(
    "Latest Apple quarterly snapshots:"
)


apple = data[
    data[
        "CIK"
    ] == 320193
]


if not apple.empty:

    columns = [
        "Filed",
        "Quarter End",
        "Quarterly Revenue Growth YoY",
        "Quarterly Net Margin",
        "Quarterly Net Margin Change YoY",
        "Quarterly Net Income To Assets",
        "Quarterly Asset Growth YoY",
        "Quarterly Diluted Share Growth YoY",
    ]


    print(
        apple[
            columns
        ]
        .tail(10)
        .to_string(
            index=False
        )
    )