from __future__ import annotations

import json

from pathlib import Path

import pandas as pd

from src.sec_fundamentals import (
    CONCEPT_ALIASES,
    audit_company_concepts,
)


STATUS_FILE = Path(
    "data/cache/"
    "sec_companyfacts_download_status.csv"
)


SEC_CACHE = Path(
    "data/cache/sec"
)


status = pd.read_csv(
    STATUS_FILE
)


status = status[
    status[
        "Success"
    ] == True
].copy()


rows = []


for cik in (
    status[
        "CIK"
    ]
    .astype(int)
):

    filename = (
        SEC_CACHE
        / (
            f"companyfacts_"
            f"{cik:010d}.json"
        )
    )


    if not filename.exists():
        continue


    with filename.open(
        "r",
        encoding="utf-8",
    ) as file:

        companyfacts = (
            json.load(
                file
            )
        )


    rows.append(
        audit_company_concepts(
            companyfacts
        )
    )


audit = pd.DataFrame(
    rows
)


audit.to_csv(
    "data/cache/"
    "sec_fundamental_concept_coverage.csv",
    index=False,
)


print()
print(
    "======================================"
)

print(
    "SEC FUNDAMENTAL CONCEPT COVERAGE"
)

print(
    "======================================"
)

print(
    f"Companies audited: "
    f"{len(audit)}"
)

print()


for concept in (
    CONCEPT_ALIASES
):

    column = (
        f"{concept} Available"
    )


    count = int(
        audit[
            column
        ].sum()
    )


    rate = (
        count
        / len(audit)
    )


    print(
        f"{concept:<24} "
        f"{count:>4} / "
        f"{len(audit):<4} "
        f"{rate:>7.2%}"
    )


#
# How much survives if we required
# different combinations?
#

core = [
    "Net Income Available",
    "Assets Available",
    "Equity Available",
]


quality = [
    "Net Income Available",
    "Operating Cash Flow Available",
    "Assets Available",
]


full = [
    f"{concept} Available"

    for concept in CONCEPT_ALIASES
]


print()
print(
    "======================================"
)

print(
    "COMBINATION COVERAGE"
)

print(
    "======================================"
)


for name, columns in [
    (
        "Core",
        core,
    ),

    (
        "Quality",
        quality,
    ),

    (
        "All proposed concepts",
        full,
    ),
]:

    valid = (
        audit[
            columns
        ]
        .all(
            axis=1
        )
    )


    print(
        f"{name:<24} "
        f"{valid.sum():>4} / "
        f"{len(audit):<4} "
        f"{valid.mean():>7.2%}"
    )