from __future__ import annotations

import json

from pathlib import Path

import pandas as pd


SEC_CACHE = Path(
    "data/cache/sec"
)

STATUS_FILE = Path(
    "data/cache/"
    "sec_companyfacts_full_status.csv"
)


FLOW_ALIASES = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],

    "Net Income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],

    "Diluted Shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ],
}


INSTANT_ALIASES = {
    "Assets": [
        "Assets",
    ],

    "Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}


def concept_exists_with_quarter(
    companyfacts: dict,
    aliases: list[str],
    unit: str,
) -> bool:
    """
    Require at least one genuine approximately
    three-month 10-Q observation.
    """

    us_gaap = (
        companyfacts
        .get("facts", {})
        .get("us-gaap", {})
    )


    for tag in aliases:

        concept = us_gaap.get(
            tag
        )

        if concept is None:
            continue


        observations = (
            concept
            .get("units", {})
            .get(unit, [])
        )


        for observation in observations:

            if observation.get(
                "form"
            ) not in {
                "10-Q",
                "10-Q/A",
            }:
                continue


            start = observation.get(
                "start"
            )

            end = observation.get(
                "end"
            )

            filed = observation.get(
                "filed"
            )


            if (
                start is None
                or end is None
                or filed is None
            ):
                continue


            start = pd.to_datetime(
                start,
                errors="coerce",
            )

            end = pd.to_datetime(
                end,
                errors="coerce",
            )


            if (
                pd.isna(start)
                or pd.isna(end)
            ):
                continue


            days = (
                end
                - start
            ).days


            #
            # Genuine single-quarter duration.
            #

            if 60 <= days <= 120:
                return True


    return False


def instant_concept_exists(
    companyfacts: dict,
    aliases: list[str],
) -> bool:

    us_gaap = (
        companyfacts
        .get("facts", {})
        .get("us-gaap", {})
    )


    for tag in aliases:

        concept = us_gaap.get(
            tag
        )

        if concept is None:
            continue


        observations = (
            concept
            .get("units", {})
            .get("USD", [])
        )


        for observation in observations:

            if observation.get(
                "form"
            ) not in {
                "10-Q",
                "10-Q/A",
            }:
                continue


            if (
                observation.get("end")
                is not None
                and observation.get("filed")
                is not None
                and observation.get("val")
                is not None
            ):
                return True


    return False


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
    ].astype(int)
):

    file = (
        SEC_CACHE
        / f"companyfacts_{cik:010d}.json"
    )


    if not file.exists():
        continue


    with file.open(
        "r",
        encoding="utf-8",
    ) as handle:

        companyfacts = json.load(
            handle
        )


    row = {
        "CIK":
            cik,

        "Revenue":
            concept_exists_with_quarter(
                companyfacts,
                FLOW_ALIASES[
                    "Revenue"
                ],
                "USD",
            ),

        "Net Income":
            concept_exists_with_quarter(
                companyfacts,
                FLOW_ALIASES[
                    "Net Income"
                ],
                "USD",
            ),

        "Diluted Shares":
            concept_exists_with_quarter(
                companyfacts,
                FLOW_ALIASES[
                    "Diluted Shares"
                ],
                "shares",
            ),

        "Assets":
            instant_concept_exists(
                companyfacts,
                INSTANT_ALIASES[
                    "Assets"
                ],
            ),

        "Equity":
            instant_concept_exists(
                companyfacts,
                INSTANT_ALIASES[
                    "Equity"
                ],
            ),
    }


    rows.append(
        row
    )


audit = pd.DataFrame(
    rows
)


print()
print(
    "======================================"
)
print(
    "SEC QUARTERLY FUNDAMENTAL COVERAGE"
)
print(
    "======================================"
)


print(
    f"Companies: {len(audit)}"
)


for column in [
    "Revenue",
    "Net Income",
    "Diluted Shares",
    "Assets",
    "Equity",
]:

    count = int(
        audit[
            column
        ].sum()
    )

    print(
        f"{column:<20} "
        f"{count:>4} / "
        f"{len(audit):<4} "
        f"{count / len(audit):>7.2%}"
    )


all_required = (
    audit[
        [
            "Revenue",
            "Net Income",
            "Diluted Shares",
            "Assets",
            "Equity",
        ]
    ]
    .all(
        axis=1
    )
)


print()
print(
    f"All five: "
    f"{all_required.sum()} / "
    f"{len(audit)} "
    f"({all_required.mean():.2%})"
)


audit.to_csv(
    "data/cache/"
    "sec_quarterly_coverage.csv",
    index=False,
)