from __future__ import annotations

import pandas as pd


#
# Canonical fundamental concepts and the
# US-GAAP tags that may represent them.
#
# Earlier aliases are preferred.
#

CONCEPT_ALIASES = {
    "Revenue": [
        (
            "us-gaap",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
        ),
        (
            "us-gaap",
            "Revenues",
        ),
        (
            "us-gaap",
            "SalesRevenueNet",
        ),
    ],

    "Operating Income": [
        (
            "us-gaap",
            "OperatingIncomeLoss",
        ),
    ],

    "Net Income": [
        (
            "us-gaap",
            "NetIncomeLoss",
        ),
        (
            "us-gaap",
            "ProfitLoss",
        ),
    ],

    "Operating Cash Flow": [
        (
            "us-gaap",
            "NetCashProvidedByUsedInOperatingActivities",
        ),
        (
            "us-gaap",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
    ],

    "Assets": [
        (
            "us-gaap",
            "Assets",
        ),
    ],

    "Liabilities": [
        (
            "us-gaap",
            "Liabilities",
        ),
    ],

    "Equity": [
        (
            "us-gaap",
            "StockholdersEquity",
        ),
        (
            "us-gaap",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    ],

    "Diluted Shares": [
        (
            "us-gaap",
            "WeightedAverageNumberOfDilutedSharesOutstanding",
        ),
        (
            "us-gaap",
            "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        ),
    ],
}


def get_concept(
    companyfacts: dict,
    aliases: list[
        tuple[str, str]
    ],
) -> tuple[
    str | None,
    dict | None,
]:
    """
    Find the first available XBRL concept from
    an ordered alias list.

    Returns:

        selected tag name,
        full concept dictionary
    """

    facts = companyfacts.get(
        "facts",
        {}
    )

    for namespace, tag in aliases:

        namespace_data = facts.get(
            namespace,
            {}
        )

        concept = namespace_data.get(
            tag
        )

        if concept is not None:

            return (
                tag,
                concept,
            )

    return (
        None,
        None,
    )


def concept_has_annual_10k(
    concept: dict | None,
) -> bool:
    """
    Check whether a concept contains at least
    one usable annual 10-K observation.
    """

    if concept is None:
        return False

    units = concept.get(
        "units",
        {}
    )

    for observations in units.values():

        for observation in observations:

            if (
                observation.get(
                    "form"
                ) == "10-K"
                and
                observation.get(
                    "val"
                ) is not None
                and
                observation.get(
                    "filed"
                ) is not None
                and
                observation.get(
                    "end"
                ) is not None
            ):

                return True

    return False


def audit_company_concepts(
    companyfacts: dict,
) -> dict:
    """
    Return availability information for our
    proposed fundamental feature inputs.
    """

    row = {
        "Entity":
            companyfacts.get(
                "entityName"
            ),

        "CIK":
            companyfacts.get(
                "cik"
            ),
    }

    for (
        canonical_name,
        aliases,
    ) in CONCEPT_ALIASES.items():

        tag, concept = get_concept(
            companyfacts=companyfacts,
            aliases=aliases,
        )

        row[
            f"{canonical_name} Available"
        ] = concept_has_annual_10k(
            concept
        )

        row[
            f"{canonical_name} Tag"
        ] = tag

    return row