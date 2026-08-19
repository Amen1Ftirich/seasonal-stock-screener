from __future__ import annotations

import numpy as np
import pandas as pd

from src.sec_fundamentals import (
    CONCEPT_ALIASES,
)


FLOW_CONCEPTS = {
    "Revenue",
    "Net Income",
    "Operating Cash Flow",
    "Diluted Shares",
}


BALANCE_CONCEPTS = {
    "Assets",
    "Equity",
}


FEATURE_COLUMNS_SEC = [
    "Revenue Growth YoY",
    "Net Margin",
    "ROA",
    "CFO To Assets",
    "Accruals To Assets",
    "Asset Growth YoY",
    "Equity To Assets",
    "Diluted Share Growth YoY",
]


def _preferred_unit(
    concept_name: str,
) -> str:

    if concept_name == "Diluted Shares":
        return "shares"

    return "USD"


def collect_concept_observations(
    companyfacts: dict,
    concept_name: str,
) -> pd.DataFrame:
    """
    Collect every usable annual 10-K observation
    across all aliases for one canonical concept.

    We intentionally do NOT select one XBRL tag
    globally because companies often switch tags
    through time.
    """

    aliases = CONCEPT_ALIASES[
        concept_name
    ]

    preferred_unit = (
        _preferred_unit(
            concept_name
        )
    )

    facts = companyfacts.get(
        "facts",
        {}
    )

    rows = []


    for priority, (
        namespace,
        tag,
    ) in enumerate(
        aliases
    ):

        concept = (
            facts
            .get(
                namespace,
                {}
            )
            .get(
                tag
            )
        )


        if concept is None:
            continue


        units = concept.get(
            "units",
            {}
        )


        observations = units.get(
            preferred_unit,
            []
        )


        for observation in observations:

            if observation.get(
                "form"
            ) not in {
                "10-K",
                "10-K/A",
            }:
                continue


            if observation.get(
                "val"
            ) is None:
                continue


            if observation.get(
                "filed"
            ) is None:
                continue


            if observation.get(
                "end"
            ) is None:
                continue


            filed = pd.to_datetime(
                observation[
                    "filed"
                ],
                errors="coerce",
            )

            end = pd.to_datetime(
                observation[
                    "end"
                ],
                errors="coerce",
            )


            if (
                pd.isna(filed)
                or pd.isna(end)
            ):
                continue


            start = observation.get(
                "start"
            )


            if concept_name in FLOW_CONCEPTS:

                if start is None:
                    continue


                start = pd.to_datetime(
                    start,
                    errors="coerce",
                )


                if pd.isna(start):
                    continue


                duration_days = (
                    end
                    - start
                ).days


                #
                # Keep genuine annual periods.
                #
                # Allows ordinary 52/53-week years
                # while removing quarterly values.
                #

                if not (
                    300
                    <= duration_days
                    <= 400
                ):
                    continue


            else:

                start = pd.NaT


            rows.append(
                {
                    "Concept":
                        concept_name,

                    "Tag":
                        tag,

                    "Alias Priority":
                        priority,

                    "Value":
                        float(
                            observation[
                                "val"
                            ]
                        ),

                    "Start":
                        start,

                    "End":
                        end,

                    "Filed":
                        filed,

                    "Accession":
                        observation.get(
                            "accn"
                        ),

                    "Form":
                        observation.get(
                            "form"
                        ),
                }
            )


    if not rows:

        return pd.DataFrame(
            columns=[
                "Concept",
                "Tag",
                "Alias Priority",
                "Value",
                "Start",
                "End",
                "Filed",
                "Accession",
                "Form",
            ]
        )


    return (
        pd.DataFrame(
            rows
        )
        .drop_duplicates()
        .sort_values(
            [
                "End",
                "Filed",
                "Alias Priority",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def _value_for_period_as_of(
    observations: pd.DataFrame,
    fiscal_end: pd.Timestamp,
    as_of: pd.Timestamp,
) -> float:

    if observations.empty:
        return np.nan


    available = observations[
        (
            observations[
                "End"
            ] == fiscal_end
        )
        &
        (
            observations[
                "Filed"
            ] <= as_of
        )
    ].copy()


    if available.empty:
        return np.nan


    #
    # Use the most recently filed value that
    # was actually known by this date.
    #

    latest_filing = (
        available[
            "Filed"
        ].max()
    )


    latest = available[
        available[
            "Filed"
        ] == latest_filing
    ].copy()


    #
    # If multiple aliases represent the same
    # concept on that filing, prefer our
    # highest-priority standardized tag.
    #

    latest = latest.sort_values(
        "Alias Priority"
    )


    return float(
        latest.iloc[0][
            "Value"
        ]
    )


def _safe_growth(
    current: float,
    previous: float,
) -> float:

    if (
        pd.isna(current)
        or pd.isna(previous)
        or previous == 0
    ):
        return np.nan


    return (
        current
        / previous
        - 1.0
    )


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:

    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan


    return (
        numerator
        / denominator
    )


def build_company_fundamental_timeline(
    companyfacts: dict,
) -> pd.DataFrame:
    """
    Construct a point-in-time annual fundamental
    timeline for one company.

    A row becomes available on its SEC filing date.
    """

    concepts_needed = [
        "Revenue",
        "Net Income",
        "Operating Cash Flow",
        "Assets",
        "Equity",
        "Diluted Shares",
    ]


    observations = {
        concept:
            collect_concept_observations(
                companyfacts=companyfacts,
                concept_name=concept,
            )

        for concept
        in concepts_needed
    }


    assets = observations[
        "Assets"
    ]


    if assets.empty:

        return pd.DataFrame()


    #
    # A feature snapshot can change whenever a
    # relevant 10-K or 10-K/A is filed.
    #

    filing_dates = sorted(
        {
            filed

            for table in (
                observations.values()
            )

            if not table.empty

            for filed in table[
                "Filed"
            ].dropna()
        }
    )


    rows = []


    for as_of in filing_dates:

        known_assets = assets[
            assets[
                "Filed"
            ] <= as_of
        ]


        if known_assets.empty:
            continue


        fiscal_ends = sorted(
            known_assets[
                "End"
            ]
            .dropna()
            .unique()
        )


        if len(
            fiscal_ends
        ) < 2:
            continue


        current_end = pd.Timestamp(
            fiscal_ends[-1]
        )

        previous_end = pd.Timestamp(
            fiscal_ends[-2]
        )


        #
        # Prevent strange historical/stub periods
        # from becoming a fake year-over-year pair.
        #

        year_gap = (
            current_end
            - previous_end
        ).days


        if not (
            300
            <= year_gap
            <= 430
        ):
            continue


        values = {}


        for concept in concepts_needed:

            table = observations[
                concept
            ]


            values[
                f"{concept} Current"
            ] = (
                _value_for_period_as_of(
                    observations=table,
                    fiscal_end=current_end,
                    as_of=as_of,
                )
            )


            values[
                f"{concept} Previous"
            ] = (
                _value_for_period_as_of(
                    observations=table,
                    fiscal_end=previous_end,
                    as_of=as_of,
                )
            )


        assets_current = values[
            "Assets Current"
        ]

        assets_previous = values[
            "Assets Previous"
        ]


        if (
            pd.notna(
                assets_current
            )
            and
            pd.notna(
                assets_previous
            )
        ):

            average_assets = (
                assets_current
                + assets_previous
            ) / 2.0

        else:

            average_assets = np.nan


        revenue_current = values[
            "Revenue Current"
        ]

        revenue_previous = values[
            "Revenue Previous"
        ]

        net_income = values[
            "Net Income Current"
        ]

        cash_flow = values[
            "Operating Cash Flow Current"
        ]

        equity = values[
            "Equity Current"
        ]

        shares_current = values[
            "Diluted Shares Current"
        ]

        shares_previous = values[
            "Diluted Shares Previous"
        ]


        rows.append(
            {
                "CIK":
                    int(
                        companyfacts[
                            "cik"
                        ]
                    ),

                "Entity":
                    companyfacts.get(
                        "entityName"
                    ),

                "Filed":
                    as_of,

                "Fiscal End":
                    current_end,

                "Revenue Growth YoY":
                    _safe_growth(
                        revenue_current,
                        revenue_previous,
                    ),

                "Net Margin":
                    _safe_ratio(
                        net_income,
                        revenue_current,
                    ),

                "ROA":
                    _safe_ratio(
                        net_income,
                        average_assets,
                    ),

                "CFO To Assets":
                    _safe_ratio(
                        cash_flow,
                        average_assets,
                    ),

                "Accruals To Assets":
                    _safe_ratio(
                        (
                            net_income
                            - cash_flow
                        )
                        if (
                            pd.notna(
                                net_income
                            )
                            and
                            pd.notna(
                                cash_flow
                            )
                        )
                        else np.nan,
                        average_assets,
                    ),

                "Asset Growth YoY":
                    _safe_growth(
                        assets_current,
                        assets_previous,
                    ),

                "Equity To Assets":
                    _safe_ratio(
                        equity,
                        assets_current,
                    ),

                "Diluted Share Growth YoY":
                    _safe_growth(
                        shares_current,
                        shares_previous,
                    ),
            }
        )


    if not rows:
        return pd.DataFrame()


    result = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "Filed"
        )
    )


    #
    # Multiple facts from the same filing date can
    # generate the same snapshot. Keep one.
    #

    result = (
        result
        .drop_duplicates(
            subset=[
                "CIK",
                "Filed",
                "Fiscal End",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


    return result