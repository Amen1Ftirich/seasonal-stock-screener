from __future__ import annotations

import numpy as np
import pandas as pd

from src.sec_fundamentals import (
    CONCEPT_ALIASES,
)


QUARTERLY_FLOW_CONCEPTS = [
    "Revenue",
    "Net Income",
    "Diluted Shares",
]


QUARTERLY_INSTANT_CONCEPTS = [
    "Assets",
    "Equity",
]


FEATURE_COLUMNS_SEC_QUARTERLY = [
    "Quarterly Revenue Growth YoY",
    "Quarterly Net Margin",
    "Quarterly Net Margin Change YoY",
    "Quarterly Net Income To Assets",
    "Quarterly Asset Growth YoY",
    "Quarterly Equity To Assets",
    "Quarterly Diluted Share Growth YoY",
]


def _preferred_unit(
    concept_name: str,
) -> str:

    if concept_name == "Diluted Shares":
        return "shares"

    return "USD"


def _parse_date(
    value,
) -> pd.Timestamp:

    return pd.to_datetime(
        value,
        errors="coerce",
    )


def collect_flow_observations(
    companyfacts: dict,
    concept_name: str,
) -> pd.DataFrame:
    """
    Collect quarterly, 9-month YTD and annual
    observations for one flow concept.

    We keep all filing dates because a later filing
    can legitimately update a previously reported
    value.
    """

    aliases = CONCEPT_ALIASES[
        concept_name
    ]

    unit = _preferred_unit(
        concept_name
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


        observations = (
            concept
            .get(
                "units",
                {}
            )
            .get(
                unit,
                []
            )
        )


        for observation in observations:

            form = observation.get(
                "form"
            )


            if form not in {
                "10-Q",
                "10-Q/A",
                "10-K",
                "10-K/A",
            }:
                continue


            if (
                observation.get("val")
                is None
                or observation.get("start")
                is None
                or observation.get("end")
                is None
                or observation.get("filed")
                is None
            ):
                continue


            start = _parse_date(
                observation[
                    "start"
                ]
            )

            end = _parse_date(
                observation[
                    "end"
                ]
            )

            filed = _parse_date(
                observation[
                    "filed"
                ]
            )


            if (
                pd.isna(start)
                or pd.isna(end)
                or pd.isna(filed)
            ):
                continue


            duration = (
                end - start
            ).days


            period_type = None


            #
            # Genuine single-quarter value.
            #

            if 60 <= duration <= 120:

                period_type = (
                    "quarter"
                )


            #
            # Roughly nine months.
            #

            elif 200 <= duration <= 300:

                period_type = (
                    "nine_month"
                )


            #
            # Full fiscal year.
            #

            elif 300 <= duration <= 400:

                period_type = (
                    "annual"
                )


            if period_type is None:
                continue


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

                    "Form":
                        form,

                    "Period Type":
                        period_type,

                    "Accession":
                        observation.get(
                            "accn"
                        ),
                }
            )


    if not rows:

        return pd.DataFrame()


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


def collect_instant_observations(
    companyfacts: dict,
    concept_name: str,
) -> pd.DataFrame:
    """
    Collect point-in-time balance-sheet observations
    from 10-Q and 10-K filings.
    """

    aliases = CONCEPT_ALIASES[
        concept_name
    ]

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


        observations = (
            concept
            .get(
                "units",
                {}
            )
            .get(
                "USD",
                []
            )
        )


        for observation in observations:

            if observation.get(
                "form"
            ) not in {
                "10-Q",
                "10-Q/A",
                "10-K",
                "10-K/A",
            }:
                continue


            if (
                observation.get("val")
                is None
                or observation.get("end")
                is None
                or observation.get("filed")
                is None
            ):
                continue


            end = _parse_date(
                observation[
                    "end"
                ]
            )

            filed = _parse_date(
                observation[
                    "filed"
                ]
            )


            if (
                pd.isna(end)
                or pd.isna(filed)
            ):
                continue


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

                    "End":
                        end,

                    "Filed":
                        filed,

                    "Form":
                        observation.get(
                            "form"
                        ),

                    "Accession":
                        observation.get(
                            "accn"
                        ),
                }
            )


    if not rows:
        return pd.DataFrame()


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


def build_single_quarter_observations(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce genuine single-quarter values.

    Q1-Q3 usually come directly from 10-Q filings.

    When a 10-K does not provide an explicit Q4
    observation, derive:

        Q4 = FY - first nine months

    using only the 9M observation available by the
    annual filing date.
    """

    if observations.empty:
        return pd.DataFrame()


    direct = observations[
        observations[
            "Period Type"
        ] == "quarter"
    ].copy()


    direct[
        "Quarter Source"
    ] = "direct"


    direct[
        "Source Priority"
    ] = 0


    derived_rows = []


    annual_rows = observations[
        observations[
            "Period Type"
        ] == "annual"
    ]


    nine_month_rows = observations[
        observations[
            "Period Type"
        ] == "nine_month"
    ]


    for _, annual in (
        annual_rows.iterrows()
    ):

        candidates = nine_month_rows[
            nine_month_rows[
                "Filed"
            ] <= annual[
                "Filed"
            ]
        ].copy()


        if candidates.empty:
            continue


        #
        # The 9M and FY observations should start
        # at approximately the same fiscal-year
        # beginning.
        #

        candidates[
            "Start Difference"
        ] = (
            candidates[
                "Start"
            ]
            - annual[
                "Start"
            ]
        ).abs().dt.days


        candidates[
            "End Difference"
        ] = (
            annual[
                "End"
            ]
            - candidates[
                "End"
            ]
        ).dt.days


        candidates = candidates[
            (
                candidates[
                    "Start Difference"
                ] <= 15
            )
            &
            (
                candidates[
                    "End Difference"
                ].between(
                    60,
                    120,
                )
            )
        ]


        if candidates.empty:
            continue


        #
        # Use the latest 9M value that was actually
        # known when the annual report was filed.
        #

        candidates = candidates.sort_values(
            [
                "Filed",
                "Alias Priority",
            ],
            ascending=[
                False,
                True,
            ],
        )


        nine_month = candidates.iloc[
            0
        ]


        q4_value = (
            annual[
                "Value"
            ]
            - nine_month[
                "Value"
            ]
        )


        derived_rows.append(
            {
                "Concept":
                    annual[
                        "Concept"
                    ],

                "Tag":
                    annual[
                        "Tag"
                    ],

                "Alias Priority":
                    annual[
                        "Alias Priority"
                    ],

                "Value":
                    float(
                        q4_value
                    ),

                "Start":
                    nine_month[
                        "End"
                    ],

                "End":
                    annual[
                        "End"
                    ],

                "Filed":
                    annual[
                        "Filed"
                    ],

                "Form":
                    annual[
                        "Form"
                    ],

                "Period Type":
                    "quarter",

                "Accession":
                    annual[
                        "Accession"
                    ],

                "Quarter Source":
                    "derived_q4",

                "Source Priority":
                    1,
            }
        )


    if derived_rows:

        derived = pd.DataFrame(
            derived_rows
        )


        combined = pd.concat(
            [
                direct,
                derived,
            ],
            ignore_index=True,
        )

    else:

        combined = direct


    if combined.empty:
        return combined


    #
    # If a company reports an explicit quarterly Q4
    # AND we derived one, prefer the direct value.
    #

    combined = (
        combined
        .sort_values(
            [
                "End",
                "Filed",
                "Source Priority",
                "Alias Priority",
            ]
        )
        .drop_duplicates(
            subset=[
                "Concept",
                "End",
                "Filed",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


    return combined


def _latest_value_for_end(
    observations: pd.DataFrame,
    end: pd.Timestamp,
    as_of: pd.Timestamp,
) -> float:

    if observations.empty:
        return np.nan


    available = observations[
        (
            observations[
                "End"
            ] == end
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


    available = available.sort_values(
        [
            "Filed",
            "Source Priority"
            if "Source Priority"
            in available.columns
            else "Alias Priority",
            "Alias Priority",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )


    return float(
        available.iloc[
            0
        ][
            "Value"
        ]
    )


def _find_prior_year_end(
    tables: list[pd.DataFrame],
    current_end: pd.Timestamp,
    as_of: pd.Timestamp,
) -> pd.Timestamp | None:
    """
    Find the closest historical quarter end about
    one year before the current quarter.
    """

    candidates = set()


    for table in tables:

        if table.empty:
            continue


        valid = table[
            table[
                "Filed"
            ] <= as_of
        ]


        for end in valid[
            "End"
        ].dropna():

            end = pd.Timestamp(
                end
            )

            gap = (
                current_end
                - end
            ).days


            if 330 <= gap <= 400:

                candidates.add(
                    end
                )


    if not candidates:
        return None


    return min(
        candidates,
        key=lambda end:
            abs(
                (
                    current_end
                    - end
                ).days
                - 365
            ),
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


def build_company_quarterly_timeline(
    companyfacts: dict,
) -> pd.DataFrame:
    """
    Build point-in-time quarterly fundamental
    snapshots for one company.

    Each row becomes available on its filing date.
    """

    flow_tables = {}


    for concept in (
        QUARTERLY_FLOW_CONCEPTS
    ):

        raw = collect_flow_observations(
            companyfacts=companyfacts,
            concept_name=concept,
        )


        flow_tables[
            concept
        ] = (
            build_single_quarter_observations(
                raw
            )
        )


    instant_tables = {
        concept:
            collect_instant_observations(
                companyfacts=companyfacts,
                concept_name=concept,
            )

        for concept
        in QUARTERLY_INSTANT_CONCEPTS
    }


    #
    # A snapshot can update whenever one of these
    # SEC facts is filed.
    #

    filing_dates = sorted(
        {
            pd.Timestamp(
                filed
            )

            for table in (
                list(
                    flow_tables.values()
                )
                + list(
                    instant_tables.values()
                )
            )

            if not table.empty

            for filed in table[
                "Filed"
            ].dropna()
        }
    )


    rows = []


    for as_of in filing_dates:

        #
        # Determine the latest actual single-quarter
        # period known by this filing date.
        #

        known_ends = []


        for table in (
            flow_tables.values()
        ):

            if table.empty:
                continue


            known = table[
                table[
                    "Filed"
                ] <= as_of
            ]


            known_ends.extend(
                known[
                    "End"
                ].dropna().tolist()
            )


        if not known_ends:
            continue


        current_end = pd.Timestamp(
            max(
                known_ends
            )
        )


        prior_year_end = (
            _find_prior_year_end(
                tables=(
                    list(
                        flow_tables.values()
                    )
                    + list(
                        instant_tables.values()
                    )
                ),
                current_end=current_end,
                as_of=as_of,
            )
        )


        if prior_year_end is None:
            continue


        revenue = _latest_value_for_end(
            flow_tables[
                "Revenue"
            ],
            current_end,
            as_of,
        )


        revenue_prior = (
            _latest_value_for_end(
                flow_tables[
                    "Revenue"
                ],
                prior_year_end,
                as_of,
            )
        )


        net_income = (
            _latest_value_for_end(
                flow_tables[
                    "Net Income"
                ],
                current_end,
                as_of,
            )
        )


        net_income_prior = (
            _latest_value_for_end(
                flow_tables[
                    "Net Income"
                ],
                prior_year_end,
                as_of,
            )
        )


        shares = (
            _latest_value_for_end(
                flow_tables[
                    "Diluted Shares"
                ],
                current_end,
                as_of,
            )
        )


        shares_prior = (
            _latest_value_for_end(
                flow_tables[
                    "Diluted Shares"
                ],
                prior_year_end,
                as_of,
            )
        )


        assets = (
            _latest_value_for_end(
                instant_tables[
                    "Assets"
                ],
                current_end,
                as_of,
            )
        )


        assets_prior = (
            _latest_value_for_end(
                instant_tables[
                    "Assets"
                ],
                prior_year_end,
                as_of,
            )
        )


        equity = (
            _latest_value_for_end(
                instant_tables[
                    "Equity"
                ],
                current_end,
                as_of,
            )
        )


        net_margin = (
            _safe_ratio(
                net_income,
                revenue,
            )
        )


        prior_net_margin = (
            _safe_ratio(
                net_income_prior,
                revenue_prior,
            )
        )


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

                "Quarter End":
                    current_end,

                "Quarterly Revenue Growth YoY":
                    _safe_growth(
                        revenue,
                        revenue_prior,
                    ),

                "Quarterly Net Margin":
                    net_margin,

                "Quarterly Net Margin Change YoY":
                    (
                        net_margin
                        - prior_net_margin
                    )
                    if (
                        pd.notna(
                            net_margin
                        )
                        and
                        pd.notna(
                            prior_net_margin
                        )
                    )
                    else np.nan,

                "Quarterly Net Income To Assets":
                    _safe_ratio(
                        net_income,
                        assets,
                    ),

                "Quarterly Asset Growth YoY":
                    _safe_growth(
                        assets,
                        assets_prior,
                    ),

                "Quarterly Equity To Assets":
                    _safe_ratio(
                        equity,
                        assets,
                    ),

                "Quarterly Diluted Share Growth YoY":
                    _safe_growth(
                        shares,
                        shares_prior,
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
        .drop_duplicates(
            subset=[
                "CIK",
                "Filed",
                "Quarter End",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


    return result