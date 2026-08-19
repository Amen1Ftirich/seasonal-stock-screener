from __future__ import annotations

import numpy as np
import pandas as pd

from src.sec_data import (
    normalize_ticker,
)

from src.sec_quarterly import (
    FEATURE_COLUMNS_SEC_QUARTERLY,
)


CROSS_SECTIONAL_QUARTERLY_COLUMNS = [
    f"CS {column}"
    for column
    in FEATURE_COLUMNS_SEC_QUARTERLY
]


QUARTERLY_MISSING_COLUMNS = [
    f"Missing {column}"
    for column
    in FEATURE_COLUMNS_SEC_QUARTERLY
]


FEATURE_COLUMNS_V8C = (
    CROSS_SECTIONAL_QUARTERLY_COLUMNS
    + QUARTERLY_MISSING_COLUMNS
)


def attach_point_in_time_quarterly(
    data: pd.DataFrame,
    resolution: pd.DataFrame,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the latest quarterly SEC snapshot
    known BEFORE each prediction month.

    Example:

        prediction month = 2025-06
        cutoff           = 2025-05-31

    A filing dated 2025-06-01 is not available.
    """

    result = data.copy()


    result[
        "Period"
    ] = pd.PeriodIndex(
        result[
            "Period"
        ],
        freq="M",
    )


    result[
        "Ticker"
    ] = (
        result[
            "Ticker"
        ]
        .astype(str)
        .apply(
            normalize_ticker
        )
    )


    result[
        "Quarterly Fundamental Cutoff"
    ] = (
        result[
            "Period"
        ].dt.start_time
        - pd.Timedelta(
            days=1
        )
    )


    #
    # Ticker -> CIK
    #

    cik_map = resolution[
        [
            "Ticker",
            "CIK",
        ]
    ].copy()


    cik_map[
        "Ticker"
    ] = (
        cik_map[
            "Ticker"
        ]
        .astype(str)
        .apply(
            normalize_ticker
        )
    )


    cik_map = (
        cik_map
        .dropna(
            subset=[
                "CIK",
            ]
        )
        .drop_duplicates(
            subset=[
                "Ticker",
            ],
            keep="last",
        )
    )


    cik_map[
        "CIK"
    ] = (
        cik_map[
            "CIK"
        ]
        .astype(int)
    )


    result = result.merge(
        cik_map,
        on="Ticker",
        how="left",
    )


    result[
        "_Row ID"
    ] = np.arange(
        len(result)
    )


    quarterly = timeline.copy()


    quarterly[
        "CIK"
    ] = (
        quarterly[
            "CIK"
        ]
        .astype(int)
    )


    quarterly[
        "Filed"
    ] = pd.to_datetime(
        quarterly[
            "Filed"
        ]
    )


    quarterly[
        "Quarter End"
    ] = pd.to_datetime(
        quarterly[
            "Quarter End"
        ]
    )


    merged_frames = []


    mapped = result[
        result[
            "CIK"
        ].notna()
    ].copy()


    unmapped = result[
        result[
            "CIK"
        ].isna()
    ].copy()


    for cik, company_rows in mapped.groupby(
        "CIK"
    ):

        company_rows = (
            company_rows
            .sort_values(
                "Quarterly Fundamental Cutoff"
            )
        )


        company_timeline = (
            quarterly[
                quarterly[
                    "CIK"
                ] == int(
                    cik
                )
            ]
            .sort_values(
                "Filed"
            )
        )


        if company_timeline.empty:

            company_rows[
                "Filed"
            ] = pd.NaT

            company_rows[
                "Quarter End"
            ] = pd.NaT


            for column in (
                FEATURE_COLUMNS_SEC_QUARTERLY
            ):

                company_rows[
                    column
                ] = np.nan


            merged_frames.append(
                company_rows
            )

            continue


        sec_columns = [
            "Filed",
            "Quarter End",
        ] + (
            FEATURE_COLUMNS_SEC_QUARTERLY
        )


        merged = pd.merge_asof(
            company_rows,
            company_timeline[
                sec_columns
            ],
            left_on=(
                "Quarterly Fundamental Cutoff"
            ),
            right_on="Filed",
            direction="backward",
            allow_exact_matches=True,
        )


        merged_frames.append(
            merged
        )


    if not unmapped.empty:

        unmapped[
            "Filed"
        ] = pd.NaT

        unmapped[
            "Quarter End"
        ] = pd.NaT


        for column in (
            FEATURE_COLUMNS_SEC_QUARTERLY
        ):

            unmapped[
                column
            ] = np.nan


        merged_frames.append(
            unmapped
        )


    result = pd.concat(
        merged_frames,
        ignore_index=True,
    )


    result = (
        result
        .sort_values(
            "_Row ID"
        )
        .drop(
            columns=[
                "_Row ID",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    #
    # Hard look-ahead assertion.
    #

    month_start = (
        result[
            "Period"
        ].dt.start_time
    )


    violations = result[
        result[
            "Filed"
        ].notna()
        &
        (
            result[
                "Filed"
            ]
            >= month_start
        )
    ]


    if not violations.empty:

        raise RuntimeError(
            "QUARTERLY POINT-IN-TIME "
            "VIOLATION: "
            f"{len(violations)} rows"
        )


    result[
        "Quarterly Fundamental Age Days"
    ] = (
        result[
            "Quarterly Fundamental Cutoff"
        ]
        - result[
            "Filed"
        ]
    ).dt.days


    return result


def add_cross_sectional_quarterly_features(
    data: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Monthly cross-sectional processing:

        missing indicator
        1% / 99% winsorization
        z-score within the S&P cross-section
        missing values -> neutral score of zero
    """

    result = data.copy()


    for column in (
        FEATURE_COLUMNS_SEC_QUARTERLY
    ):

        missing_column = (
            f"Missing {column}"
        )

        cs_column = (
            f"CS {column}"
        )


        result[
            missing_column
        ] = (
            result[
                column
            ]
            .isna()
            .astype(float)
        )


        grouped = result.groupby(
            "Period"
        )[
            column
        ]


        lower = grouped.transform(
            lambda values:
                values.quantile(
                    lower_quantile
                )
        )


        upper = grouped.transform(
            lambda values:
                values.quantile(
                    upper_quantile
                )
        )


        clipped = (
            result[
                column
            ]
            .clip(
                lower=lower,
                upper=upper,
            )
        )


        mean = (
            clipped.groupby(
                result[
                    "Period"
                ]
            )
            .transform(
                "mean"
            )
        )


        std = (
            clipped.groupby(
                result[
                    "Period"
                ]
            )
            .transform(
                "std"
            )
            .replace(
                0,
                np.nan,
            )
        )


        result[
            cs_column
        ] = (
            (
                clipped
                - mean
            )
            / std
        ).fillna(
            0.0
        )


    return result