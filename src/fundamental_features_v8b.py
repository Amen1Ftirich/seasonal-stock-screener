from __future__ import annotations

import numpy as np
import pandas as pd

from src.sec_data import (
    normalize_ticker,
)

from src.sec_point_in_time import (
    FEATURE_COLUMNS_SEC,
)


CROSS_SECTIONAL_FUNDAMENTAL_COLUMNS = [
    f"CS {column}"
    for column in FEATURE_COLUMNS_SEC
]


FUNDAMENTAL_MISSING_COLUMNS = [
    f"Missing {column}"
    for column in FEATURE_COLUMNS_SEC
]


FEATURE_COLUMNS_V8B = (
    CROSS_SECTIONAL_FUNDAMENTAL_COLUMNS
    + FUNDAMENTAL_MISSING_COLUMNS
)


def attach_point_in_time_fundamentals(
    data: pd.DataFrame,
    resolution: pd.DataFrame,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the most recently FILED annual fundamental
    snapshot that was known before each stock-month.

    For a prediction for May 2021:

        cutoff = 2021-04-30

    A filing dated 2021-05-01 is therefore NOT allowed.

    This is intentionally conservative because SEC
    Company Facts gives us a filing date, not the exact
    intraday publication time.
    """

    result = data.copy()


    #
    # -----------------------------------------
    # NORMALIZE MONTH
    # -----------------------------------------
    #

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


    #
    # Last calendar day BEFORE the prediction month.
    #

    result[
        "Fundamental Cutoff"
    ] = (
        result[
            "Period"
        ].dt.start_time
        - pd.Timedelta(
            days=1
        )
    )


    #
    # -----------------------------------------
    # TICKER -> PERMANENT CIK
    # -----------------------------------------
    #

    cik_map = (
        resolution[
            [
                "Ticker",
                "CIK",
            ]
        ]
        .copy()
    )


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
            subset=["CIK"]
        )
        .drop_duplicates(
            subset=["Ticker"],
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


    #
    # -----------------------------------------
    # PREPARE SEC TIMELINE
    # -----------------------------------------
    #

    fundamental_data = (
        timeline.copy()
    )


    fundamental_data[
        "CIK"
    ] = (
        fundamental_data[
            "CIK"
        ]
        .astype(int)
    )


    fundamental_data[
        "Filed"
    ] = pd.to_datetime(
        fundamental_data[
            "Filed"
        ]
    )


    fundamental_data[
        "Fiscal End"
    ] = pd.to_datetime(
        fundamental_data[
            "Fiscal End"
        ]
    )


    #
    # We perform merge_asof separately for each CIK.
    #
    # This is more explicit and avoids subtle sorting
    # problems with a global merge_asof.
    #

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
                "Fundamental Cutoff"
            )
        )


        company_timeline = (
            fundamental_data[
                fundamental_data[
                    "CIK"
                ] == int(cik)
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
                "Fiscal End"
            ] = pd.NaT


            for column in (
                FEATURE_COLUMNS_SEC
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
            "Fiscal End",
        ] + FEATURE_COLUMNS_SEC


        merged = pd.merge_asof(
            company_rows,
            company_timeline[
                sec_columns
            ],
            left_on=(
                "Fundamental Cutoff"
            ),
            right_on="Filed",
            direction="backward",
            allow_exact_matches=True,
        )


        merged_frames.append(
            merged
        )


    #
    # Companies without a resolved CIK.
    #

    if not unmapped.empty:

        unmapped[
            "Filed"
        ] = pd.NaT

        unmapped[
            "Fiscal End"
        ] = pd.NaT


        for column in (
            FEATURE_COLUMNS_SEC
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
    # -----------------------------------------
    # HARD LOOK-AHEAD CHECK
    # -----------------------------------------
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
            "POINT-IN-TIME VIOLATION: "
            f"{len(violations)} rows contain "
            "fundamentals filed during or after "
            "the prediction month's start."
        )


    result[
        "Fundamental Age Days"
    ] = (
        result[
            "Fundamental Cutoff"
        ]
        - result[
            "Filed"
        ]
    ).dt.days


    return result


def add_cross_sectional_fundamental_features(
    data: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Transform raw fundamentals into monthly
    cross-sectional model inputs.

    For each month and feature:

        1. record missingness
        2. winsorize at 1st / 99th percentile
        3. cross-sectionally z-score
        4. replace missing z-score with zero

    Zero therefore means the cross-sectional average,
    while a separate indicator tells the model that the
    original value was missing.
    """

    result = data.copy()


    for column in FEATURE_COLUMNS_SEC:

        missing_column = (
            f"Missing {column}"
        )

        output_column = (
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


        monthly_mean = (
            clipped.groupby(
                result[
                    "Period"
                ]
            )
            .transform(
                "mean"
            )
        )


        monthly_std = (
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
            output_column
        ] = (
            (
                clipped
                - monthly_mean
            )
            / monthly_std
        )


        #
        # Missing = neutral cross-sectional score.
        #
        # Missingness is separately represented by
        # the binary missing indicator.
        #

        result[
            output_column
        ] = (
            result[
                output_column
            ]
            .fillna(
                0.0
            )
        )


    return result