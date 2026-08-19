from __future__ import annotations

import numpy as np
import pandas as pd

from src.features_v3 import (
    prepare_monthly_prices,
)


SECTOR_ETFS = [
    "VGT",
    "VFH",
    "VHT",
    "VDE",
    "VIS",
    "VCR",
    "VDC",
    "VPU",
    "VAW",
    "VOX",
    "VNQ",
]


SECTOR_FEATURE_COLUMNS = [
    "Sector Correlation 24M",
    "Sector Relative Momentum 3M",
    "Sector Relative Momentum 6M",
    "Sector Relative Momentum 12M",
    "Sector Relative Volatility 6M",
    "Sector Relative Volatility 12M",
]


def prepare_sector_returns(
    sector_price_map: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """
    Create one monthly-return table containing
    all sector ETFs.
    """

    frames = []

    for ticker, prices in (
        sector_price_map.items()
    ):

        monthly = (
            prepare_monthly_prices(
                prices
            )
        )

        monthly = monthly[
            [
                "Period",
                "Return",
            ]
        ].copy()

        monthly = monthly.rename(
            columns={
                "Return":
                    ticker,
            }
        )

        frames.append(
            monthly
        )

    if not frames:

        raise ValueError(
            "No sector ETF data available"
        )

    result = frames[0]

    for frame in frames[1:]:

        result = result.merge(
            frame,
            on="Period",
            how="outer",
        )

    return (
        result
        .sort_values("Period")
        .reset_index(drop=True)
    )


def add_sector_relative_features(
    stock_features: pd.DataFrame,
    sector_returns: pd.DataFrame,
    lookback_months: int = 24,
) -> pd.DataFrame:
    """
    For every stock-month:

    1. Look backward only.
    2. Determine which sector ETF had the strongest
       trailing correlation with the stock.
    3. Measure stock performance relative to that ETF.
    4. Build lagged sector-relative momentum and risk
       features.

    Current-month return never enters a predictor for
    the same month.
    """

    data = stock_features.copy()

    data = data.merge(
        sector_returns,
        on="Period",
        how="left",
    )


    selected_sector = []
    selected_correlation = []
    sector_return = []


    #
    # -----------------------------------------
    # DYNAMIC HISTORICAL SECTOR SELECTION
    # -----------------------------------------
    #

    for index, row in data.iterrows():

        history_start = max(
            0,
            index - lookback_months,
        )

        history = data.iloc[
            history_start:index
        ]

        best_sector = None
        best_correlation = np.nan


        for sector in SECTOR_ETFS:

            if sector not in data.columns:
                continue

            pair = history[
                [
                    "Return",
                    sector,
                ]
            ].dropna()


            if len(pair) < 12:
                continue


            correlation = pair[
                "Return"
            ].corr(
                pair[
                    sector
                ]
            )


            if pd.isna(
                correlation
            ):
                continue


            if (
                best_sector is None
                or correlation
                > best_correlation
            ):

                best_sector = sector

                best_correlation = (
                    float(
                        correlation
                    )
                )


        selected_sector.append(
            best_sector
        )

        selected_correlation.append(
            best_correlation
        )


        if (
            best_sector is None
            or pd.isna(
                row.get(
                    best_sector,
                    np.nan,
                )
            )
        ):

            sector_return.append(
                np.nan
            )

        else:

            sector_return.append(
                float(
                    row[
                        best_sector
                    ]
                )
            )


    data[
        "Selected Sector ETF"
    ] = selected_sector


    data[
        "Sector Correlation 24M"
    ] = selected_correlation


    data[
        "Selected Sector Return"
    ] = sector_return


    #
    # Realized sector-relative return.
    #
    # Current month's value can exist here because
    # it will be shifted before being used as a
    # predictor.
    #

    data[
        "Sector Relative Return"
    ] = (
        data[
            "Return"
        ]
        - data[
            "Selected Sector Return"
        ]
    )


    #
    # -----------------------------------------
    # LAG BEFORE FEATURE CREATION
    # -----------------------------------------
    #

    lagged_relative = (
        data[
            "Sector Relative Return"
        ].shift(1)
    )


    data[
        "Sector Relative Momentum 3M"
    ] = (
        (
            1
            + lagged_relative
        )
        .rolling(3)
        .apply(
            np.prod,
            raw=True,
        )
        - 1
    )


    data[
        "Sector Relative Momentum 6M"
    ] = (
        (
            1
            + lagged_relative
        )
        .rolling(6)
        .apply(
            np.prod,
            raw=True,
        )
        - 1
    )


    data[
        "Sector Relative Momentum 12M"
    ] = (
        (
            1
            + lagged_relative
        )
        .rolling(12)
        .apply(
            np.prod,
            raw=True,
        )
        - 1
    )


    data[
        "Sector Relative Volatility 6M"
    ] = (
        lagged_relative
        .rolling(6)
        .std()
    )


    data[
        "Sector Relative Volatility 12M"
    ] = (
        lagged_relative
        .rolling(12)
        .std()
    )


    #
    # We no longer need 11 ETF-return columns
    # inside every stock's feature table.
    #

    drop_columns = [
        sector

        for sector in SECTOR_ETFS

        if sector in data.columns
    ]


    data = data.drop(
        columns=drop_columns
    )


    return data