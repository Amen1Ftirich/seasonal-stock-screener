from __future__ import annotations

import pandas as pd

from src.features_v3 import (
    build_stock_features,
)

from src.sector_features_v8 import (
    add_sector_relative_features,
    prepare_sector_returns,
)


def build_panel_dataset_v8(
    price_map: dict[
        str,
        pd.DataFrame,
    ],
    benchmark_prices: pd.DataFrame,
    sector_price_map: dict[
        str,
        pd.DataFrame,
    ],
    minimum_season_samples: int = 5,
) -> pd.DataFrame:
    """
    Build V8 panel with all V7 features plus
    sector-relative price information.
    """

    sector_returns = (
        prepare_sector_returns(
            sector_price_map
        )
    )


    frames = []

    total = len(
        price_map
    )


    for number, (
        ticker,
        prices,
    ) in enumerate(
        price_map.items(),
        start=1,
    ):

        if number % 50 == 0:

            print(
                f"V8 feature engineering "
                f"{number}/{total}"
            )


        try:

            features = (
                build_stock_features(
                    stock_prices=prices,
                    benchmark_prices=(
                        benchmark_prices
                    ),
                )
            )


            features = (
                add_sector_relative_features(
                    stock_features=features,
                    sector_returns=(
                        sector_returns
                    ),
                )
            )


            features[
                "Ticker"
            ] = ticker


            frames.append(
                features
            )


        except Exception as exc:

            print(
                f"ERROR V8 features "
                f"{ticker}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )


    if not frames:

        raise ValueError(
            "No V8 feature tables "
            "were successfully built"
        )


    panel = pd.concat(
        frames,
        ignore_index=True,
    )


    panel = panel[
        panel[
            "Season Samples"
        ] >= minimum_season_samples
    ].copy()


    return panel