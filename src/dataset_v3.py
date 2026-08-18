from __future__ import annotations

import pandas as pd

from src.features_v3 import (
    build_stock_features,
)


def build_panel_dataset(
    price_map: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
    minimum_season_samples: int = 5,
) -> pd.DataFrame:
    """
    Construct one panel:

        Month | Ticker | features | future return

    across the entire universe.
    """

    frames = []

    total = len(price_map)

    for number, (
        ticker,
        prices,
    ) in enumerate(
        price_map.items(),
        start=1,
    ):

        if number % 50 == 0:

            print(
                f"Feature engineering "
                f"{number}/{total}"
            )

        try:

            features = (
                build_stock_features(
                    stock_prices=prices,
                    benchmark_prices=benchmark_prices,
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
                f"ERROR building features for "
                f"{ticker}: {type(exc).__name__}: {exc}"
            )

            continue

    if not frames:

        raise ValueError(
            "No stock feature tables were successfully built. "
            "Check the feature-engineering errors printed above."
        )

    panel = pd.concat(
        frames,
        ignore_index=True,
    )

    panel = panel[
        panel["Season Samples"]
        >= minimum_season_samples
    ].copy()

    return panel