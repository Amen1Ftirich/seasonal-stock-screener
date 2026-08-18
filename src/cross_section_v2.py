from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.signals_v2 import (
    momentum_12_1,
    same_month_seasonality,
)


def build_cross_section(
    price_map: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
    target_month: int,
    as_of_date: date,
    lookback_years: int = 15,
    recent_years: int = 5,
) -> pd.DataFrame:
    """
    Build one cross-sectional snapshot of the
    stock universe using only information available
    as of the specified date.
    """

    rows = []

    for ticker, prices in price_map.items():

        try:

            seasonal = same_month_seasonality(
                stock_prices=prices,
                benchmark_prices=benchmark_prices,
                target_month=target_month,
                as_of_date=as_of_date,
                lookback_years=lookback_years,
                recent_years=recent_years,
            )

            momentum = momentum_12_1(
                prices=prices,
                as_of_date=as_of_date,
            )

            rows.append(
                {
                    "Ticker": ticker,
                    **seasonal,
                    "Momentum 12-1": momentum,
                }
            )

        except Exception:
            continue

    table = pd.DataFrame(rows)

    if table.empty:
        return table

    table = table.dropna(
        subset=[
            "Season Mean Excess",
            "Recent Mean Excess",
            "Momentum 12-1",
            "Season Q25 Excess",
        ]
    ).copy()

    #
    # Everything from this point is
    # CROSS-SECTIONAL.
    #
    # A score of 0.90 means that stock is
    # stronger than roughly 90% of the universe
    # on that factor.
    #

    table["Long Seasonal Rank"] = (
        table["Season Mean Excess"]
        .rank(
            pct=True,
            method="average",
        )
    )

    table["Recent Seasonal Rank"] = (
        table["Recent Mean Excess"]
        .rank(
            pct=True,
            method="average",
        )
    )

    table["Momentum Rank"] = (
        table["Momentum 12-1"]
        .rank(
            pct=True,
            method="average",
        )
    )

    table["Downside Rank"] = (
        table["Season Q25 Excess"]
        .rank(
            pct=True,
            method="average",
        )
    )

    #
    # A seasonal tendency should survive
    # both long and recent history.
    #

    table["Persistent Seasonal Rank"] = (
        table[
            [
                "Long Seasonal Rank",
                "Recent Seasonal Rank",
            ]
        ].min(axis=1)
    )

    #
    # Equal-weight geometric combination.
    #
    # We are deliberately NOT fitting
    # weights against your 2016-2025 results.
    #
    # A weak component materially penalizes
    # the final score.
    #

    components = (
        table[
            [
                "Persistent Seasonal Rank",
                "Momentum Rank",
                "Downside Rank",
            ]
        ]
        .clip(lower=1e-6)
    )

    table["Edge Score"] = (
        components.prod(axis=1)
        ** (1 / 3)
    )

    return table.sort_values(
        "Edge Score",
        ascending=False,
    ).reset_index(drop=True)


def select_candidates(
    cross_section: pd.DataFrame,
    top_n: int = 10,
    minimum_samples: int = 8,
    minimum_long_season_rank: float = 0.80,
    minimum_recent_season_rank: float = 0.60,
    minimum_momentum_rank: float = 0.50,
    minimum_downside_rank: float = 0.40,
) -> pd.DataFrame:
    """
    Select candidates without optimizing thresholds
    against the current backtest.

    Main idea:
        strong long-term seasonal rank
        + still alive recently
        + positive relative momentum
        + acceptable historical downside
    """

    if cross_section.empty:
        return cross_section.copy()

    selected = cross_section[
        (
            cross_section["Season Samples"]
            >= minimum_samples
        )
        & (
            cross_section["Long Seasonal Rank"]
            >= minimum_long_season_rank
        )
        & (
            cross_section["Recent Seasonal Rank"]
            >= minimum_recent_season_rank
        )
        & (
            cross_section["Momentum Rank"]
            >= minimum_momentum_rank
        )
        & (
            cross_section["Downside Rank"]
            >= minimum_downside_rank
        )
    ].copy()

    selected = selected.sort_values(
        [
            "Edge Score",
            "Persistent Seasonal Rank",
            "Momentum Rank",
        ],
        ascending=False,
    )

    selected = selected.head(
        top_n
    ).reset_index(drop=True)

    selected.insert(
        0,
        "Rank",
        range(
            1,
            len(selected) + 1,
        ),
    )

    return selected