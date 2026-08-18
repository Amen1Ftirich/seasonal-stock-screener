from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.cross_section_v2 import (
    build_cross_section,
    select_candidates,
)

from src.signals_v2 import monthly_returns


def get_month_return(
    prices: pd.DataFrame,
    year: int,
    month: int,
) -> float | None:
    """
    Return the stock's realized return during one calendar month.

    Example:
        year=2020
        month=9

    uses the August month-end close to September month-end close.
    """

    monthly = monthly_returns(prices)

    target_period = pd.Period(
        year=year,
        month=month,
        freq="M",
    )

    row = monthly[
        monthly["Month"] == target_period
    ]

    if row.empty:
        return None

    return float(
        row.iloc[0]["Return"]
    )


def evaluate_selected_stocks(
    selected: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
    test_year: int,
    target_month: int,
) -> pd.DataFrame:
    """
    Reveal the unseen target-month returns after
    stocks have already been selected.
    """

    benchmark_return = get_month_return(
        prices=benchmark_prices,
        year=test_year,
        month=target_month,
    )

    if benchmark_return is None:
        return pd.DataFrame()

    rows = []

    for _, candidate in selected.iterrows():

        ticker = candidate["Ticker"]

        if ticker not in price_map:
            continue

        realized_return = get_month_return(
            prices=price_map[ticker],
            year=test_year,
            month=target_month,
        )

        if realized_return is None:
            continue

        excess_return = (
            realized_return
            - benchmark_return
        )

        rows.append(
            {
                "Test Year": test_year,
                "Ticker": ticker,

                "Prior Rank":
                    int(candidate["Rank"]),

                "Edge Score":
                    float(candidate["Edge Score"]),

                "Long Seasonal Rank":
                    float(
                        candidate[
                            "Long Seasonal Rank"
                        ]
                    ),

                "Recent Seasonal Rank":
                    float(
                        candidate[
                            "Recent Seasonal Rank"
                        ]
                    ),

                "Persistent Seasonal Rank":
                    float(
                        candidate[
                            "Persistent Seasonal Rank"
                        ]
                    ),

                "Momentum Rank":
                    float(
                        candidate["Momentum Rank"]
                    ),

                "Downside Rank":
                    float(
                        candidate["Downside Rank"]
                    ),

                "Historical Mean Excess":
                    float(
                        candidate[
                            "Season Mean Excess"
                        ]
                    ),

                "Historical Recent Excess":
                    float(
                        candidate[
                            "Recent Mean Excess"
                        ]
                    ),

                "Historical Q25 Excess":
                    float(
                        candidate[
                            "Season Q25 Excess"
                        ]
                    ),

                "Realized Return":
                    realized_return,

                "Benchmark Return":
                    benchmark_return,

                "Realized Excess Return":
                    excess_return,

                "Realized Win":
                    realized_return > 0,

                "Beat Benchmark":
                    excess_return > 0,
            }
        )

    return pd.DataFrame(rows)


def summarize_backtest(
    trades: pd.DataFrame,
) -> dict:
    """
    Summarize fully out-of-sample trades.
    """

    if trades.empty:

        return {
            "Trades": 0,
            "Win Rate": 0.0,
            "Beat Benchmark Rate": 0.0,
            "Average Return": 0.0,
            "Median Return": 0.0,
            "Average Excess Return": 0.0,
            "Median Excess Return": 0.0,
            "Worst Return": 0.0,
            "Best Return": 0.0,
            "Return Std": 0.0,
            "Excess Std": 0.0,
        }

    realized = trades[
        "Realized Return"
    ]

    excess = trades[
        "Realized Excess Return"
    ]

    return {
        "Trades":
            len(trades),

        "Win Rate":
            float(
                trades["Realized Win"].mean()
            ),

        "Beat Benchmark Rate":
            float(
                trades[
                    "Beat Benchmark"
                ].mean()
            ),

        "Average Return":
            float(realized.mean()),

        "Median Return":
            float(realized.median()),

        "Average Excess Return":
            float(excess.mean()),

        "Median Excess Return":
            float(excess.median()),

        "Worst Return":
            float(realized.min()),

        "Best Return":
            float(realized.max()),

        "Return Std":
            float(realized.std(ddof=1))
            if len(realized) > 1
            else 0.0,

        "Excess Std":
            float(excess.std(ddof=1))
            if len(excess) > 1
            else 0.0,
    }


def summarize_by_year(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Show how the portfolio performed each historical year.
    """

    if trades.empty:
        return pd.DataFrame()

    yearly = (
        trades
        .groupby("Test Year")
        .agg(
            Stocks=("Ticker", "count"),

            Portfolio_Return=(
                "Realized Return",
                "mean",
            ),

            Benchmark_Return=(
                "Benchmark Return",
                "first",
            ),

            Portfolio_Excess=(
                "Realized Excess Return",
                "mean",
            ),

            Win_Rate=(
                "Realized Win",
                "mean",
            ),

            Beat_Rate=(
                "Beat Benchmark",
                "mean",
            ),
        )
        .reset_index()
    )

    return yearly


def bootstrap_mean_excess(
    trades: pd.DataFrame,
    simulations: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap the average out-of-sample excess return.

    This does not solve every statistical issue, but it tells us
    how uncertain our observed alpha estimate is.
    """

    if trades.empty:

        return {
            "Bootstrap Mean Excess": 0.0,
            "Bootstrap Lower 5%": 0.0,
            "Bootstrap Upper 95%": 0.0,
            "Probability Mean Excess > 0": 0.0,
        }

    excess = (
        trades[
            "Realized Excess Return"
        ]
        .dropna()
        .to_numpy()
    )

    if len(excess) == 0:

        return {
            "Bootstrap Mean Excess": 0.0,
            "Bootstrap Lower 5%": 0.0,
            "Bootstrap Upper 95%": 0.0,
            "Probability Mean Excess > 0": 0.0,
        }

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        excess,
        size=(
            simulations,
            len(excess),
        ),
        replace=True,
    )

    sample_means = samples.mean(axis=1)

    return {
        "Bootstrap Mean Excess":
            float(sample_means.mean()),

        "Bootstrap Lower 5%":
            float(
                np.quantile(
                    sample_means,
                    0.05,
                )
            ),

        "Bootstrap Upper 95%":
            float(
                np.quantile(
                    sample_means,
                    0.95,
                )
            ),

        "Probability Mean Excess > 0":
            float(
                (
                    sample_means > 0
                ).mean()
            ),
    }


def backtest_cross_section_v2(
    price_map: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
    test_years: list[int],
    target_month: int,

    top_n: int = 10,

    lookback_years: int = 15,
    recent_years: int = 5,

    minimum_samples: int = 8,

    minimum_long_season_rank: float = 0.80,
    minimum_recent_season_rank: float = 0.60,
    minimum_momentum_rank: float = 0.50,
    minimum_downside_rank: float = 0.40,

) -> tuple[
    pd.DataFrame,
    dict,
    pd.DataFrame,
    dict,
]:
    """
    Historical out-of-sample test of V2.

    For each test year:

        1. Move to the first day of target_month.
        2. Use only data available before that date.
        3. Calculate cross-sectional factors.
        4. Rank the universe.
        5. Select top candidates.
        6. Reveal target-month returns.
    """

    all_trades = []

    for test_year in test_years:

        print()
        print(
            "===================================="
        )

        print(
            f"V2 TEST YEAR {test_year}"
        )

        print(
            "===================================="
        )

        as_of_date = date(
            test_year,
            target_month,
            1,
        )

        snapshot = build_cross_section(
            price_map=price_map,
            benchmark_prices=benchmark_prices,

            target_month=target_month,
            as_of_date=as_of_date,

            lookback_years=lookback_years,
            recent_years=recent_years,
        )

        print(
            f"Eligible cross-section: "
            f"{len(snapshot)} stocks"
        )

        selected = select_candidates(
            cross_section=snapshot,

            top_n=top_n,

            minimum_samples=minimum_samples,

            minimum_long_season_rank=(
                minimum_long_season_rank
            ),

            minimum_recent_season_rank=(
                minimum_recent_season_rank
            ),

            minimum_momentum_rank=(
                minimum_momentum_rank
            ),

            minimum_downside_rank=(
                minimum_downside_rank
            ),
        )

        print(
            f"Selected: {len(selected)} stocks"
        )

        if selected.empty:
            continue

        print(
            selected[
                [
                    "Rank",
                    "Ticker",
                    "Edge Score",
                    "Long Seasonal Rank",
                    "Recent Seasonal Rank",
                    "Momentum Rank",
                ]
            ].to_string(
                index=False
            )
        )

        realized = evaluate_selected_stocks(
            selected=selected,
            price_map=price_map,
            benchmark_prices=benchmark_prices,
            test_year=test_year,
            target_month=target_month,
        )

        if not realized.empty:
            all_trades.append(
                realized
            )

    if all_trades:

        trades = pd.concat(
            all_trades,
            ignore_index=True,
        )

    else:

        trades = pd.DataFrame()

    summary = summarize_backtest(
        trades
    )

    yearly = summarize_by_year(
        trades
    )

    bootstrap = bootstrap_mean_excess(
        trades
    )

    return (
        trades,
        summary,
        yearly,
        bootstrap,
    )