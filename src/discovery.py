from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.data import (
    get_price_history,
    load_price_map,
)
from src.scanner import scan_loaded_tickers


def generate_upcoming_dates(
    start_date: date | None = None,
    days_ahead: int = 30,
) -> list[date]:

    if start_date is None:
        start_date = date.today()

    if days_ahead < 0:
        raise ValueError(
            "days_ahead cannot be negative"
        )

    return [
        start_date + timedelta(days=offset)
        for offset in range(days_ahead + 1)
    ]


def discover_upcoming_windows(
    tickers: list[str],
    start_date: date | None = None,
    days_ahead: int = 30,
    holding_periods: list[int] | None = None,
    lookback_years: int = 15,
    benchmark_name: str = "SPY",
    minimum_win_rate: float = 0.70,
    minimum_sample_size: int = 10,
    minimum_median_return: float = 0.0,
    minimum_beat_benchmark_rate: float = 0.60,
    minimum_median_excess_return: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if holding_periods is None:
        holding_periods = [
            5,
            10,
            15,
            20,
            30,
        ]

    candidate_dates = generate_upcoming_dates(
        start_date=start_date,
        days_ahead=days_ahead,
    )

    print(
        f"Loading {len(tickers)} tickers..."
    )

    price_map, loading_errors = load_price_map(
        tickers
    )

    print(
        f"Loaded {len(price_map)} tickers successfully."
    )

    benchmark_prices = get_price_history(
        benchmark_name
    )

    all_results = []
    all_errors = []

    if not loading_errors.empty:
        all_errors.append(loading_errors)

    total_windows = (
        len(candidate_dates)
        * len(holding_periods)
    )

    window_number = 0

    for candidate_date in candidate_dates:

        for holding_days in holding_periods:

            window_number += 1

            print(
                f"[{window_number}/{total_windows}] "
                f"{candidate_date} "
                f"+ {holding_days} sessions"
            )

            results, errors = scan_loaded_tickers(
                price_map=price_map,
                benchmark_prices=benchmark_prices,

                entry_month=candidate_date.month,
                entry_day=candidate_date.day,

                holding_days=holding_days,
                lookback_years=lookback_years,

                benchmark_name=benchmark_name,

                minimum_win_rate=minimum_win_rate,
                minimum_sample_size=minimum_sample_size,

                minimum_median_return=(
                    minimum_median_return
                ),

                minimum_beat_benchmark_rate=(
                    minimum_beat_benchmark_rate
                ),

                minimum_median_excess_return=(
                    minimum_median_excess_return
                ),
            )

            if not results.empty:

                results = results.copy()

                results[
                    "Upcoming Entry Date"
                ] = pd.Timestamp(
                    candidate_date
                )

                all_results.append(results)

            if not errors.empty:

                errors = errors.copy()

                errors[
                    "Entry Date Tested"
                ] = pd.Timestamp(
                    candidate_date
                )

                errors[
                    "Holding Days Tested"
                ] = holding_days

                all_errors.append(errors)

    if all_results:

        opportunities = pd.concat(
            all_results,
            ignore_index=True,
        )

    else:

        opportunities = pd.DataFrame()

    if all_errors:

        error_df = pd.concat(
            all_errors,
            ignore_index=True,
        )

    else:

        error_df = pd.DataFrame()

    if opportunities.empty:

        return opportunities, error_df

    opportunities = opportunities.sort_values(
        by=[
            "Wilson Lower Bound",
            f"Beat {benchmark_name} Rate",
            "Median Excess Return",
            "Median Return",
        ],
        ascending=False,
    ).reset_index(drop=True)

    # scan_loaded_tickers creates local ranks for
    # each individual window. Remove those before
    # assigning the global rank.
    if "Rank" in opportunities.columns:
        opportunities = opportunities.drop(
            columns=["Rank"]
        )

    opportunities.insert(
        0,
        "Rank",
        range(
            1,
            len(opportunities) + 1,
        ),
    )

    return opportunities, error_df