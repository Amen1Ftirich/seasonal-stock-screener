from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.benchmark import add_benchmark_returns
from src.data import get_price_history, load_price_map
from src.dedup import deduplicate_opportunities
from src.scanner import scan_loaded_tickers
from src.seasonality import get_seasonal_returns


def _prices_before_year(
    prices: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """
    Return only information that existed before January 1
    of the specified test year.
    """

    data = prices.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    cutoff = pd.Timestamp(
        year=year,
        month=1,
        day=1,
    )

    return (
        data[data["Date"] < cutoff]
        .copy()
        .reset_index(drop=True)
    )


def _generate_historical_candidate_dates(
    test_year: int,
    anchor_month: int,
    anchor_day: int,
    days_ahead: int,
) -> list[date]:
    """
    Recreate today's upcoming search window inside
    a historical test year.
    """

    anchor = date(
        test_year,
        anchor_month,
        anchor_day,
    )

    return [
        anchor + timedelta(days=offset)
        for offset in range(days_ahead + 1)
    ]


def _evaluate_candidate_in_year(
    ticker: str,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    test_year: int,
    entry_month: int,
    entry_day: int,
    holding_days: int,
    benchmark_name: str = "SPY",
) -> dict | None:
    """
    Evaluate one previously selected seasonal candidate
    during the unseen test year.
    """

    cutoff = pd.Timestamp(
        year=test_year + 1,
        month=1,
        day=1,
    )

    test_available_prices = prices[
        pd.to_datetime(prices["Date"]) < cutoff
    ].copy()

    seasonal = get_seasonal_returns(
        prices=test_available_prices,
        entry_month=entry_month,
        entry_day=entry_day,
        holding_days=holding_days,
        lookback_years=1000,
    )

    if seasonal.empty:
        return None

    test_observation = seasonal[
        seasonal["Year"] == test_year
    ].copy()

    if test_observation.empty:
        return None

    comparison = add_benchmark_returns(
        seasonal_returns=test_observation,
        benchmark_prices=benchmark_prices,
        benchmark_name=benchmark_name,
    )

    if comparison.empty:
        return None

    row = comparison.iloc[0]

    return {
        "Ticker": ticker,
        "Test Year": test_year,

        "Entry Date":
            row["Entry Date"],

        "Exit Date":
            row["Exit Date"],

        "Holding Days":
            holding_days,

        "Realized Return":
            float(row["Return"]),

        "Realized Benchmark Return":
            float(row[f"{benchmark_name} Return"]),

        "Realized Excess Return":
            float(row["Excess Return"]),

        "Realized Win":
            bool(row["Return"] > 0),

        f"Realized Beat {benchmark_name}":
            bool(row[f"Beat {benchmark_name}"]),
    }


def nested_walk_forward_discovery(
    tickers: list[str],
    anchor_month: int,
    anchor_day: int,
    test_years: list[int],
    days_ahead: int = 7,
    holding_periods: list[int] | None = None,
    lookback_years: int = 15,
    top_n: int = 3,
    benchmark_name: str = "SPY",

    minimum_win_rate: float = 0.65,
    minimum_sample_size: int = 8,
    minimum_median_return: float = 0.0,
    minimum_beat_benchmark_rate: float = 0.55,
    minimum_median_excess_return: float = 0.0,

) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Simulate the complete historical screener.

    For every test year:

        1. Hide that year and everything after it.
        2. Search upcoming seasonal windows using only past data.
        3. Rank candidates.
        4. Select the best opportunities.
        5. Reveal the test year.
        6. Measure what actually happened.

    This is substantially stronger than validating a window
    discovered using the complete dataset.
    """

    if holding_periods is None:
        holding_periods = [
            5,
            10,
            15,
            20,
            30,
        ]

    full_price_map, loading_errors = load_price_map(
        tickers
    )

    full_benchmark = get_price_history(
        benchmark_name
    )

    all_realized = []
    all_errors = []

    if not loading_errors.empty:
        all_errors.append(
            loading_errors.copy()
        )

    for test_year in test_years:

        print()
        print(
            f"===== NESTED TEST YEAR {test_year} ====="
        )

        # ---------------------------------------
        # Build historical information set
        # ---------------------------------------

        training_price_map = {}

        for ticker, prices in full_price_map.items():

            historical = _prices_before_year(
                prices=prices,
                year=test_year,
            )

            if not historical.empty:
                training_price_map[ticker] = historical

        training_benchmark = _prices_before_year(
            prices=full_benchmark,
            year=test_year,
        )

        candidate_dates = (
            _generate_historical_candidate_dates(
                test_year=test_year,
                anchor_month=anchor_month,
                anchor_day=anchor_day,
                days_ahead=days_ahead,
            )
        )

        candidate_results = []

        # ---------------------------------------
        # Historical discovery
        # ---------------------------------------

        for candidate_date in candidate_dates:

            for holding_days in holding_periods:

                results, errors = scan_loaded_tickers(
                    price_map=training_price_map,
                    benchmark_prices=training_benchmark,

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
                        "Historical Entry Candidate"
                    ] = pd.Timestamp(
                        candidate_date
                    )

                    candidate_results.append(
                        results
                    )

                if not errors.empty:

                    errors = errors.copy()

                    errors["Test Year"] = test_year

                    all_errors.append(errors)

        if not candidate_results:

            print(
                f"No candidates qualified for {test_year}"
            )

            continue

        candidates = pd.concat(
            candidate_results,
            ignore_index=True,
        )

        # Remove each individual window's local rank.
        if "Rank" in candidates.columns:
            candidates = candidates.drop(
                columns=["Rank"]
            )

        # ---------------------------------------
        # Rank using ONLY information that
        # existed before the test year.
        # ---------------------------------------

        candidates = candidates.sort_values(
            by=[
                "Train Excess LCB",
                "Train Median Excess",
                "Train Profit Factor",
                "Train Wilson",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        ).reset_index(drop=True)
        
        candidates.insert(
            0,
            "Rank",
            range(
                1,
                len(candidates) + 1,
            ),
        )

        # The dedup module expects this name.
        candidates[
            "Upcoming Entry Date"
        ] = candidates[
            "Historical Entry Candidate"
        ]

        candidates = deduplicate_opportunities(
            candidates,
            entry_tolerance_days=3,
            holding_tolerance_days=5,
        )
        # Only allow one seasonal setup per stock
        # in each historical selection period.
        candidates = (
            candidates
            .drop_duplicates(
                subset=["Ticker"],
                keep="first",
            )
            .reset_index(drop=True)
        )
        selected = candidates.head(
            top_n
        )

        print(
            f"Selected {len(selected)} opportunities"
        )

        # ---------------------------------------
        # Reveal the unseen year
        # ---------------------------------------

        for _, candidate in selected.iterrows():

            ticker = candidate["Ticker"]

            result = _evaluate_candidate_in_year(
                ticker=ticker,

                prices=full_price_map[ticker],

                benchmark_prices=full_benchmark,

                test_year=test_year,

                entry_month=int(
                    candidate["Entry Month"]
                ),

                entry_day=int(
                    candidate["Entry Day"]
                ),

                holding_days=int(
                    candidate["Holding Days"]
                ),

                benchmark_name=benchmark_name,
            )

            if result is None:
                continue

            # Save what the algorithm believed
            # BEFORE seeing the test year.

            result["Prior Train Win Rate"] = (
                candidate["Train Win Rate"]
            )

            result["Prior Train Wilson"] = (
                candidate["Train Wilson"]
            )

            result["Prior Median Return"] = (
                candidate["Train Median Return"]
            )

            result["Prior Median Excess"] = (
                candidate["Train Median Excess"]
            )

            result["Prior Rank"] = int(
                candidate["Rank"]
            )

            all_realized.append(result)

    # -------------------------------------------
    # Build final backtest results
    # -------------------------------------------

    realized = pd.DataFrame(
        all_realized
    )

    if all_errors:

        error_df = pd.concat(
            all_errors,
            ignore_index=True,
        )

    else:

        error_df = pd.DataFrame()

    if realized.empty:

        return (
            realized,
            {
                "Trades": 0,
                "Win Rate": 0.0,
                "Beat Benchmark Rate": 0.0,
                "Average Return": 0.0,
                "Median Return": 0.0,
                "Average Excess Return": 0.0,
                "Median Excess Return": 0.0,
                "Worst Return": 0.0,
            },
            error_df,
        )

    summary = {

        "Trades":
            len(realized),

        "Win Rate":
            float(
                realized[
                    "Realized Win"
                ].mean()
            ),

        "Beat Benchmark Rate":
            float(
                realized[
                    f"Realized Beat {benchmark_name}"
                ].mean()
            ),

        "Average Return":
            float(
                realized[
                    "Realized Return"
                ].mean()
            ),

        "Median Return":
            float(
                realized[
                    "Realized Return"
                ].median()
            ),

        "Average Excess Return":
            float(
                realized[
                    "Realized Excess Return"
                ].mean()
            ),

        "Median Excess Return":
            float(
                realized[
                    "Realized Excess Return"
                ].median()
            ),

        "Worst Return":
            float(
                realized[
                    "Realized Return"
                ].min()
            ),
    }

    return (
        realized,
        summary,
        error_df,
    )