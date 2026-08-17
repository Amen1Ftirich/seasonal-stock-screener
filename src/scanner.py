from __future__ import annotations

import pandas as pd

from src.benchmark import add_benchmark_returns
from src.data import get_price_history
from src.metrics import (
    calculate_metrics,
    calculate_relative_metrics,
)
from src.seasonality import get_seasonal_returns


def analyze_ticker(
    ticker: str,
    benchmark_prices: pd.DataFrame,
    entry_month: int,
    entry_day: int,
    holding_days: int,
    lookback_years: int = 15,
    benchmark_name: str = "SPY",
) -> dict:
    """
    Run the complete seasonal analysis for one ticker.
    """

    ticker = ticker.upper().strip()

    prices = get_price_history(ticker)

    seasonal = get_seasonal_returns(
        prices=prices,
        entry_month=entry_month,
        entry_day=entry_day,
        holding_days=holding_days,
        lookback_years=lookback_years,
    )

    if seasonal.empty:
        raise ValueError(
            f"No valid seasonal observations for {ticker}"
        )

    comparison = add_benchmark_returns(
        seasonal_returns=seasonal,
        benchmark_prices=benchmark_prices,
        benchmark_name=benchmark_name,
    )

    absolute = calculate_metrics(comparison)

    relative = calculate_relative_metrics(
        comparison,
        benchmark_name=benchmark_name,
    )

    return {
        "Ticker": ticker,
        "Entry Month": entry_month,
        "Entry Day": entry_day,
        "Holding Days": holding_days,

        "Sample Size": absolute["sample_size"],
        "Wins": absolute["wins"],
        "Losses": absolute["losses"],

        "Win Rate": absolute["win_rate"],
        "Average Return": absolute["average_return"],
        "Median Return": absolute["median_return"],
        "Std Dev": absolute["std_dev"],

        "Best Return": absolute["best_return"],
        "Worst Return": absolute["worst_return"],

        "Average Gain": absolute["average_gain"],
        "Average Loss": absolute["average_loss"],
        "Profit Factor": absolute["profit_factor"],

        "Wilson Lower Bound":
            absolute["wilson_lower_bound"],

        f"Beat {benchmark_name} Rate":
            relative["beat_benchmark_rate"],

        "Average Excess Return":
            relative["average_excess_return"],

        "Median Excess Return":
            relative["median_excess_return"],

        "Best Excess Return":
            relative["best_excess_return"],

        "Worst Excess Return":
            relative["worst_excess_return"],
    }


def scan_tickers(
    tickers: list[str],
    entry_month: int,
    entry_day: int,
    holding_days: int,
    lookback_years: int = 15,
    benchmark_name: str = "SPY",
    minimum_win_rate: float = 0.0,
    minimum_sample_size: int = 0,
    minimum_median_return: float | None = None,
    minimum_beat_benchmark_rate: float | None = None,
    minimum_median_excess_return: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan multiple tickers and return:

    1. ranked qualifying opportunities
    2. tickers that failed during processing
    """

    benchmark_prices = get_price_history(
        benchmark_name
    )

    rows = []
    errors = []

    cleaned_tickers = sorted(
        {
            ticker.upper().strip()
            for ticker in tickers
            if ticker.strip()
        }
    )

    for ticker in cleaned_tickers:

        try:
            result = analyze_ticker(
                ticker=ticker,
                benchmark_prices=benchmark_prices,
                entry_month=entry_month,
                entry_day=entry_day,
                holding_days=holding_days,
                lookback_years=lookback_years,
                benchmark_name=benchmark_name,
            )

            rows.append(result)

        except Exception as exc:

            errors.append(
                {
                    "Ticker": ticker,
                    "Error": str(exc),
                }
            )

    results = pd.DataFrame(rows)

    error_df = pd.DataFrame(errors)

    if results.empty:
        return results, error_df

    results = results[
        results["Sample Size"]
        >= minimum_sample_size
    ]

    results = results[
        results["Win Rate"]
        >= minimum_win_rate
    ]

    if minimum_median_return is not None:

        results = results[
            results["Median Return"]
            >= minimum_median_return
        ]

    if minimum_beat_benchmark_rate is not None:

        column = f"Beat {benchmark_name} Rate"

        results = results[
            results[column]
            >= minimum_beat_benchmark_rate
        ]

    if minimum_median_excess_return is not None:

        results = results[
            results["Median Excess Return"]
            >= minimum_median_excess_return
        ]

    # Rank statistical reliability first,
    # then market-relative performance.
    results = results.sort_values(
        by=[
            "Wilson Lower Bound",
            f"Beat {benchmark_name} Rate",
            "Median Excess Return",
            "Median Return",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    results = results.reset_index(drop=True)

    results.insert(
        0,
        "Rank",
        range(1, len(results) + 1),
    )

    return results, error_df