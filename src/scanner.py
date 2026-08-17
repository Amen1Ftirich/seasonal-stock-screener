from __future__ import annotations

import pandas as pd
from src.validation import calculate_validation_metrics
from src.benchmark import add_benchmark_returns
from src.data import get_price_history
from src.metrics import (
    calculate_metrics,
    calculate_relative_metrics,
)
from src.seasonality import get_seasonal_returns


def analyze_prices(
    ticker: str,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    entry_month: int,
    entry_day: int,
    holding_days: int,
    lookback_years: int = 15,
    benchmark_name: str = "SPY",
) -> dict:
    """
    Analyze already-loaded price data.

    This is the fast path used by the discovery engine.
    """

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

    validation = calculate_validation_metrics(
    observations=comparison,
    benchmark_name=benchmark_name,
    test_years=5,
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
                "Train Sample Size":
            validation["Train Sample Size"],

        "Train Win Rate":
            validation["Train Win Rate"],

        "Train Median Return":
            validation["Train Median Return"],

        "Train Wilson":
            validation["Train Wilson"],

        "Train Beat SPY Rate":
            validation["Train Beat SPY Rate"],

        "Train Median Excess":
            validation["Train Median Excess"],


        "OOS Sample Size":
            validation["OOS Sample Size"],

        "OOS Win Rate":
            validation["OOS Win Rate"],

        "OOS Median Return":
            validation["OOS Median Return"],

        "OOS Beat SPY Rate":
            validation["OOS Beat SPY Rate"],

        "OOS Median Excess":
            validation["OOS Median Excess"],
    }


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
    Convenience wrapper for analyzing one ticker from disk/cache.
    """

    ticker = ticker.upper().strip()

    prices = get_price_history(ticker)

    return analyze_prices(
        ticker=ticker,
        prices=prices,
        benchmark_prices=benchmark_prices,
        entry_month=entry_month,
        entry_day=entry_day,
        holding_days=holding_days,
        lookback_years=lookback_years,
        benchmark_name=benchmark_name,
    )


def passes_filters(
    result: dict,
    benchmark_name: str,
    minimum_win_rate: float,
    minimum_sample_size: int,
    minimum_median_return: float | None,
    minimum_beat_benchmark_rate: float | None,
    minimum_median_excess_return: float | None,
) -> bool:

    if result["Train Sample Size"] < minimum_sample_size:
        return False

    if result["Train Win Rate"] < minimum_win_rate:
        return False

    if (
        minimum_median_return is not None
        and result["Train Median Return"]
        < minimum_median_return
    ):
        return False

    if (
        minimum_beat_benchmark_rate is not None
        and result["Train Beat SPY Rate"]
        < minimum_beat_benchmark_rate
    ):
        return False

    if (
        minimum_median_excess_return is not None
        and result["Train Median Excess"]
        < minimum_median_excess_return
    ):
        return False

    return True


def scan_loaded_tickers(
    price_map: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
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
    Scan stocks that have already been loaded into memory.
    """

    rows = []
    errors = []

    for ticker, prices in price_map.items():

        try:
            result = analyze_prices(
                ticker=ticker,
                prices=prices,
                benchmark_prices=benchmark_prices,
                entry_month=entry_month,
                entry_day=entry_day,
                holding_days=holding_days,
                lookback_years=lookback_years,
                benchmark_name=benchmark_name,
            )

            if passes_filters(
                result=result,
                benchmark_name=benchmark_name,
                minimum_win_rate=minimum_win_rate,
                minimum_sample_size=minimum_sample_size,
                minimum_median_return=minimum_median_return,
                minimum_beat_benchmark_rate=(
                    minimum_beat_benchmark_rate
                ),
                minimum_median_excess_return=(
                    minimum_median_excess_return
                ),
            ):
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

    results = results.sort_values(
        by=[
        "Train Wilson",
        "Train Beat SPY Rate",
        "Train Median Excess",
        "Train Median Return",
    ],
    ascending=[
        False,
        False,
        False,
        False,
    ],
    ).reset_index(drop=True)

    results.insert(
        0,
        "Rank",
        range(1, len(results) + 1),
    )

    return results, error_df


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
    Normal scanner interface.

    Loads each ticker once, then uses the in-memory scanner.
    """

    benchmark_prices = get_price_history(
        benchmark_name
    )

    price_map = {}
    loading_errors = []

    cleaned_tickers = sorted(
        {
            ticker.upper().strip()
            for ticker in tickers
            if ticker.strip()
        }
    )

    for ticker in cleaned_tickers:

        try:
            price_map[ticker] = get_price_history(
                ticker
            )

        except Exception as exc:
            loading_errors.append(
                {
                    "Ticker": ticker,
                    "Error": str(exc),
                }
            )

    results, scan_errors = scan_loaded_tickers(
        price_map=price_map,
        benchmark_prices=benchmark_prices,
        entry_month=entry_month,
        entry_day=entry_day,
        holding_days=holding_days,
        lookback_years=lookback_years,
        benchmark_name=benchmark_name,
        minimum_win_rate=minimum_win_rate,
        minimum_sample_size=minimum_sample_size,
        minimum_median_return=minimum_median_return,
        minimum_beat_benchmark_rate=(
            minimum_beat_benchmark_rate
        ),
        minimum_median_excess_return=(
            minimum_median_excess_return
        ),
    )

    all_errors = []

    if loading_errors:
        all_errors.append(
            pd.DataFrame(loading_errors)
        )

    if not scan_errors.empty:
        all_errors.append(scan_errors)

    if all_errors:
        error_df = pd.concat(
            all_errors,
            ignore_index=True,
        )
    else:
        error_df = pd.DataFrame()

    return results, error_df