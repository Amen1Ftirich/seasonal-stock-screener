from __future__ import annotations

import pandas as pd


def add_benchmark_returns(
    seasonal_returns: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    benchmark_name: str = "SPY",
    price_column: str = "Adj Close",
) -> pd.DataFrame:
    """
    Add benchmark and excess returns to seasonal observations.

    The benchmark uses the exact same entry and exit trading dates
    as the stock.
    """

    if seasonal_returns.empty:
        return seasonal_returns.copy()

    required_benchmark_columns = {"Date", price_column}
    missing = required_benchmark_columns.difference(
        benchmark_prices.columns
    )

    if missing:
        raise ValueError(
            f"Benchmark data missing columns: {sorted(missing)}"
        )

    benchmark = benchmark_prices[
        ["Date", price_column]
    ].copy()

    benchmark["Date"] = pd.to_datetime(
        benchmark["Date"]
    )

    benchmark = (
        benchmark
        .dropna()
        .drop_duplicates("Date")
        .sort_values("Date")
        .set_index("Date")
    )

    results = seasonal_returns.copy()

    benchmark_returns = []

    for _, row in results.iterrows():

        entry_date = pd.Timestamp(row["Entry Date"])
        exit_date = pd.Timestamp(row["Exit Date"])

        if (
            entry_date not in benchmark.index
            or exit_date not in benchmark.index
        ):
            benchmark_returns.append(float("nan"))
            continue

        entry_price = float(
            benchmark.loc[entry_date, price_column]
        )

        exit_price = float(
            benchmark.loc[exit_date, price_column]
        )

        benchmark_return = (
            exit_price / entry_price
        ) - 1

        benchmark_returns.append(
            benchmark_return
        )

    benchmark_column = f"{benchmark_name} Return"

    results[benchmark_column] = benchmark_returns

    results["Excess Return"] = (
        results["Return"]
        - results[benchmark_column]
    )

    results[f"Beat {benchmark_name}"] = (
        results["Excess Return"] > 0
    )

    return results