from __future__ import annotations

from math import sqrt

import pandas as pd


def wilson_lower_bound(
    wins: int,
    total: int,
    z: float = 1.96,
) -> float:
    """
    Conservative lower bound for a binomial win rate.

    z = 1.96 corresponds roughly to a 95% confidence interval.
    """

    if total <= 0:
        return 0.0

    p = wins / total

    denominator = 1 + (z**2 / total)

    center = p + (z**2 / (2 * total))

    adjustment = z * sqrt(
        (p * (1 - p) / total)
        + (z**2 / (4 * total**2))
    )

    return (center - adjustment) / denominator


def calculate_metrics(
    seasonal_returns: pd.DataFrame,
) -> dict:
    """
    Calculate summary statistics from get_seasonal_returns().
    """

    if seasonal_returns.empty:
        return {
            "sample_size": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "average_return": 0.0,
            "median_return": 0.0,
            "std_dev": 0.0,
            "best_return": 0.0,
            "worst_return": 0.0,
            "average_gain": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "wilson_lower_bound": 0.0,
        }

    returns = seasonal_returns["Return"].dropna()

    if returns.empty:
        raise ValueError("No valid return observations found")

    total = len(returns)

    winners = returns[returns > 0]
    losers = returns[returns <= 0]

    wins = len(winners)
    losses = len(losers)

    win_rate = wins / total

    average_gain = (
        float(winners.mean())
        if not winners.empty
        else 0.0
    )

    average_loss = (
        float(losers.mean())
        if not losers.empty
        else 0.0
    )

    gross_profit = float(winners.sum())

    gross_loss = abs(float(losers.sum()))

    if gross_loss == 0:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    return {
        "sample_size": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "average_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "std_dev": float(returns.std(ddof=1)) if total > 1 else 0.0,
        "best_return": float(returns.max()),
        "worst_return": float(returns.min()),
        "average_gain": average_gain,
        "average_loss": average_loss,
        "profit_factor": profit_factor,
        "wilson_lower_bound": wilson_lower_bound(
            wins=wins,
            total=total,
        ),
    }


def metrics_to_series(metrics: dict) -> pd.Series:
    """
    Convenience function for scanner.py later.
    """

    return pd.Series(metrics)
def calculate_relative_metrics(
    seasonal_returns: pd.DataFrame,
    benchmark_name: str = "SPY",
) -> dict:
    """
    Calculate performance relative to a benchmark.
    """

    benchmark_column = f"{benchmark_name} Return"
    beat_column = f"Beat {benchmark_name}"

    required = {
        benchmark_column,
        "Excess Return",
        beat_column,
    }

    missing = required.difference(
        seasonal_returns.columns
    )

    if missing:
        raise ValueError(
            f"Missing benchmark columns: {sorted(missing)}"
        )

    valid = seasonal_returns.dropna(
        subset=[
            benchmark_column,
            "Excess Return",
        ]
    )

    if valid.empty:
        return {
            "benchmark_sample_size": 0,
            "beat_benchmark_rate": 0.0,
            "average_excess_return": 0.0,
            "median_excess_return": 0.0,
            "best_excess_return": 0.0,
            "worst_excess_return": 0.0,
        }

    excess = valid["Excess Return"]

    return {
        "benchmark_sample_size": len(valid),
        "beat_benchmark_rate": float(
            valid[beat_column].mean()
        ),
        "average_excess_return": float(
            excess.mean()
        ),
        "median_excess_return": float(
            excess.median()
        ),
        "best_excess_return": float(
            excess.max()
        ),
        "worst_excess_return": float(
            excess.min()
        ),
    }