from __future__ import annotations

import numpy as np
import pandas as pd


def select_portfolio(
    cross_section: pd.DataFrame,
    strategy: str,
) -> pd.Series:
    """
    Generate target portfolio weights from
    frozen V8C predictions.

    Strategies:

        Top Decile EW
        Top Quintile EW
        Ex Bottom Quintile EW
    """

    data = cross_section.dropna(
        subset=[
            "Ticker",
            "V8C Prediction",
        ]
    ).copy()


    if data.empty:

        return pd.Series(
            dtype=float
        )


    data[
        "Percentile"
    ] = (
        data[
            "V8C Prediction"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )


    if strategy == "Top Decile EW":

        selected = data[
            data[
                "Percentile"
            ] > 0.90
        ].copy()


    elif strategy == "Top Quintile EW":

        selected = data[
            data[
                "Percentile"
            ] > 0.80
        ].copy()


    elif strategy == "Ex Bottom Quintile EW":

        selected = data[
            data[
                "Percentile"
            ] > 0.20
        ].copy()


    else:

        raise ValueError(
            f"Unknown strategy: {strategy}"
        )


    if selected.empty:

        return pd.Series(
            dtype=float
        )


    weight = (
        1.0
        / len(selected)
    )


    return pd.Series(
        weight,
        index=selected[
            "Ticker"
        ],
        dtype=float,
    )


def calculate_end_weights(
    target_weights: pd.Series,
    returns: pd.Series,
) -> pd.Series:

    returns = returns.reindex(
        target_weights.index
    ).fillna(
        0.0
    )


    ending_values = (
        target_weights
        * (
            1.0
            + returns
        )
    )


    total = ending_values.sum()


    if total <= 0:

        return target_weights.copy()


    return (
        ending_values
        / total
    )


def calculate_gross_traded_notional(
    new_weights: pd.Series,
    old_weights: pd.Series | None,
) -> float:
    """
    Sum of absolute trades.

    Example:

        0.40 means trades equal to 40%
        of portfolio value.

    Buys and sells are both counted.
    """

    if old_weights is None:

        return 1.0


    tickers = (
        new_weights.index
        .union(
            old_weights.index
        )
    )


    new = new_weights.reindex(
        tickers,
        fill_value=0.0,
    )


    old = old_weights.reindex(
        tickers,
        fill_value=0.0,
    )


    return float(
        (
            new
            - old
        )
        .abs()
        .sum()
    )


def run_v8c_portfolios(
    predictions: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:

    data = predictions.copy()


    period_column = (
        "Test Period"
        if "Test Period" in data.columns
        else "Period"
    )


    data[
        period_column
    ] = pd.PeriodIndex(
        data[
            period_column
        ],
        freq="M",
    )


    required = [
        "Ticker",
        "V8C Prediction",
        "Return",
        "Benchmark Return",
        "Market Beta 24M",
    ]


    data = data.dropna(
        subset=required
    ).copy()


    strategies = [
        "Top Decile EW",
        "Top Quintile EW",
        "Ex Bottom Quintile EW",
    ]


    previous_end_weights = {
        strategy:
            None

        for strategy
        in strategies
    }


    cost_rate = (
        transaction_cost_bps
        / 10000.0
    )


    monthly_rows = []


    for period, cross_section in (
        data.groupby(
            period_column
        )
    ):

        cross_section = (
            cross_section
            .drop_duplicates(
                subset=[
                    "Ticker",
                ]
            )
            .copy()
        )


        if len(
            cross_section
        ) < 100:

            continue


        indexed = (
            cross_section
            .set_index(
                "Ticker"
            )
        )


        benchmark_return = float(
            cross_section[
                "Benchmark Return"
            ].iloc[0]
        )


        for strategy in strategies:

            weights = (
                select_portfolio(
                    cross_section=(
                        cross_section
                    ),
                    strategy=strategy,
                )
            )


            if weights.empty:
                continue


            returns = (
                indexed.loc[
                    weights.index,
                    "Return",
                ]
                .astype(float)
            )


            betas = (
                indexed.loc[
                    weights.index,
                    "Market Beta 24M",
                ]
                .astype(float)
            )


            gross_return = float(
                np.dot(
                    weights,
                    returns,
                )
            )


            estimated_beta = float(
                np.dot(
                    weights,
                    betas,
                )
            )


            gross_traded = (
                calculate_gross_traded_notional(
                    new_weights=weights,
                    old_weights=(
                        previous_end_weights[
                            strategy
                        ]
                    ),
                )
            )


            trading_cost = (
                gross_traded
                * cost_rate
            )


            net_return = (
                gross_return
                - trading_cost
            )


            effective_holdings = float(
                1.0
                / np.sum(
                    weights.to_numpy()
                    ** 2
                )
            )


            monthly_rows.append(
                {
                    "Period":
                        period,

                    "Strategy":
                        strategy,

                    "Gross Return":
                        gross_return,

                    "Net Return":
                        net_return,

                    "Benchmark Return":
                        benchmark_return,

                    "Net Excess Return":
                        (
                            net_return
                            - benchmark_return
                        ),

                    "Beat SPY":
                        (
                            net_return
                            > benchmark_return
                        ),

                    "Estimated Beta":
                        estimated_beta,

                    "Gross Traded Notional":
                        gross_traded,

                    "Trading Cost":
                        trading_cost,

                    "Holdings":
                        len(weights),

                    "Effective Holdings":
                        effective_holdings,

                    "Maximum Weight":
                        float(
                            weights.max()
                        ),
                }
            )


            previous_end_weights[
                strategy
            ] = (
                calculate_end_weights(
                    target_weights=weights,
                    returns=returns,
                )
            )


    return pd.DataFrame(
        monthly_rows
    )


def calculate_cagr(
    returns: pd.Series,
) -> float:

    returns = (
        returns
        .dropna()
        .astype(float)
    )


    if returns.empty:

        return np.nan


    wealth = float(
        (
            1.0
            + returns
        ).prod()
    )


    years = (
        len(returns)
        / 12.0
    )


    if (
        wealth <= 0
        or years <= 0
    ):

        return np.nan


    return (
        wealth
        ** (
            1.0 / years
        )
        - 1.0
    )


def calculate_max_drawdown(
    returns: pd.Series,
) -> float:

    wealth = (
        1.0
        + returns
    ).cumprod()


    peak = wealth.cummax()


    drawdown = (
        wealth
        / peak
        - 1.0
    )


    return float(
        drawdown.min()
    )


def summarize_v8c_portfolios(
    monthly: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    for strategy, group in (
        monthly.groupby(
            "Strategy"
        )
    ):

        group = group.sort_values(
            "Period"
        )


        returns = group[
            "Net Return"
        ]


        spy = group[
            "Benchmark Return"
        ]


        excess = group[
            "Net Excess Return"
        ]


        annual_vol = float(
            returns.std(
                ddof=1
            )
            * np.sqrt(12)
        )


        annual_mean = float(
            returns.mean()
            * 12
        )


        sharpe = (
            annual_mean
            / annual_vol

            if annual_vol > 0

            else np.nan
        )


        spy_variance = float(
            spy.var(
                ddof=1
            )
        )


        realized_beta = (
            float(
                returns.cov(
                    spy
                )
                / spy_variance
            )

            if spy_variance > 0

            else np.nan
        )


        rows.append(
            {
                "Strategy":
                    strategy,

                "Months":
                    len(group),

                "Net CAGR":
                    calculate_cagr(
                        returns
                    ),

                "SPY CAGR":
                    calculate_cagr(
                        spy
                    ),

                "Annualized Volatility":
                    annual_vol,

                "Sharpe RF=0":
                    sharpe,

                "Maximum Drawdown":
                    calculate_max_drawdown(
                        returns
                    ),

                "Average Monthly Excess":
                    float(
                        excess.mean()
                    ),

                "Beat SPY Rate":
                    float(
                        group[
                            "Beat SPY"
                        ].mean()
                    ),

                "Average Estimated Beta":
                    float(
                        group[
                            "Estimated Beta"
                        ].mean()
                    ),

                "Realized Beta":
                    realized_beta,

                "Average Gross Turnover":
                    float(
                        group[
                            "Gross Traded Notional"
                        ].mean()
                    ),

                "Average Monthly Cost":
                    float(
                        group[
                            "Trading Cost"
                        ].mean()
                    ),

                "Average Holdings":
                    float(
                        group[
                            "Holdings"
                        ].mean()
                    ),

                "Average Effective Holdings":
                    float(
                        group[
                            "Effective Holdings"
                        ].mean()
                    ),

                "Average Maximum Weight":
                    float(
                        group[
                            "Maximum Weight"
                        ].mean()
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )