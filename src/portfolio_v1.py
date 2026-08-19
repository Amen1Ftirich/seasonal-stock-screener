from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.optimize import (
    linprog,
    minimize,
)


def make_rank_tilt_weights(
    cross_section: pd.DataFrame,
    bottom_fraction: float = 0.10,
) -> pd.Series:
    """
    Exclude the predicted bottom decile.

    Among the remaining stocks, weight continuously
    toward stronger V5 predictions.

    No top-N cutoff is used.
    """

    data = cross_section.copy()

    data[
        "Signal Percentile"
    ] = data[
        "V5 Prediction"
    ].rank(
        pct=True,
        method="average",
    )

    data = data[
        data[
            "Signal Percentile"
        ] > bottom_fraction
    ].copy()

    if data.empty:
        return pd.Series(
            dtype=float
        )

    #
    # Bottom accepted stock receives almost zero
    # signal weight.
    #
    # Stronger stocks receive progressively more.
    #

    data["Raw Weight"] = (
        data["Signal Percentile"]
        - bottom_fraction
    )

    total = data[
        "Raw Weight"
    ].sum()

    if total <= 0:
        return pd.Series(
            dtype=float
        )

    data["Weight"] = (
        data["Raw Weight"]
        / total
    )

    return data.set_index(
        "Ticker"
    )["Weight"]


def beta_target_weights(
    cross_section: pd.DataFrame,
    base_weights: pd.Series,
    target_beta: float = 1.0,
    max_weight: float = 0.02,
) -> pd.Series:
    """
    Stay as close as possible to the rank-tilt
    portfolio while targeting market beta near 1.

    Constraints:

        weights sum to 1
        no shorting
        max 2% per stock
        portfolio beta = target when feasible
    """

    if base_weights.empty:
        return base_weights

    data = (
        cross_section
        .set_index("Ticker")
        .loc[
            base_weights.index
        ]
        .copy()
    )

    valid = data[
        "Market Beta 24M"
    ].notna()

    data = data[
        valid
    ].copy()

    base = base_weights.loc[
        data.index
    ].copy()

    if base.empty:
        return pd.Series(
            dtype=float
        )

    base = (
        base
        / base.sum()
    )

    betas = (
        data[
            "Market Beta 24M"
        ]
        .astype(float)
        .to_numpy()
    )

    base_array = (
        base
        .astype(float)
        .to_numpy()
    )

    n = len(base_array)

    #
    # A 2% cap requires at least 50 securities.
    #

    if (
        n * max_weight
        < 1.0
    ):

        max_weight = (
            1.0 / n
        ) + 1e-6

    bounds = [
        (
            0.0,
            max_weight,
        )
        for _ in range(n)
    ]

    #
    # First determine the feasible beta interval
    # under the long-only and max-weight constraints.
    #

    equality_matrix = np.ones(
        (
            1,
            n,
        )
    )

    equality_target = np.array(
        [1.0]
    )

    minimum_beta_result = linprog(
        c=betas,
        A_eq=equality_matrix,
        b_eq=equality_target,
        bounds=bounds,
        method="highs",
    )

    maximum_beta_result = linprog(
        c=-betas,
        A_eq=equality_matrix,
        b_eq=equality_target,
        bounds=bounds,
        method="highs",
    )

    if (
        not minimum_beta_result.success
        or not maximum_beta_result.success
    ):

        return base

    minimum_beta = float(
        minimum_beta_result.fun
    )

    maximum_beta = float(
        -maximum_beta_result.fun
    )

    feasible_target = float(
        np.clip(
            target_beta,
            minimum_beta,
            maximum_beta,
        )
    )

    #
    # Starting portfolio.
    #

    x0 = base_array.copy()

    #
    # Base rank weights are normally far below
    # the 2% cap, but enforce it defensively.
    #

    x0 = np.minimum(
        x0,
        max_weight,
    )

    x0 = (
        x0
        / x0.sum()
    )

    def objective(
        weights: np.ndarray,
    ) -> float:

        #
        # Change the signal portfolio as little
        # as possible.
        #

        return float(
            np.sum(
                (
                    weights
                    - base_array
                ) ** 2
            )
        )

    constraints = [
        {
            "type":
                "eq",

            "fun":
                lambda weights:
                    np.sum(weights)
                    - 1.0,
        },

        {
            "type":
                "eq",

            "fun":
                lambda weights:
                    np.dot(
                        weights,
                        betas,
                    )
                    - feasible_target,
        },
    ]

    result = minimize(
        objective,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        },
    )

    if not result.success:

        return base

    weights = pd.Series(
        result.x,
        index=data.index,
    )

    weights[
        weights < 1e-10
    ] = 0.0

    weights = (
        weights
        / weights.sum()
    )

    return weights


def calculate_turnover(
    target_weights: pd.Series,
    previous_end_weights: pd.Series | None,
) -> float:
    """
    Gross notional traded.

    1.00 means trades totaling 100% of portfolio
    value.

    This counts both purchases and sales.
    """

    if previous_end_weights is None:

        return 1.0

    tickers = (
        target_weights.index
        .union(
            previous_end_weights.index
        )
    )

    new = target_weights.reindex(
        tickers,
        fill_value=0.0,
    )

    old = previous_end_weights.reindex(
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


def calculate_end_weights(
    target_weights: pd.Series,
    realized_returns: pd.Series,
) -> pd.Series:
    """
    Drift target weights through the realized month.

    These become the starting holdings for calculating
    next month's rebalance turnover.
    """

    returns = realized_returns.reindex(
        target_weights.index
    ).fillna(
        0.0
    )

    values = (
        target_weights
        * (
            1.0
            + returns
        )
    )

    total = values.sum()

    if total <= 0:

        return target_weights.copy()

    return (
        values
        / total
    )


def run_portfolio_backtest(
    predictions: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build portfolios using SAVED V5 out-of-sample
    predictions.

    No model is retrained here.
    """

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
        "V5 Prediction",
        "Return",
        "Benchmark Return",
        "Market Beta 24M",
    ]

    data = data.dropna(
        subset=required
    )

    periods = sorted(
        data[
            period_column
        ].unique()
    )

    monthly_rows = []

    weight_rows = []

    previous_end_weights = {
        "Rank Tilt":
            None,

        "Beta Target":
            None,
    }

    cost_rate = (
        transaction_cost_bps
        / 10000.0
    )

    for period in periods:

        cross_section = data[
            data[
                period_column
            ] == period
        ].copy()

        if len(cross_section) < 50:
            continue

        base_weights = (
            make_rank_tilt_weights(
                cross_section
            )
        )

        if base_weights.empty:
            continue

        beta_weights = (
            beta_target_weights(
                cross_section=(
                    cross_section
                ),
                base_weights=(
                    base_weights
                ),
                target_beta=1.0,
                max_weight=0.02,
            )
        )

        strategy_weights = {
            "Rank Tilt":
                base_weights,

            "Beta Target":
                beta_weights,
        }

        indexed = (
            cross_section
            .drop_duplicates(
                subset=["Ticker"]
            )
            .set_index("Ticker")
        )

        benchmark_return = float(
            cross_section[
                "Benchmark Return"
            ].iloc[0]
        )

        for (
            strategy,
            weights,
        ) in strategy_weights.items():

            weights = weights[
                weights > 0
            ]

            returns = indexed.loc[
                weights.index,
                "Return",
            ].astype(float)

            betas = indexed.loc[
                weights.index,
                "Market Beta 24M",
            ].astype(float)

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

            gross_traded_notional = (
                calculate_turnover(
                    target_weights=weights,
                    previous_end_weights=(
                        previous_end_weights[
                            strategy
                        ]
                    ),
                )
            )

            trading_cost = (
                gross_traded_notional
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

                    "Trading Cost":
                        trading_cost,

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
                        gross_traded_notional,

                    "Holdings":
                        len(weights),

                    "Maximum Weight":
                        float(
                            weights.max()
                        ),

                    "Effective Holdings":
                        effective_holdings,
                }
            )

            for ticker, weight in (
                weights.items()
            ):

                weight_rows.append(
                    {
                        "Period":
                            period,

                        "Strategy":
                            strategy,

                        "Ticker":
                            ticker,

                        "Weight":
                            weight,
                    }
                )

            previous_end_weights[
                strategy
            ] = calculate_end_weights(
                target_weights=weights,
                realized_returns=returns,
            )

    return (
        pd.DataFrame(
            monthly_rows
        ),
        pd.DataFrame(
            weight_rows
        ),
    )


def calculate_cagr(
    returns: pd.Series,
) -> float:

    values = (
        returns
        .dropna()
        .astype(float)
    )

    if values.empty:
        return 0.0

    wealth = float(
        (
            1.0
            + values
        ).prod()
    )

    years = (
        len(values)
        / 12.0
    )

    if (
        wealth <= 0
        or years <= 0
    ):
        return -1.0

    return float(
        wealth ** (
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


def summarize_portfolios(
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

        annualized_volatility = float(
            returns.std(
                ddof=1
            )
            * np.sqrt(12)
        )

        annualized_mean = float(
            returns.mean()
            * 12
        )

        if annualized_volatility > 0:

            sharpe_zero_rf = (
                annualized_mean
                / annualized_volatility
            )

        else:

            sharpe_zero_rf = 0.0

        benchmark_variance = float(
            spy.var(
                ddof=1
            )
        )

        if benchmark_variance > 0:

            realized_beta = float(
                returns.cov(
                    spy
                )
                / benchmark_variance
            )

        else:

            realized_beta = 0.0

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
                    annualized_volatility,

                "Sharpe RF=0":
                    sharpe_zero_rf,

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