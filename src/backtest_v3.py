from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_v3 import (
    FEATURE_COLUMNS,
    predict_cross_section,
    train_model,
)


def run_walk_forward_v3(
    panel: pd.DataFrame,
    start_year: int = 2016,
    end_year: int = 2025,
    top_n: int = 10,
    training_years: int = 10,
    alpha: float = 10.0,

    membership_by_period: (
        dict[pd.Period, set[str]]
        | None
    ) = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    True monthly walk-forward simulation.

    Every month:

        Train using ONLY earlier months.
        Predict current-month excess returns.
        Buy top N predictions.
        Reveal actual returns.

    No current/future month enters training.
    """

    data = panel.copy()

    data["Period"] = (
        data["Period"]
        .astype("period[M]")
    )
    #
    # POINT-IN-TIME UNIVERSE FILTER
    #
    # A stock-month is allowed only if that stock
    # actually belonged to the S&P 500 at the
    # beginning of that historical month.
    #

    if membership_by_period is not None:

        membership_flags = []

        for ticker, period in zip(
            data["Ticker"],
            data["Period"],
        ):

            members = (
                membership_by_period.get(
                    period
                )
            )

            if members is None:

                membership_flags.append(
                    False
                )

            else:

                membership_flags.append(
                    ticker in members
                )

        data[
            "Point In Time Member"
        ] = membership_flags

        before = len(data)

        data = data[
            data[
                "Point In Time Member"
            ]
        ].copy()

        after = len(data)

        print(
            f"Point-in-time filter: "
            f"{before:,} -> {after:,} "
            f"stock-month observations"
        )
    test_periods = sorted(
        data[
            (
                data["Period"].dt.year
                >= start_year
            )
            & (
                data["Period"].dt.year
                <= end_year
            )
        ]["Period"].unique()
    )

    trades = []
    monthly_results = []
    diagnostic_results = []

    for number, period in enumerate(
        test_periods,
        start=1,
    ):

        print(
            f"[{number}/{len(test_periods)}] "
            f"{period}"
        )

        test = data[
            data["Period"] == period
        ].copy()

        if test.empty:
            continue

        training_end = period - 1

        training_start = (
            period
            - training_years * 12
        )

        train = data[
            (
                data["Period"]
                >= training_start
            )
            & (
                data["Period"]
                <= training_end
            )
        ].copy()

        train = train.dropna(
            subset=(
                FEATURE_COLUMNS
                + [
                    "Residual Return",
                ]
            )
        )

        test = test.dropna(
            subset=FEATURE_COLUMNS
        )

        if (
            train.empty
            or len(test) < top_n
        ):
            continue

        model = train_model(
            training_data=train,
            alpha=alpha,
        )

        predicted = (
            predict_cross_section(
                model=model,
                cross_section=test,
            )
        )
        #
        # ---------------------------------------------
        # CROSS-SECTIONAL SIGNAL DIAGNOSTICS
        # ---------------------------------------------
        #
        # Predictions were made without seeing the
        # current month's outcome.
        #
        # We now reveal current returns purely for
        # evaluation.
        #

        diagnostic = predicted.dropna(
            subset=[
                "Predicted Residual Return",
                "Alpha Quality Score",
                "Residual Return",
                "Excess Return",
            ]
        ).copy()


        if len(diagnostic) >= 20:

            #
            # Information Coefficient
            #
            # Spearman correlation asks whether stocks
            # ranked highly by the model actually tended
            # to produce higher realized residual returns.
            #

            prediction_ic = (
                diagnostic[
                    "Predicted Residual Return"
                ]
                .corr(
                    diagnostic[
                        "Residual Return"
                    ],
                    method="spearman",
                )
            )


            quality_ic = (
                diagnostic[
                    "Alpha Quality Score"
                ]
                .corr(
                    diagnostic[
                        "Residual Return"
                    ],
                    method="spearman",
                )
            )


            #
            # Sort by predicted residual alpha.
            #

            ranked = diagnostic.sort_values(
                "Predicted Residual Return",
                ascending=False,
            ).reset_index(drop=True)


            decile_size = max(
                1,
                len(ranked) // 10,
            )


            top_decile = ranked.head(
                decile_size
            )


            bottom_decile = ranked.tail(
                decile_size
            )


            top_residual = float(
                top_decile[
                    "Residual Return"
                ].mean()
            )


            bottom_residual = float(
                bottom_decile[
                    "Residual Return"
                ].mean()
            )


            residual_spread = (
                top_residual
                - bottom_residual
            )


            top_excess = float(
                top_decile[
                    "Excess Return"
                ].mean()
            )


            bottom_excess = float(
                bottom_decile[
                    "Excess Return"
                ].mean()
            )


            excess_spread = (
                top_excess
                - bottom_excess
            )


            diagnostic_results.append(
                {
                    "Period":
                        period,

                    "Universe Size":
                        len(diagnostic),

                    "Prediction IC":
                        prediction_ic,

                    "Quality IC":
                        quality_ic,

                    "Top Decile Residual":
                        top_residual,

                    "Bottom Decile Residual":
                        bottom_residual,

                    "Residual Spread":
                        residual_spread,

                    "Top Decile Excess":
                        top_excess,

                    "Bottom Decile Excess":
                        bottom_excess,

                    "Excess Spread":
                        excess_spread,
                }
            )
        selected = (
            predicted
            .head(top_n)
            .copy()
        )

        selected[
            "Test Period"
        ] = period

        trades.append(
            selected
        )

        portfolio_return = float(
            selected["Return"].mean()
        )

        benchmark_return = float(
            selected[
                "Benchmark Return"
            ].iloc[0]
        )
        portfolio_beta = float(
            selected[
                "Market Beta 24M"
            ].mean()
        )


        realized_residual_return = float(
            selected[
                "Residual Return"
            ].mean()
        )
        excess_return = (
            portfolio_return
            - benchmark_return
        )

        monthly_results.append(
            {
                "Period": period,

                "Stocks":
                    len(selected),

                "Portfolio Return":
                    portfolio_return,

                "Benchmark Return":
                    benchmark_return,

                "Excess Return":
                    excess_return,

                "Beat Benchmark":
                    excess_return > 0,

                "Positive Month":
                    portfolio_return > 0,
                "Portfolio Beta":
                    portfolio_beta,

                "Residual Alpha":
                    realized_residual_return,
            }
        )

    if trades:

        trade_df = pd.concat(
            trades,
            ignore_index=True,
        )

    else:

        trade_df = pd.DataFrame()

    monthly_df = pd.DataFrame(
        monthly_results
    )


    diagnostic_df = pd.DataFrame(
        diagnostic_results
    )


    return (
        trade_df,
        monthly_df,
        diagnostic_df,
    )


def summarize_v3(
    monthly_results: pd.DataFrame,
) -> dict:

    if monthly_results.empty:

        return {}

    excess = (
        monthly_results[
            "Excess Return"
        ]
    )

    returns = (
        monthly_results[
            "Portfolio Return"
        ]
    )
    residual_alpha = (
        monthly_results[
            "Residual Alpha"
        ]
    )
    return {
        "Months":
            len(monthly_results),

        "Positive Month Rate":
            float(
                monthly_results[
                    "Positive Month"
                ].mean()
            ),

        "Beat SPY Rate":
            float(
                monthly_results[
                    "Beat Benchmark"
                ].mean()
            ),

        "Average Monthly Return":
            float(
                returns.mean()
            ),

        "Median Monthly Return":
            float(
                returns.median()
            ),

        "Average Monthly Excess":
            float(
                excess.mean()
            ),

        "Median Monthly Excess":
            float(
                excess.median()
            ),

        "Excess Volatility":
            float(
                excess.std(ddof=1)
            ),

        "Worst Month":
            float(
                returns.min()
            ),

        "Best Month":
            float(
                returns.max()
            ),
        "Average Residual Alpha":
            float(
                residual_alpha.mean()
            ),

        "Median Residual Alpha":
            float(
                residual_alpha.median()
            ),

        "Positive Residual Alpha Rate":
            float(
                (
                    residual_alpha > 0
                ).mean()
            ),

        "Average Portfolio Beta":
            float(
                monthly_results[
                    "Portfolio Beta"
                ].mean()
            ),
    }
def hac_mean_t_stat(
    values: pd.Series,
    max_lag: int = 3,
) -> float:
    """
    Newey-West style HAC t-statistic for the mean.

    More appropriate than the ordinary t-stat when
    monthly observations may be serially correlated.
    """

    values = (
        values
        .dropna()
        .astype(float)
    )

    n = len(values)

    if n < 2:
        return 0.0

    x = (
        values.to_numpy()
        - values.mean()
    )

    gamma_zero = (
        x @ x
    ) / n

    long_run_variance = gamma_zero

    for lag in range(
        1,
        min(max_lag, n - 1) + 1,
    ):

        weight = (
            1
            - lag
            / (max_lag + 1)
        )

        covariance = (
            x[lag:]
            @ x[:-lag]
        ) / n

        long_run_variance += (
            2
            * weight
            * covariance
        )

    variance_of_mean = (
        long_run_variance
        / n
    )

    if variance_of_mean <= 0:
        return 0.0

    return float(
        values.mean()
        / (
            variance_of_mean
            ** 0.5
        )
    )

def summarize_signal_diagnostics(
    diagnostics: pd.DataFrame,
) -> dict:
    """
    Measure whether the model possesses genuine
    cross-sectional ranking information.

    These metrics are evaluated one month at a time,
    so months rather than individual stocks are the
    independent observations.
    """

    if diagnostics.empty:

        return {
            "Diagnostic Months": 0,
            "Average IC": 0.0,
            "Median IC": 0.0,
            "Positive IC Rate": 0.0,
            "IC T-Stat": 0.0,
            "Average Residual Spread": 0.0,
            "Median Residual Spread": 0.0,
            "Positive Spread Rate": 0.0,
            "Spread T-Stat": 0.0,
            "Average Excess Spread": 0.0,
        }


    ic = diagnostics[
        "Prediction IC"
    ].dropna()


    quality_ic = diagnostics[
        "Quality IC"
    ].dropna()


    spread = diagnostics[
        "Residual Spread"
    ].dropna()


    excess_spread = diagnostics[
        "Excess Spread"
    ].dropna()


    prediction_hac_t = (
        hac_mean_t_stat(
            ic,
            max_lag=3,
        )
    )


    quality_hac_t = (
        hac_mean_t_stat(
            quality_ic,
            max_lag=3,
        )
    )


    spread_hac_t = (
        hac_mean_t_stat(
            spread,
            max_lag=3,
        )
    )


    #
    # Monthly t-stat of average IC.
    #

    if (
        len(ic) > 1
        and ic.std(ddof=1) > 0
    ):

        ic_t_stat = (
            ic.mean()
            / (
                ic.std(ddof=1)
                / (len(ic) ** 0.5)
            )
        )

    else:

        ic_t_stat = 0.0


    #
    # Monthly t-stat of top-minus-bottom
    # residual spread.
    #

    if (
        len(spread) > 1
        and spread.std(ddof=1) > 0
    ):

        spread_t_stat = (
            spread.mean()
            / (
                spread.std(ddof=1)
                / (len(spread) ** 0.5)
            )
        )

    else:

        spread_t_stat = 0.0


    return {
        "Diagnostic Months":
            len(diagnostics),

        "Average IC":
            float(ic.mean()),

        "Median IC":
            float(ic.median()),

        "Positive IC Rate":
            float(
                (ic > 0).mean()
            ),

        "IC T-Stat":
            float(ic_t_stat),

        "Average Residual Spread":
            float(
                spread.mean()
            ),

        "Median Residual Spread":
            float(
                spread.median()
            ),

        "Positive Spread Rate":
            float(
                (spread > 0).mean()
            ),

        "Spread T-Stat":
            float(
                spread_t_stat
            ),

        "Average Excess Spread":
            float(
                excess_spread.mean()
            ),
        "Average Quality IC":
            float(
                quality_ic.mean()
            ),

        "Median Quality IC":
            float(
                quality_ic.median()
            ),

        "Positive Quality IC Rate":
            float(
                (
                    quality_ic > 0
                ).mean()
            ),

        "Prediction IC HAC T-Stat":
            prediction_hac_t,

        "Quality IC HAC T-Stat":
            quality_hac_t,

        "Spread HAC T-Stat":
            spread_hac_t,

        "Average Top Decile Residual":
            float(
                diagnostics[
                    "Top Decile Residual"
                ].mean()
            ),

        "Average Bottom Decile Residual":
            float(
                diagnostics[
                    "Bottom Decile Residual"
                ].mean()
            ),

        "Average Top Decile Excess":
            float(
                diagnostics[
                    "Top Decile Excess"
                ].mean()
            ),
    }