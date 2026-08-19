from __future__ import annotations

import numpy as np
import pandas as pd

from src.cross_sectional import (
    add_beta_neutral_target,
)

from src.model_v8c import (
    TARGET_COLUMN_V8C,
    predict_cross_section_v8c,
    train_model_v8c,
)

from src.fundamental_features_v8c import (
    FEATURE_COLUMNS_V8C,
    add_cross_sectional_quarterly_features,
)

def run_walk_forward_v8c(
    panel: pd.DataFrame,
    membership_by_period: (
        dict[pd.Period, set[str]]
        | None
    ) = None,
    start_year: int = 2016,
    end_year: int = 2025,
    training_years: int = 10,
    alpha: float = 10.0,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    V5 cross-sectional walk-forward backtest.

    Model parameters remain fixed.

    Three portfolio constructions are evaluated:

        Top Decile
        Ex-Bottom Decile
        Long-Short Decile
    """

    data = panel.copy()

    data["Period"] = (
        data["Period"]
        .astype("period[M]")
    )


    #
    # Point-in-time universe
    #

    if membership_by_period is not None:

        flags = []

        for ticker, period in zip(
            data["Ticker"],
            data["Period"],
        ):

            members = (
                membership_by_period.get(
                    period
                )
            )

            flags.append(
                members is not None
                and ticker in members
            )

        data[
            "Point In Time Member"
        ] = flags

        data = data[
            data[
                "Point In Time Member"
            ]
        ].copy()


    #
    # Add normalized target.
    #
    # It will only enter TRAINING data.
    #

    data = (
        add_beta_neutral_target(
            data
        )        
    )
    data = (
        add_cross_sectional_quarterly_features(
            data
        )
    )

    periods = sorted(
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


    monthly_results = []

    prediction_rows = []


    for number, period in enumerate(
        periods,
        start=1,
    ):

        print(
            f"[{number}/{len(periods)}] "
            f"{period}"
        )


        training_start = (
            period
            - training_years * 12
        )

        training_end = (
            period - 1
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


        test = data[
            data["Period"] == period
        ].copy()


        train = train.dropna(
            subset=(
                FEATURE_COLUMNS_V8C
                + [
                    TARGET_COLUMN_V8C,
                ]
            )
        )


        test = test.dropna(
            subset=FEATURE_COLUMNS_V8C
        )


        if (
            train.empty
            or len(test) < 30
        ):
            continue


        model = train_model_v8c(
            training_data=train,
            alpha=alpha,
        )


        predicted = (
            predict_cross_section_v8c(
                model=model,
                cross_section=test,
            )
        )


        # -----------------------------------------
        # PREDICTION-BETA DIAGNOSTIC
        # -----------------------------------------
        #
        # This MUST come after "predicted" is created.
        #

        prediction_beta_corr = (
            predicted[
                "V8C Prediction"
            ]
            .corr(
                predicted[
                    "Market Beta 24M"
                ],
                method="spearman",
            )
        )


        #
        # Save all out-of-sample predictions.
        #

        prediction_copy = (
            predicted.copy()
        )

        prediction_copy[
            "Test Period"
        ] = period

        prediction_rows.append(
            prediction_copy
        )


        #
        # Evaluate prediction IC.
        #

        valid = predicted.dropna(
            subset=[
                "V8C Prediction",
                TARGET_COLUMN_V8C,
                "Beta Neutral Return",
            ]
        )


        ic = (
            valid[
                "V8C Prediction"
            ]
            .corr(
                valid[
                    TARGET_COLUMN_V8C
                ],
                method="spearman",
            )
        )

        #
        # Fixed deciles
        #

        decile_size = max(
            1,
            len(valid) // 10,
        )


        top = valid.head(
            decile_size
        )


        bottom = valid.tail(
            decile_size
        )


        ex_bottom = valid.iloc[
            :-decile_size
        ]


        benchmark_return = float(
            valid[
                "Benchmark Return"
            ].iloc[0]
        )


        #
        # Portfolio 1:
        # Top Decile
        #

        top_return = float(
            top["Return"].mean()
        )

        top_residual = float(
            top[
                "Beta Neutral Return"
            ].mean()
        )


        #
        # Portfolio 2:
        # Everything except predicted bottom decile
        #

        ex_bottom_return = float(
            ex_bottom[
                "Return"
            ].mean()
        )

        ex_bottom_residual = float(
            ex_bottom[
                "Residual Return"
            ].mean()
        )


        #
        # Portfolio 3:
        # Equal-dollar long top / short bottom.
        #
        # This is a research diagnostic, not yet
        # transaction-cost adjusted.
        #

        bottom_return = float(
            bottom[
                "Return"
            ].mean()
        )

        bottom_residual = float(
            bottom[
                "Beta Neutral Return"
            ].mean()
        )


        long_short_return = (
            top_return
            - bottom_return
        )

        long_short_residual = (
            top_residual
            - bottom_residual
        )


        monthly_results.append(
            {
                "Period":
                    period,

                "Universe Size":
                    len(valid),

                "IC":
                    ic,

                "Top Decile Return":
                    top_return,

                "Top Decile Excess":
                    (
                        top_return
                        - benchmark_return
                    ),

                "Top Decile Residual":
                    top_residual,

                "Ex Bottom Return":
                    ex_bottom_return,

                "Ex Bottom Excess":
                    (
                        ex_bottom_return
                        - benchmark_return
                    ),

                "Ex Bottom Residual":
                    ex_bottom_residual,

                "Bottom Decile Return":
                    bottom_return,

                "Bottom Decile Residual":
                    bottom_residual,

                "Long Short Return":
                    long_short_return,

                "Long Short Residual":
                    long_short_residual,

                "Benchmark Return":
                    benchmark_return,
                "Prediction Beta Correlation":
                    prediction_beta_corr,
            }
        )


    monthly = pd.DataFrame(
        monthly_results
    )


    if prediction_rows:

        predictions = pd.concat(
            prediction_rows,
            ignore_index=True,
        )

    else:

        predictions = pd.DataFrame()


    return (
        monthly,
        predictions,
    )
def _mean_t_stat(
    values: pd.Series,
) -> float:

    values = (
        values
        .dropna()
        .astype(float)
    )

    if (
        len(values) < 2
        or values.std(ddof=1) == 0
    ):
        return 0.0

    return float(
        values.mean()
        / (
            values.std(ddof=1)
            / np.sqrt(
                len(values)
            )
        )
    )


def summarize_v8c(
    monthly: pd.DataFrame,
) -> dict:

    if monthly.empty:
        return {}

    
    return {

        "Months":
            len(monthly),

        "Average Prediction Beta Correlation":
            float(
                monthly[
                    "Prediction Beta Correlation"
                ].mean()
            ),

        "Average IC":
            float(
                monthly[
                    "IC"
                ].mean()
            ),

        "Median IC":
            float(
                monthly[
                    "IC"
                ].median()
            ),

        "Positive IC Rate":
            float(
                (
                    monthly[
                        "IC"
                    ] > 0
                ).mean()
            ),

        "IC T-Stat":
            _mean_t_stat(
                monthly[
                    "IC"
                ]
            ),


        "Top Decile Return":
            float(
                monthly[
                    "Top Decile Return"
                ].mean()
            ),

        "Top Decile Excess":
            float(
                monthly[
                    "Top Decile Excess"
                ].mean()
            ),

        "Top Decile Residual":
            float(
                monthly[
                    "Top Decile Residual"
                ].mean()
            ),


        "Ex Bottom Return":
            float(
                monthly[
                    "Ex Bottom Return"
                ].mean()
            ),

        "Ex Bottom Excess":
            float(
                monthly[
                    "Ex Bottom Excess"
                ].mean()
            ),

        "Ex Bottom Residual":
            float(
                monthly[
                    "Ex Bottom Residual"
                ].mean()
            ),


        "Bottom Decile Residual":
            float(
                monthly[
                    "Bottom Decile Residual"
                ].mean()
            ),


        "Long Short Return":
            float(
                monthly[
                    "Long Short Return"
                ].mean()
            ),

        "Long Short Residual":
            float(
                monthly[
                    "Long Short Residual"
                ].mean()
            ),

        "Long Short T-Stat":
            _mean_t_stat(
                monthly[
                    "Long Short Return"
                ]
            ),

        "Long Short Residual T-Stat":
            _mean_t_stat(
                monthly[
                    "Long Short Residual"
                ]
            ),
    }