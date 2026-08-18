from __future__ import annotations

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [

    #
    # Original seasonality
    #

    "Season Mean",
    "Season Recent",
    "Season Median",
    "Season Beat Rate",
    "Season Q25",

    #
    # Residual seasonality
    #

    "Residual Season Mean",
    "Residual Season Recent",
    "Residual Season Median",

    #
    # Raw momentum
    #

    "Momentum 3M",
    "Momentum 6M",
    "Momentum 12M",

    #
    # Residual momentum
    #

    "Residual Momentum 3M",
    "Residual Momentum 6M",
    "Residual Momentum 12M",

    #
    # Short-term behavior
    #

    "Last Month Return",
    "Last Month Excess",

    #
    # Market sensitivity
    #

    "Market Beta 24M",

    #
    # Risk
    #

    "Volatility 6M",
    "Volatility 12M",

    "Idiosyncratic Volatility 6M",
    "Idiosyncratic Volatility 12M",

    #
    # Market regime
    #

    "Market Momentum 3M",
    "Market Momentum 6M",
    "Market Volatility 6M",
]
TARGET_COLUMN = "Residual Return"


def make_model(
    alpha: float = 10.0,
) -> Pipeline:
    """
    Strongly regularized linear model.

    We deliberately start with Ridge instead of
    Random Forest / XGBoost to reduce overfitting.
    """

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "model",
                Ridge(
                    alpha=alpha,
                ),
            ),
        ]
    )


def train_model(
    training_data: pd.DataFrame,
    alpha: float = 10.0,
) -> Pipeline:

    clean = training_data.dropna(
        subset=(
            FEATURE_COLUMNS
            + [
                TARGET_COLUMN,
            ]
        )
    )

    if clean.empty:
        raise ValueError(
            "No training observations available"
        )

    model = make_model(
        alpha=alpha
    )

    model.fit(
        clean[FEATURE_COLUMNS],
        clean[TARGET_COLUMN],
    )

    return model


def predict_cross_section(
    model: Pipeline,
    cross_section: pd.DataFrame,
) -> pd.DataFrame:

    clean = cross_section.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    if clean.empty:
        return clean

    clean["Predicted Residual Return"] = (
        model.predict(
            clean[FEATURE_COLUMNS]
        )
    )
    volatility_floor = 0.02


    clean["Alpha Quality Score"] = (
        clean["Predicted Residual Return"]
        / (
            clean[
                "Idiosyncratic Volatility 6M"
            ]
            + volatility_floor
        )
    )
    clean = clean.sort_values(
        "Alpha Quality Score",
        ascending=False,
    ).reset_index(drop=True)
    clean["Prediction Rank"] = range(
        1,
        len(clean) + 1,
    )

    return clean