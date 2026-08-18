from __future__ import annotations

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "Season Mean",
    "Season Recent",
    "Season Median",
    "Season Beat Rate",
    "Season Q25",

    "Momentum 3M",
    "Momentum 6M",
    "Momentum 12M",

    "Last Month Return",
    "Last Month Excess",

    "Volatility 6M",
    "Volatility 12M",

    "Market Momentum 3M",
    "Market Momentum 6M",
    "Market Volatility 6M",
]


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
                "Excess Return",
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
        clean["Excess Return"],
    )

    return model


def predict_cross_section(
    model: Pipeline,
    cross_section: pd.DataFrame,
) -> pd.DataFrame:

    clean = cross_section.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    clean[
        "Predicted Excess Return"
    ] = model.predict(
        clean[FEATURE_COLUMNS]
    )

    clean = clean.sort_values(
        "Predicted Excess Return",
        ascending=False,
    )

    clean["Prediction Rank"] = range(
        1,
        len(clean) + 1,
    )

    return clean