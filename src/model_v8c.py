from __future__ import annotations

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)

from src.fundamental_features_v8c import (
    FEATURE_COLUMNS_V8C,
)


TARGET_COLUMN_V8C = (
    "Beta Neutral Target"
)


def make_model_v8c(
    alpha: float = 10.0,
) -> Pipeline:

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


def train_model_v8c(
    training_data: pd.DataFrame,
    alpha: float = 10.0,
) -> Pipeline:

    clean = training_data.dropna(
        subset=(
            FEATURE_COLUMNS_V8C
            + [
                TARGET_COLUMN_V8C,
            ]
        )
    )


    if clean.empty:

        raise ValueError(
            "No V8C training observations available"
        )


    model = make_model_v8c(
        alpha=alpha,
    )


    model.fit(
        clean[
            FEATURE_COLUMNS_V8C
        ],
        clean[
            TARGET_COLUMN_V8C
        ],
    )


    return model


def predict_cross_section_v8c(
    model: Pipeline,
    cross_section: pd.DataFrame,
) -> pd.DataFrame:

    clean = cross_section.dropna(
        subset=FEATURE_COLUMNS_V8C
    ).copy()


    if clean.empty:
        return clean


    clean[
        "V8C Prediction"
    ] = model.predict(
        clean[
            FEATURE_COLUMNS_V8C
        ]
    )


    clean = clean.sort_values(
        "V8C Prediction",
        ascending=False,
    ).reset_index(
        drop=True
    )


    clean[
        "V8C Rank"
    ] = range(
        1,
        len(clean) + 1,
    )


    return clean