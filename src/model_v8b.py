from __future__ import annotations

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)

from src.fundamental_features_v8b import (
    FEATURE_COLUMNS_V8B,
)


TARGET_COLUMN_V8B = (
    "Beta Neutral Target"
)


def make_model_v8b(
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


def train_model_v8b(
    training_data: pd.DataFrame,
    alpha: float = 10.0,
) -> Pipeline:

    clean = training_data.dropna(
        subset=(
            FEATURE_COLUMNS_V8B
            + [
                TARGET_COLUMN_V8B,
            ]
        )
    )


    if clean.empty:

        raise ValueError(
            "No V8B training observations available"
        )


    model = make_model_v8b(
        alpha=alpha,
    )


    model.fit(
        clean[
            FEATURE_COLUMNS_V8B
        ],
        clean[
            TARGET_COLUMN_V8B
        ],
    )


    return model


def predict_cross_section_v8b(
    model: Pipeline,
    cross_section: pd.DataFrame,
) -> pd.DataFrame:

    clean = cross_section.dropna(
        subset=FEATURE_COLUMNS_V8B
    ).copy()


    if clean.empty:

        return clean


    clean[
        "V8B Prediction"
    ] = model.predict(
        clean[
            FEATURE_COLUMNS_V8B
        ]
    )


    clean = clean.sort_values(
        "V8B Prediction",
        ascending=False,
    ).reset_index(
        drop=True
    )


    clean[
        "V8B Rank"
    ] = range(
        1,
        len(clean) + 1,
    )


    return clean