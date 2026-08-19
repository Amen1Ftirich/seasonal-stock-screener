from __future__ import annotations

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model_v5 import (
    FEATURE_COLUMNS_V5,
)


FEATURE_COLUMNS_V7 = (
    FEATURE_COLUMNS_V5
)


TARGET_COLUMN_V7 = (
    "Beta Neutral Target"
)


def make_model_v7(
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


def train_model_v7(
    training_data: pd.DataFrame,
    alpha: float = 10.0,
) -> Pipeline:

    clean = training_data.dropna(
        subset=(
            FEATURE_COLUMNS_V7
            + [
                TARGET_COLUMN_V7,
            ]
        )
    )

    if clean.empty:

        raise ValueError(
            "No V7 training observations available"
        )


    model = make_model_v7(
        alpha=alpha,
    )


    model.fit(
        clean[
            FEATURE_COLUMNS_V7
        ],
        clean[
            TARGET_COLUMN_V7
        ],
    )


    return model


def predict_cross_section_v7(
    model: Pipeline,
    cross_section: pd.DataFrame,
) -> pd.DataFrame:

    clean = cross_section.dropna(
        subset=FEATURE_COLUMNS_V7
    ).copy()


    if clean.empty:
        return clean


    clean[
        "V7 Prediction"
    ] = model.predict(
        clean[
            FEATURE_COLUMNS_V7
        ]
    )


    clean = clean.sort_values(
        "V7 Prediction",
        ascending=False,
    ).reset_index(
        drop=True
    )


    clean[
        "V7 Rank"
    ] = range(
        1,
        len(clean) + 1,
    )


    return clean