from __future__ import annotations

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)

from src.model_v7 import (
    FEATURE_COLUMNS_V7,
    TARGET_COLUMN_V7,
)

from src.sector_features_v8 import (
    SECTOR_FEATURE_COLUMNS,
)


FEATURE_COLUMNS_V8 = (
    FEATURE_COLUMNS_V7
    + SECTOR_FEATURE_COLUMNS
)


TARGET_COLUMN_V8 = (
    TARGET_COLUMN_V7
)


def make_model_v8(
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


def train_model_v8(
    training_data: pd.DataFrame,
    alpha: float = 10.0,
) -> Pipeline:

    clean = training_data.dropna(
        subset=(
            FEATURE_COLUMNS_V8
            + [
                TARGET_COLUMN_V8,
            ]
        )
    )


    if clean.empty:

        raise ValueError(
            "No V8 training "
            "observations available"
        )


    model = make_model_v8(
        alpha=alpha,
    )


    model.fit(
        clean[
            FEATURE_COLUMNS_V8
        ],
        clean[
            TARGET_COLUMN_V8
        ],
    )


    return model


def predict_cross_section_v8(
    model: Pipeline,
    cross_section: pd.DataFrame,
) -> pd.DataFrame:

    clean = cross_section.dropna(
        subset=FEATURE_COLUMNS_V8
    ).copy()


    if clean.empty:
        return clean


    clean[
        "V8 Prediction"
    ] = model.predict(
        clean[
            FEATURE_COLUMNS_V8
        ]
    )


    clean = clean.sort_values(
        "V8 Prediction",
        ascending=False,
    ).reset_index(
        drop=True
    )


    clean[
        "V8 Rank"
    ] = range(
        1,
        len(clean) + 1,
    )


    return clean