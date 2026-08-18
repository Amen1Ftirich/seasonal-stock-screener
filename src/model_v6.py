from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model_v5 import (
    FEATURE_COLUMNS_V5,
    TARGET_COLUMN_V5,
)


FEATURE_COLUMNS_V6 = [
    f"CS {column}"
    for column in FEATURE_COLUMNS_V5
]


def add_cross_sectional_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize each predictor relative to the other
    investable stocks in the SAME historical month.

    A value of:

        +1.0

    means the stock's characteristic was one
    cross-sectional standard deviation above the
    monthly universe average.

    All underlying predictors were already known
    before the target month, so this does not use
    the target return.
    """

    result = data.copy()

    for column in FEATURE_COLUMNS_V5:

        grouped = result.groupby(
            "Period"
        )[column]

        monthly_mean = grouped.transform(
            "mean"
        )

        monthly_std = grouped.transform(
            "std"
        )

        monthly_std = monthly_std.replace(
            0,
            np.nan,
        )

        result[
            f"CS {column}"
        ] = (
            result[column]
            - monthly_mean
        ) / monthly_std

    return result


def make_model_v6(
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


def train_model_v6(
    training_data: pd.DataFrame,
    alpha: float = 10.0,
) -> Pipeline:

    clean = training_data.dropna(
        subset=(
            FEATURE_COLUMNS_V6
            + [
                TARGET_COLUMN_V5,
            ]
        )
    )

    if clean.empty:

        raise ValueError(
            "No V6 training observations available"
        )

    model = make_model_v6(
        alpha=alpha,
    )

    model.fit(
        clean[
            FEATURE_COLUMNS_V6
        ],
        clean[
            TARGET_COLUMN_V5
        ],
    )

    return model


def predict_cross_section_v6(
    model: Pipeline,
    cross_section: pd.DataFrame,
) -> pd.DataFrame:

    clean = cross_section.dropna(
        subset=FEATURE_COLUMNS_V6
    ).copy()

    if clean.empty:
        return clean

    clean[
        "V6 Prediction"
    ] = model.predict(
        clean[
            FEATURE_COLUMNS_V6
        ]
    )

    clean = clean.sort_values(
        "V6 Prediction",
        ascending=False,
    ).reset_index(
        drop=True
    )

    clean[
        "V6 Rank"
    ] = range(
        1,
        len(clean) + 1,
    )

    clean[
        "V6 Percentile"
    ] = (
        1
        - (
            clean.index
            / len(clean)
        )
    )

    return clean    