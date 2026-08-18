from __future__ import annotations

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


#
# Market-level features are deliberately excluded.
#
# They are identical for every stock in a given
# cross-section and therefore do not directly help a
# linear model rank stocks within that month.
#

FEATURE_COLUMNS_V5 = [

    # Seasonality
    "Season Mean",
    "Season Recent",
    "Season Median",
    "Season Beat Rate",
    "Season Q25",

    # Residual seasonality
    "Residual Season Mean",
    "Residual Season Recent",
    "Residual Season Median",

    # Raw momentum
    "Momentum 3M",
    "Momentum 6M",
    "Momentum 12M",

    # Residual momentum
    "Residual Momentum 3M",
    "Residual Momentum 6M",
    "Residual Momentum 12M",

    # Short-term behavior
    "Last Month Return",
    "Last Month Excess",

    # Market exposure
    "Market Beta 24M",

    # Risk
    "Volatility 6M",
    "Volatility 12M",
    "Idiosyncratic Volatility 6M",
    "Idiosyncratic Volatility 12M",
]


TARGET_COLUMN_V5 = (
    "Cross Sectional Residual Target"
)


def add_cross_sectional_target(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert realized residual return into a
    within-month cross-sectional standardized target.

    For each historical month:

        residual return
              ↓
        subtract cross-sectional mean
              ↓
        divide by cross-sectional std

    This changes the learning problem from:

        "How large will this stock's return be?"

    to:

        "How strong will this stock be relative to
         its peers this month?"
    """

    result = data.copy()

    grouped = result.groupby(
        "Period"
    )["Residual Return"]

    monthly_mean = grouped.transform(
        "mean"
    )

    monthly_std = grouped.transform(
        "std"
    )

    result[
        TARGET_COLUMN_V5
    ] = (
        result["Residual Return"]
        - monthly_mean
    ) / monthly_std

    return result


def make_model_v5(
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


def train_model_v5(
    training_data: pd.DataFrame,
    alpha: float = 10.0,
) -> Pipeline:

    clean = training_data.dropna(
        subset=(
            FEATURE_COLUMNS_V5
            + [
                TARGET_COLUMN_V5,
            ]
        )
    )

    if clean.empty:

        raise ValueError(
            "No V5 training observations available"
        )

    model = make_model_v5(
        alpha=alpha
    )

    model.fit(
        clean[
            FEATURE_COLUMNS_V5
        ],
        clean[
            TARGET_COLUMN_V5
        ],
    )

    return model


def predict_cross_section_v5(
    model: Pipeline,
    cross_section: pd.DataFrame,
) -> pd.DataFrame:

    clean = cross_section.dropna(
        subset=FEATURE_COLUMNS_V5
    ).copy()

    if clean.empty:
        return clean

    clean[
        "V5 Prediction"
    ] = model.predict(
        clean[
            FEATURE_COLUMNS_V5
        ]
    )

    clean = clean.sort_values(
        "V5 Prediction",
        ascending=False,
    ).reset_index(
        drop=True
    )

    clean[
        "V5 Rank"
    ] = range(
        1,
        len(clean) + 1,
    )

    clean[
        "V5 Percentile"
    ] = (
        1
        - (
            clean.index
            / len(clean)
        )
    )

    return clean