from __future__ import annotations

import numpy as np
import pandas as pd


CROSS_SECTIONAL_TARGET = (
    "Cross Sectional Residual Target"
)


def add_cross_sectional_target(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize realized residual returns within
    each historical month.

    This is a training target only.

    It answers:

        How strong was this stock relative to
        the other stocks in the same month?
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

    monthly_std = monthly_std.replace(
        0,
        np.nan,
    )

    result[
        CROSS_SECTIONAL_TARGET
    ] = (
        result["Residual Return"]
        - monthly_mean
    ) / monthly_std

    return result