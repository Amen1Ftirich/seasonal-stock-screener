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
def add_beta_neutral_target(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a monthly cross-sectional beta-neutral
    return target.

    For each historical month:

        stock return
            ~
        intercept + pre-month market beta

    The regression residual becomes the prediction
    target.

    Beta was estimated before the month begins.
    Current-month return is used only as the historical
    training label.
    """

    result = data.copy()

    result[
        "Beta Neutral Target"
    ] = np.nan


    for period, group in result.groupby(
        "Period"
    ):

        valid = group.dropna(
            subset=[
                "Return",
                "Market Beta 24M",
            ]
        )

        if len(valid) < 20:
            continue


        beta = (
            valid[
                "Market Beta 24M"
            ]
            .astype(float)
            .to_numpy()
        )


        realized_return = (
            valid[
                "Return"
            ]
            .astype(float)
            .to_numpy()
        )


        design = np.column_stack(
            [
                np.ones(
                    len(valid)
                ),
                beta,
            ]
        )


        coefficients, *_ = (
            np.linalg.lstsq(
                design,
                realized_return,
                rcond=None,
            )
        )


        fitted = (
            design
            @ coefficients
        )


        residual = (
            realized_return
            - fitted
        )


        #
        # Standardize within the month so large-volatility
        # months do not dominate Ridge training.
        #

        residual_std = (
            residual.std(
                ddof=1
            )
        )


        if residual_std <= 0:
            continue


        standardized = (
            residual
            - residual.mean()
        ) / residual_std


        result.loc[
            valid.index,
            "Beta Neutral Target",
        ] = standardized


    return result