import pandas as pd

from src.walkforward import walk_forward_validate


def test_walk_forward_validate():

    years = list(range(2011, 2026))

    observations = pd.DataFrame(
        {
            "Year": years,

            "Return": [
                0.05,
                0.04,
                0.06,
                -0.01,
                0.03,
                0.05,
                0.02,
                0.04,
                0.06,
                0.03,
                0.05,
                -0.02,
                0.04,
                0.03,
                0.06,
            ],

            "SPY Return": [
                0.02
            ] * 15,
        }
    )

    observations["Excess Return"] = (
        observations["Return"]
        - observations["SPY Return"]
    )

    observations["Beat SPY"] = (
        observations["Excess Return"] > 0
    )

    folds, summary = walk_forward_validate(
        observations=observations,
        minimum_training_years=8,
        minimum_win_rate=0.60,
        minimum_median_return=0.0,
        minimum_beat_benchmark_rate=0.50,
        minimum_median_excess_return=0.0,
    )

    assert len(folds) == 7

    assert folds.iloc[0]["Test Year"] == 2019

    assert folds.iloc[0]["Training Years"] == 8

    assert summary["WF Folds"] == 7

    assert 0 <= summary["WF Win Rate"] <= 1