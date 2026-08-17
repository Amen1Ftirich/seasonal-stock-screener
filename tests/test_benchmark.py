import pandas as pd
import pytest

from src.benchmark import add_benchmark_returns


def test_add_benchmark_returns():

    seasonal = pd.DataFrame(
        {
            "Entry Date": pd.to_datetime(
                ["2020-01-02"]
            ),
            "Exit Date": pd.to_datetime(
                ["2020-01-06"]
            ),
            "Return": [0.10],
        }
    )

    benchmark = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-06",
                ]
            ),
            "Adj Close": [
                100.0,
                105.0,
            ],
        }
    )

    result = add_benchmark_returns(
        seasonal,
        benchmark,
    )

    assert result.loc[
        0, "SPY Return"
    ] == pytest.approx(0.05)

    assert result.loc[
        0, "Excess Return"
    ] == pytest.approx(0.05)

    assert bool(
        result.loc[0, "Beat SPY"]
    ) is True