import pandas as pd

from src.validation import chronological_split


def test_chronological_split():

    data = pd.DataFrame(
        {
            "Year": list(range(2011, 2026)),
            "Return": [0.05] * 15,
        }
    )

    train, test = chronological_split(
        data,
        test_years=5,
    )

    assert len(train) == 10
    assert len(test) == 5

    assert train["Year"].max() == 2020
    assert test["Year"].min() == 2021
    assert test["Year"].max() == 2025