import pandas as pd
import pytest

from src.metrics import calculate_metrics, wilson_lower_bound


def test_calculate_metrics():

    data = pd.DataFrame(
        {
            "Return": [
                0.10,
                0.05,
                -0.02,
                0.03,
                -0.01,
            ]
        }
    )

    metrics = calculate_metrics(data)

    assert metrics["sample_size"] == 5
    assert metrics["wins"] == 3
    assert metrics["losses"] == 2

    assert metrics["win_rate"] == pytest.approx(0.60)

    assert metrics["average_return"] == pytest.approx(
        data["Return"].mean()
    )

    assert metrics["best_return"] == pytest.approx(0.10)

    assert metrics["worst_return"] == pytest.approx(-0.02)


def test_wilson_lower_bound():

    bound = wilson_lower_bound(
        wins=8,
        total=10,
    )

    assert 0 < bound < 0.80