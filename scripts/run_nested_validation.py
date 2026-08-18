from src.nested_validation import (
    nested_walk_forward_discovery,
)

import random

from src.universe import get_sp500_tickers

sp500 = get_sp500_tickers()

rng = random.Random(42)

research_universe = rng.sample(
    sp500,
    min(100, len(sp500)),
)
results, summary, errors = (
    nested_walk_forward_discovery(

        tickers=research_universe,

        # This should match approximately
        # today's live seasonal search date.
        anchor_month=8,
        anchor_day=17,

        test_years=[
            2016,
            2017,
            2018,
            2019,
            2020,
            2021,
            2022,
            2023,
            2024,
            2025,
        ],

        days_ahead=7,

        holding_periods=[
            10,
            15,
            20,
        ],

        lookback_years=15,

        top_n=10,

        minimum_win_rate=0.55,

        minimum_sample_size=8,

        minimum_median_return=0.0,

        minimum_beat_benchmark_rate=0.50,

        minimum_median_excess_return=0.0,
    )
)


print()
print("===================================")
print("NESTED WALK-FORWARD RESULTS")
print("===================================")

print()

for key, value in summary.items():

    if (
        "Rate" in key
        or "Return" in key
    ):
        print(
            f"{key}: {value:.2%}"
        )
    else:
        print(
            f"{key}: {value}"
        )


print()
print("TRADES")
print()

print(
    results.to_string(
        index=False
    )
)


if not errors.empty:

    print()
    print("ERRORS")
    print(
        errors.head(20).to_string(
            index=False
        )
    )