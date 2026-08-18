from src.backtest_v3 import (
    run_walk_forward_v3,
    summarize_v3,
)

from src.data import (
    get_price_history,
    load_price_map,
)

from src.dataset_v3 import (
    build_panel_dataset,
)

from src.universe import (
    build_sp500_membership_map,
    get_sp500_historical_union,
)


print("Loading S&P 500...")

#
# We need history beginning roughly ten years
# before our 2016 test begins.
#

tickers = get_sp500_historical_union(
    start_date="2006-01-01"
)


print(
    f"Historical ticker union: "
    f"{len(tickers)}"
)


price_map, errors = (
    load_price_map(
        tickers
    )
)


spy = get_price_history(
    "SPY"
)

print()
print(
    "Building point-in-time "
    "S&P 500 membership..."
)


membership_map = (
    build_sp500_membership_map(
        start_date="2006-01",
        end_date="2025-12",
    )
)


membership_sizes = [
    len(members)
    for members
    in membership_map.values()
]


print(
    f"Membership size range: "
    f"{min(membership_sizes)} "
    f"to "
    f"{max(membership_sizes)}"
)

print()
print("Building panel dataset...")


panel = build_panel_dataset(
    price_map=price_map,
    benchmark_prices=spy,
    minimum_season_samples=5,
)


print()
print(
    f"Panel observations: "
    f"{len(panel):,}"
)


print()
print("Running V3 walk-forward...")


trades, monthly = (
    run_walk_forward_v3(

        panel=panel,

        start_year=2016,
        end_year=2025,

        top_n=10,

        training_years=10,

        alpha=10.0,
        membership_by_period=(
            membership_map
        ),
    )
)


summary = summarize_v3(
    monthly
)


print()
print(
    "======================================"
)
print(
    "V3 WALK-FORWARD RESULTS"
)
print(
    "======================================"
)


for key, value in summary.items():

    if key == "Months":

        print(
            f"{key}: {value}"
        )

    elif key == "Average Portfolio Beta":

        print(
            f"{key}: {value:.2f}"
        )

    else:

        print(
            f"{key}: {value:.2%}"
        )


print()
print(
    "======================================"
)
print(
    "MONTHLY RESULTS"
)
print(
    "======================================"
)


display = monthly.copy()


for column in [
    "Portfolio Return",
    "Benchmark Return",
    "Excess Return",
    "Residual Alpha",
]:

    display[column] = (
        display[column]
        * 100
    ).round(2)
    display["Portfolio Beta"] = (
        display["Portfolio Beta"]
        .round(2)
    )


print(
    display.to_string(
        index=False
    )
)