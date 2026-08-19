from src.backtest_v7 import (
    run_walk_forward_v7,
    summarize_v7,
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


print(
    "Loading historical universe..."
)


tickers = (
    get_sp500_historical_union(
        start_date="2006-01-01"
    )
)


price_map, errors = (
    load_price_map(
        tickers
    )
)


spy = get_price_history(
    "SPY"
)


membership_map = (
    build_sp500_membership_map(
        start_date="2006-01",
        end_date="2025-12",
    )
)


print(
    "Building panel..."
)


panel = build_panel_dataset(
    price_map=price_map,
    benchmark_prices=spy,
    minimum_season_samples=5,
)


print(
    f"Panel observations: "
    f"{len(panel):,}"
)


monthly, predictions = (
    run_walk_forward_v7(

        panel=panel,

        membership_by_period=(
            membership_map
        ),

        start_year=2016,
        end_year=2025,

        training_years=10,

        alpha=10.0,
    )
)


#
# Save expensive output BEFORE summarizing.
#

monthly.to_csv(
    "data/cache/v7_monthly.csv",
    index=False,
)


predictions.to_parquet(
    "data/cache/v7_predictions.parquet",
    index=False,
)


summary = summarize_v7(
    monthly
)


print()
print(
    "======================================"
)

print(
    "V7 CROSS-SECTIONAL RESULTS"
)

print(
    "======================================"
)


for key, value in summary.items():

    if key == "Months":

        print(
            f"{key}: {value}"
        )

    elif (
        "IC" in key
        or "T-Stat" in key
    ):

        print(
            f"{key}: {value:.3f}"
        )

    else:

        print(
            f"{key}: {value:.2%}"
        )