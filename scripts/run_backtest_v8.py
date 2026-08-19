from src.backtest_v8 import (
    run_walk_forward_v8,
    summarize_v8,
)

from src.data import (
    get_price_history,
    load_price_map,
)

from src.dataset_v8 import (
    build_panel_dataset_v8,
)
from src.sector_features_v8 import (
    SECTOR_ETFS,
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

print()
print(
    "Loading sector ETFs..."
)


sector_price_map = {}


for ticker in SECTOR_ETFS:

    print(
        f"Loading {ticker}"
    )

    try:

        sector_price_map[
            ticker
        ] = get_price_history(
            ticker
        )

    except Exception as exc:

        print(
            f"FAILED sector ETF "
            f"{ticker}: {exc}"
        )


print(
    f"Sector ETFs loaded: "
    f"{len(sector_price_map)}"
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


panel = (
    build_panel_dataset_v8(

        price_map=price_map,

        benchmark_prices=spy,

        sector_price_map=(
            sector_price_map
        ),

        minimum_season_samples=5,
    )
)


print(
    f"Panel observations: "
    f"{len(panel):,}"
)


monthly, predictions = (
    run_walk_forward_v8(

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
    "data/cache/v8_monthly.csv",
    index=False,
)


predictions.to_parquet(
    "data/cache/v8_predictions.parquet",
    index=False,
)


summary = summarize_v8(
    monthly
)


print()
print(
    "======================================"
)

print(
    "V8A SECTOR-RELATIVE RESULTS"
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