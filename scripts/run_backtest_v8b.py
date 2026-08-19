from src.backtest_v8b import (
    run_walk_forward_v8b,
    summarize_v8b,
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
from pathlib import Path

import pandas as pd

from src.fundamental_features_v8b import (
    attach_point_in_time_fundamentals,
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

fundamental_panel_cache = Path(
    "data/cache/v8b_full_fundamental_panel.parquet"
)


if fundamental_panel_cache.exists():

    print(
        "Loading cached V8B fundamental panel..."
    )

    panel = pd.read_parquet(
        fundamental_panel_cache
    )


else:

    print()
    print(
        "Attaching point-in-time SEC fundamentals..."
    )


    resolution = pd.read_csv(
        "data/cache/"
        "sec_identifier_resolution.csv"
    )


    timeline = pd.read_parquet(
        "data/cache/"
        "sec_fundamental_timeline.parquet"
    )


    panel = (
        attach_point_in_time_fundamentals(
            data=panel,
            resolution=resolution,
            timeline=timeline,
        )
    )


    panel.to_parquet(
        fundamental_panel_cache,
        index=False,
    )


    print(
        "Saved V8B fundamental panel."
    )

monthly, predictions = (
    run_walk_forward_v8b(

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
    "data/cache/v8b_monthly.csv",
    index=False,
)


predictions.to_parquet(
    "data/cache/v8b_predictions.parquet",
    index=False,
)


summary = summarize_v8b(
    monthly
)


print()
print(
    "======================================"
)

print(
    "V8B FUNDAMENTALS-ONLY RESULTS"
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