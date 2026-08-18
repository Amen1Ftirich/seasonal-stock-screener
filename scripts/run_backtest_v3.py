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
    get_sp500_tickers,
)


print("Loading S&P 500...")


tickers = get_sp500_tickers()


price_map, errors = (
    load_price_map(
        tickers
    )
)


spy = get_price_history(
    "SPY"
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
]:

    display[column] = (
        display[column]
        * 100
    ).round(2)


print(
    display.to_string(
        index=False
    )
)