from src.backtest_v2 import (
    backtest_cross_section_v2,
)

from src.data import (
    get_price_history,
    load_price_map,
)

from src.universe import (
    get_sp500_tickers,
)


print()
print("Loading S&P 500 universe...")


tickers = get_sp500_tickers()


print(
    f"{len(tickers)} tickers found."
)


price_map, loading_errors = (
    load_price_map(
        tickers
    )
)


print(
    f"{len(price_map)} price histories loaded."
)


spy = get_price_history(
    "SPY"
)


trades, summary, yearly, bootstrap = (
    backtest_cross_section_v2(

        price_map=price_map,

        benchmark_prices=spy,

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

        # September
        target_month=9,

        top_n=10,

        lookback_years=15,

        recent_years=5,

        minimum_samples=8,

        minimum_long_season_rank=0.80,

        minimum_recent_season_rank=0.60,

        minimum_momentum_rank=0.50,

        minimum_downside_rank=0.40,
    )
)


print()
print()
print("======================================")
print("V2 OUT-OF-SAMPLE RESULTS")
print("======================================")
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
print("======================================")
print("BOOTSTRAP")
print("======================================")
print()


for key, value in bootstrap.items():

    if key == "Probability Mean Excess > 0":

        print(
            f"{key}: {value:.1%}"
        )

    else:

        print(
            f"{key}: {value:.2%}"
        )


print()
print("======================================")
print("YEARLY PORTFOLIO")
print("======================================")
print()


if not yearly.empty:

    yearly_display = yearly.copy()

    for column in [
        "Portfolio_Return",
        "Benchmark_Return",
        "Portfolio_Excess",
        "Win_Rate",
        "Beat_Rate",
    ]:

        yearly_display[column] = (
            yearly_display[column] * 100
        ).round(2)

    print(
        yearly_display.to_string(
            index=False
        )
    )


print()
print("======================================")
print("TRADES")
print("======================================")
print()


if not trades.empty:

    print(
        trades.to_string(
            index=False
        )
    )


if not loading_errors.empty:

    print()
    print(
        f"Price loading errors: "
        f"{len(loading_errors)}"
    )

    print(
        loading_errors
        .head(20)
        .to_string(index=False)
    )