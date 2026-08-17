from datetime import date

from src.discovery import discover_upcoming_windows


tickers = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "AVGO",
    "META",
    "GOOGL",
    "AMZN",
    "QCOM",
    "INTC",
]


results, errors = discover_upcoming_windows(
    tickers=tickers,

    start_date=date.today(),

    days_ahead=7,

    holding_periods=[
        5,
        10,
        15,
        20,
    ],

    lookback_years=15,

    minimum_win_rate=0.65,
    minimum_sample_size=10,
    minimum_median_return=0.0,
    minimum_beat_benchmark_rate=0.55,
    minimum_median_excess_return=0.0,
)


if results.empty:

    print("No opportunities found.")

else:

    print(
        results[
            [
                "Rank",
                "Ticker",
                "Upcoming Entry Date",
                "Holding Days",
                "Sample Size",
                "Win Rate",
                "Median Return",
                "Beat SPY Rate",
                "Median Excess Return",
                "Wilson Lower Bound",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


if not errors.empty:

    print()
    print("ERRORS")
    print(errors.head(20))
def main():
    results, errors = discover_upcoming_windows(
        tickers=tickers,
        start_date=date.today(),
        days_ahead=7,
        holding_periods=[5, 10, 15, 20],
        lookback_years=15,
        minimum_win_rate=0.65,
        minimum_sample_size=10,
        minimum_median_return=0.0,
        minimum_beat_benchmark_rate=0.55,
        minimum_median_excess_return=0.0,
    )

    if results.empty:
        print("No opportunities found.")
    else:
        print(results.head(30).to_string(index=False))


if __name__ == "__main__":
    main()