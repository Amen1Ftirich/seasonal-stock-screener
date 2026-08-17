from src.scanner import scan_tickers


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


for holding_days in [5, 10, 15, 20, 30]:
    results, errors = scan_tickers(
        tickers=tickers,
        entry_month=8,
        entry_day=20,
        holding_days=holding_days,
        lookback_years=15,
        minimum_sample_size=10,
    )

    print()
    print(f"===== {holding_days} DAY HOLD =====")

    if results.empty:
        print("No qualifying results.")
    else:
        print(
            results[
                [
                    "Ticker",
                    "Win Rate",
                    "Median Return",
                    "Median Excess Return",
                    "Wilson Lower Bound",
                ]
            ].head().to_string(index=False)
        )

    if not errors.empty:
        print("\nErrors:")
        print(errors.to_string(index=False))