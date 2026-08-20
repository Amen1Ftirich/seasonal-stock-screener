import pandas as pd

from src.portfolio_v8c import (
    run_v8c_portfolios,
    summarize_v8c_portfolios,
)


predictions = pd.read_parquet(
    "data/cache/"
    "v8c_predictions.parquet"
)


monthly = run_v8c_portfolios(
    predictions=predictions,

    #
    # Fixed research assumption.
    #
    transaction_cost_bps=10.0,
)


monthly.to_csv(
    "data/cache/"
    "v8c_portfolio_monthly.csv",
    index=False,
)


summary = summarize_v8c_portfolios(
    monthly
)


summary.to_csv(
    "data/cache/"
    "v8c_portfolio_summary.csv",
    index=False,
)


print()
print(
    "======================================"
)
print(
    "V8C PORTFOLIO RESULTS"
)
print(
    "======================================"
)


percentage_columns = {
    "Net CAGR",
    "SPY CAGR",
    "Annualized Volatility",
    "Maximum Drawdown",
    "Average Monthly Excess",
    "Beat SPY Rate",
    "Average Gross Turnover",
    "Average Monthly Cost",
    "Average Maximum Weight",
}


for _, row in summary.iterrows():

    print()
    print(
        f"--- {row['Strategy']} ---"
    )


    for column in summary.columns:

        if column == "Strategy":
            continue


        value = row[
            column
        ]


        if column == "Months":

            print(
                f"{column}: "
                f"{int(value)}"
            )


        elif column in (
            percentage_columns
        ):

            print(
                f"{column}: "
                f"{value:.2%}"
            )


        elif column in {
            "Sharpe RF=0",
            "Average Estimated Beta",
            "Realized Beta",
            "Average Holdings",
            "Average Effective Holdings",
        }:

            print(
                f"{column}: "
                f"{value:.2f}"
            )


        else:

            print(
                f"{column}: "
                f"{value}"
            )