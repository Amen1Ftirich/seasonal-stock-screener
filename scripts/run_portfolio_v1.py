import pandas as pd

from src.portfolio_v1 import (
    run_portfolio_backtest,
    summarize_portfolios,
)


print(
    "Loading frozen V5 OOS predictions..."
)


predictions = pd.read_parquet(
    "data/cache/v5_predictions.parquet"
)


print(
    f"Prediction rows: "
    f"{len(predictions):,}"
)


monthly, weights = (
    run_portfolio_backtest(
        predictions=predictions,

        #
        # Fixed assumption.
        # Do not optimize this against the backtest.
        #
        transaction_cost_bps=10.0,
    )
)


monthly.to_csv(
    "data/cache/portfolio_v1_monthly.csv",
    index=False,
)


weights.to_parquet(
    "data/cache/portfolio_v1_weights.parquet",
    index=False,
)


summary = summarize_portfolios(
    monthly
)


print()
print(
    "======================================"
)

print(
    "PORTFOLIO V1 RESULTS"
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

        value = row[column]

        if column == "Months":

            print(
                f"{column}: "
                f"{int(value)}"
            )

        elif column in percentage_columns:

            print(
                f"{column}: "
                f"{value:.2%}"
            )

        elif column in {
            "Average Estimated Beta",
            "Realized Beta",
            "Sharpe RF=0",
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