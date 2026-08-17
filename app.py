from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.benchmark import add_benchmark_returns
from src.charts import (
    excess_return_chart,
    stock_vs_benchmark_chart,
    yearly_return_chart,
)
from src.data import get_price_history
from src.discovery import discover_upcoming_windows
from src.metrics import (
    calculate_metrics,
    calculate_relative_metrics,
)
from src.seasonality import get_seasonal_returns
from src.universe import (
    TECH_TEST_UNIVERSE,
    get_sp500_tickers,
)


st.set_page_config(
    page_title="Seasonal Stock Screener",
    page_icon="📈",
    layout="wide",
)


st.title("Seasonal Stock Screener")

st.caption(
    "Find upcoming calendar windows with historically "
    "strong and repeatable stock performance."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Screener Settings")


universe_choice = st.sidebar.selectbox(
    "Universe",
    [
        "Tech Test",
        "S&P 500",
        "Custom",
    ],
)


if universe_choice == "Tech Test":

    tickers = TECH_TEST_UNIVERSE


elif universe_choice == "S&P 500":

    try:

        tickers = get_sp500_tickers()

        st.sidebar.success(
            f"{len(tickers)} stocks loaded"
        )

    except Exception as exc:

        st.sidebar.error(
            f"Could not load S&P 500: {exc}"
        )

        tickers = TECH_TEST_UNIVERSE


else:

    custom_input = st.sidebar.text_area(
        "Tickers",
        value="AAPL, MSFT, NVDA, AMD, AVGO",
    )

    tickers = [
        ticker.strip().upper()
        for ticker in custom_input.split(",")
        if ticker.strip()
    ]


lookback_years = st.sidebar.selectbox(
    "Historical Lookback",
    [
        10,
        15,
        20,
    ],
    index=1,
)


days_ahead = st.sidebar.selectbox(
    "Search Upcoming",
    [
        7,
        14,
        30,
    ],
    index=1,
)


holding_periods = st.sidebar.multiselect(
    "Holding Periods",
    [
        5,
        10,
        15,
        20,
        30,
    ],
    default=[
        10,
        15,
        20,
    ],
)


st.sidebar.subheader("Quality Filters")


minimum_win_rate = (
    st.sidebar.slider(
        "Minimum Win Rate",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
    )
    / 100
)


minimum_sample_size = st.sidebar.slider(
    "Minimum Sample Size",
    min_value=5,
    max_value=20,
    value=10,
)


minimum_median_return = (
    st.sidebar.slider(
        "Minimum Median Return",
        min_value=-5.0,
        max_value=10.0,
        value=1.0,
        step=0.5,
    )
    / 100
)


minimum_beat_spy = (
    st.sidebar.slider(
        "Minimum Beat SPY Rate",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
    )
    / 100
)


minimum_excess_return = (
    st.sidebar.slider(
        "Minimum Median Excess Return",
        min_value=-5.0,
        max_value=10.0,
        value=0.5,
        step=0.5,
    )
    / 100
)


run_scan = st.sidebar.button(
    "Find Opportunities",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# Run scanner
# --------------------------------------------------

if run_scan:

    if not holding_periods:

        st.error(
            "Select at least one holding period."
        )

        st.stop()

    with st.spinner(
        f"Scanning {len(tickers)} stocks..."
    ):

        try:

            results, errors = (
                discover_upcoming_windows(
                    tickers=tickers,
                    start_date=date.today(),
                    days_ahead=days_ahead,
                    holding_periods=holding_periods,
                    lookback_years=lookback_years,
                    benchmark_name="SPY",

                    minimum_win_rate=(
                        minimum_win_rate
                    ),

                    minimum_sample_size=(
                        minimum_sample_size
                    ),

                    minimum_median_return=(
                        minimum_median_return
                    ),

                    minimum_beat_benchmark_rate=(
                        minimum_beat_spy
                    ),

                    minimum_median_excess_return=(
                        minimum_excess_return
                    ),
                )
            )

            st.session_state[
                "scan_results"
            ] = results

            st.session_state[
                "scan_errors"
            ] = errors

        except Exception as exc:

            st.error(
                f"Screener failed: {exc}"
            )


# --------------------------------------------------
# Results
# --------------------------------------------------

results = st.session_state.get(
    "scan_results"
)


if results is None:

    st.info(
        "Choose your filters and click "
        "'Find Opportunities' to begin."
    )

    st.stop()


if results.empty:

    st.warning(
        "No opportunities passed the current filters."
    )

    st.stop()


st.subheader("Top Seasonal Opportunities")


display_columns = [
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
    "Worst Return",
]


available_columns = [
    column
    for column in display_columns
    if column in results.columns
]


display = results[
    available_columns
].copy()


st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,

    column_config={

        "Upcoming Entry Date":
            st.column_config.DateColumn(
                "Entry",
                format="MMM D, YYYY",
            ),

        "Win Rate":
            st.column_config.NumberColumn(
                "Win Rate",
                format="%.1%%",
            ),

        "Median Return":
            st.column_config.NumberColumn(
                "Median Return",
                format="%.2%%",
            ),

        "Beat SPY Rate":
            st.column_config.NumberColumn(
                "Beat SPY",
                format="%.1%%",
            ),

        "Median Excess Return":
            st.column_config.NumberColumn(
                "Median Excess",
                format="%.2%%",
            ),

        "Wilson Lower Bound":
            st.column_config.NumberColumn(
                "Confidence Floor",
                format="%.1%%",
            ),

        "Worst Return":
            st.column_config.NumberColumn(
                "Worst Year",
                format="%.2%%",
            ),
    },
)


csv = results.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    "Download Results CSV",
    csv,
    "seasonal_opportunities.csv",
    "text/csv",
)


# --------------------------------------------------
# Detailed Opportunity
# --------------------------------------------------

st.divider()

st.subheader("Opportunity Analysis")


options = []

for index, row in results.head(100).iterrows():

    entry_date = pd.Timestamp(
        row["Upcoming Entry Date"]
    )

    label = (
        f"#{row['Rank']} "
        f"{row['Ticker']} | "
        f"{entry_date:%b %d} | "
        f"{int(row['Holding Days'])} sessions"
    )

    options.append(
        (
            label,
            index,
        )
    )


selected_label = st.selectbox(
    "Select opportunity",
    [
        item[0]
        for item in options
    ],
)


selected_index = next(
    index
    for label, index in options
    if label == selected_label
)


selected = results.loc[
    selected_index
]


ticker = selected["Ticker"]

entry_date = pd.Timestamp(
    selected["Upcoming Entry Date"]
)

holding_days = int(
    selected["Holding Days"]
)


# --------------------------------------------------
# Reconstruct historical observations
# --------------------------------------------------

stock_prices = get_price_history(
    ticker
)

spy_prices = get_price_history(
    "SPY"
)


seasonal = get_seasonal_returns(
    prices=stock_prices,
    entry_month=entry_date.month,
    entry_day=entry_date.day,
    holding_days=holding_days,
    lookback_years=lookback_years,
)


comparison = add_benchmark_returns(
    seasonal_returns=seasonal,
    benchmark_prices=spy_prices,
    benchmark_name="SPY",
)


absolute = calculate_metrics(
    comparison
)

relative = calculate_relative_metrics(
    comparison,
    benchmark_name="SPY",
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

st.markdown(
    f"### {ticker}"
)

st.caption(
    f"Historical seasonal behavior around "
    f"{entry_date:%B %d}, held for "
    f"{holding_days} trading sessions."
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Win Rate",
    f"{absolute['win_rate']:.1%}",
)


col2.metric(
    "Median Return",
    f"{absolute['median_return']:.2%}",
)


col3.metric(
    "Beat SPY",
    f"{relative['beat_benchmark_rate']:.1%}",
)


col4.metric(
    "Median Excess",
    f"{relative['median_excess_return']:.2%}",
)


col5, col6, col7, col8 = st.columns(4)


col5.metric(
    "Sample Size",
    absolute["sample_size"],
)


col6.metric(
    "Confidence Floor",
    f"{absolute['wilson_lower_bound']:.1%}",
)


col7.metric(
    "Best Year",
    f"{absolute['best_return']:.2%}",
)


col8.metric(
    "Worst Year",
    f"{absolute['worst_return']:.2%}",
)


# --------------------------------------------------
# Charts
# --------------------------------------------------

st.plotly_chart(
    yearly_return_chart(
        comparison
    ),
    use_container_width=True,
)


st.plotly_chart(
    stock_vs_benchmark_chart(
        comparison,
        benchmark_name="SPY",
    ),
    use_container_width=True,
)


st.plotly_chart(
    excess_return_chart(
        comparison
    ),
    use_container_width=True,
)


# --------------------------------------------------
# Historical observations
# --------------------------------------------------

st.subheader(
    "Historical Observations"
)


history = comparison[
    [
        "Year",
        "Entry Date",
        "Exit Date",
        "Return",
        "SPY Return",
        "Excess Return",
        "Win",
        "Beat SPY",
    ]
].copy()


st.dataframe(
    history,
    use_container_width=True,
    hide_index=True,

    column_config={

        "Return":
            st.column_config.NumberColumn(
                "Return",
                format="%.2%%",
            ),

        "SPY Return":
            st.column_config.NumberColumn(
                "SPY",
                format="%.2%%",
            ),

        "Excess Return":
            st.column_config.NumberColumn(
                "Excess",
                format="%.2%%",
            ),
    },
)


errors = st.session_state.get(
    "scan_errors"
)


if (
    errors is not None
    and not errors.empty
):

    with st.expander(
        "Data / scanning errors"
    ):

        st.dataframe(
            errors,
            use_container_width=True,
        )