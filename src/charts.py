from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def yearly_return_chart(
    data: pd.DataFrame,
):
    """
    Bar chart showing seasonal return by historical year.
    """

    chart_data = data.copy()

    chart_data["Return %"] = (
        chart_data["Return"] * 100
    )

    fig = px.bar(
        chart_data,
        x="Year",
        y="Return %",
        title="Historical Seasonal Returns",
    )

    fig.add_hline(
        y=0,
        line_width=1,
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Return (%)",
    )

    return fig


def stock_vs_benchmark_chart(
    data: pd.DataFrame,
    benchmark_name: str = "SPY",
):
    """
    Compare seasonal stock returns with benchmark returns.
    """

    benchmark_column = (
        f"{benchmark_name} Return"
    )

    chart = data[
        [
            "Year",
            "Return",
            benchmark_column,
        ]
    ].copy()

    chart["Stock"] = (
        chart["Return"] * 100
    )

    chart[benchmark_name] = (
        chart[benchmark_column] * 100
    )

    fig = go.Figure()

    fig.add_bar(
        x=chart["Year"],
        y=chart["Stock"],
        name="Stock",
    )

    fig.add_bar(
        x=chart["Year"],
        y=chart[benchmark_name],
        name=benchmark_name,
    )

    fig.update_layout(
        title=f"Stock vs {benchmark_name}",
        barmode="group",
        xaxis_title="Year",
        yaxis_title="Return (%)",
    )

    return fig


def excess_return_chart(
    data: pd.DataFrame,
):
    """
    Historical excess return versus benchmark.
    """

    chart = data.copy()

    chart["Excess Return %"] = (
        chart["Excess Return"] * 100
    )

    fig = px.bar(
        chart,
        x="Year",
        y="Excess Return %",
        title="Historical Excess Return",
    )

    fig.add_hline(
        y=0,
        line_width=1,
    )

    return fig