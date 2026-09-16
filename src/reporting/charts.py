from __future__ import annotations
import pandas as pd

try:
    import plotly.express as px
except ImportError:
    px = None


def revenue_trend_chart(monthly_data: pd.DataFrame | None):
    if monthly_data is None or px is None:
        return None
    fig = px.line(monthly_data, x="period", y="revenue", markers=True, title="Monthly Revenue Trend")
    fig.update_layout(xaxis_title="Month", yaxis_title="Revenue")
    return fig


def revenue_by_dimension(data: pd.DataFrame | None, dimension_col: str, top_n: int = 15):
    if data is None or px is None or dimension_col not in data.columns or "revenue" not in data.columns:
        return None
    fig = px.bar(data.head(top_n), x=dimension_col, y="revenue", title=f"Revenue by {dimension_col.title()}")
    return fig


def pareto_chart(pareto_df: pd.DataFrame | None, dimension_col: str):
    if pareto_df is None or px is None:
        return None
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_bar(x=pareto_df[dimension_col], y=pareto_df["revenue"], name="Revenue")
    fig.add_scatter(
        x=pareto_df[dimension_col], y=pareto_df["cumulative_contribution_pct"],
        name="Cumulative %", yaxis="y2", mode="lines+markers",
    )
    fig.update_layout(
        title=f"Pareto Analysis — {dimension_col.title()}",
        yaxis=dict(title="Revenue"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
    )
    return fig


def forecast_chart(historical: pd.DataFrame | None, forecast: pd.DataFrame | None):
    if px is None:
        return None
    import plotly.graph_objects as go
    fig = go.Figure()
    if historical is not None:
        fig.add_scatter(x=historical["period"], y=historical["revenue"], name="Actual", mode="lines")
    if forecast is not None:
        fig.add_scatter(x=forecast["period"], y=forecast["forecast_revenue"], name="Forecast", mode="lines", line=dict(dash="dash"))
    fig.update_layout(title="Revenue: Actual vs Forecast", xaxis_title="Date", yaxis_title="Revenue")
    return fig
