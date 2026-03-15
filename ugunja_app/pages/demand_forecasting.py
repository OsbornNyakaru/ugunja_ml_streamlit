"""Demand forecasting page rendering."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st


def render(demand_df):
    st.title("📈 Demand Forecasting Insights")
    st.markdown(
        "Predicting demand to move from a *reactive* to a *predictive* supply chain."
    )

    with st.container(border=True):
        total_predicted = demand_df["predicted_demand_cylinders"].sum()
        highest_demand_zone = (
            demand_df.groupby("neighborhood")["predicted_demand_cylinders"]
            .sum()
            .idxmax()
        )
        peak_day = (
            demand_df.groupby("day_of_week")["predicted_demand_cylinders"]
            .sum()
            .idxmax()
        )

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Total Predicted Demand (Dataset)", f"{total_predicted:,} Cylinders"
        )
        col2.metric("Projected Hotspot", highest_demand_zone)
        col3.metric("Projected Peak Day", peak_day)

    with st.container(border=True):
        st.header("Daily Demand Pattern (Calendar Heatmap)")
        st.markdown(
            "This shows daily demand intensity. Our model sees that demand peaks on weekends and holidays."
        )

        try:
            daily_demand = (
                demand_df.groupby("date")["predicted_demand_cylinders"]
                .sum()
                .reset_index()
            )
            daily_demand["day_of_week"] = daily_demand["date"].dt.day_name()
            daily_demand["week_of_year"] = daily_demand["date"].dt.isocalendar().week
            daily_demand["month_name"] = daily_demand["date"].dt.month_name()
            daily_demand["month_num"] = daily_demand["date"].dt.month

            daily_demand = daily_demand.sort_values("month_num")

            fig_cal = px.density_heatmap(
                daily_demand,
                x="week_of_year",
                y="day_of_week",
                z="predicted_demand_cylinders",
                histfunc="avg",
                title="Average Predicted Demand by Day of Week",
                color_continuous_scale="OrRd",
                facet_col="month_name",
                facet_col_wrap=4,
                labels={"predicted_demand_cylinders": "Avg Demand"},
            )
            fig_cal.update_layout(
                coloraxis_showscale=False, template="plotly_dark", height=700
            )
            st.plotly_chart(fig_cal, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not generate calendar heatmap: {exc}")

    with st.container(border=True):
        st.header("Understanding the 'Why' of Demand")
        tab1, tab2 = st.tabs(["Time Series Decomposition", "Demand Driver Correlation"])

        with tab1:
            st.markdown(
                "We break down demand into its core components: the **Trend** (long-term), "
                "**Seasonality** (weekly pattern), and **Residuals** (noise)."
            )
            daily_demand_ts = (
                demand_df.groupby("date")["predicted_demand_cylinders"]
                .sum()
                .reset_index()
                .set_index("date")
            )
            if not daily_demand_ts.empty and len(daily_demand_ts) > 14:
                decomposition = seasonal_decompose(
                    daily_demand_ts["predicted_demand_cylinders"],
                    model="additive",
                    period=7,
                )
                fig_decomp = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=(
                        "1. Observed",
                        "2. Trend",
                        "3. Seasonal (Weekly)",
                        "4. Residual",
                    ),
                )
                fig_decomp.add_trace(
                    go.Scatter(
                        x=decomposition.observed.index,
                        y=decomposition.observed,
                        mode="lines",
                        name="Observed",
                    ),
                    row=1,
                    col=1,
                )
                fig_decomp.add_trace(
                    go.Scatter(
                        x=decomposition.trend.index,
                        y=decomposition.trend,
                        mode="lines",
                        name="Trend",
                        line=dict(color="orange"),
                    ),
                    row=2,
                    col=1,
                )
                fig_decomp.add_trace(
                    go.Scatter(
                        x=decomposition.seasonal.index,
                        y=decomposition.seasonal,
                        mode="lines",
                        name="Seasonal",
                        line=dict(color="green"),
                    ),
                    row=3,
                    col=1,
                )
                fig_decomp.add_trace(
                    go.Scatter(
                        x=decomposition.resid.index,
                        y=decomposition.resid,
                        mode="markers",
                        name="Residual",
                        marker=dict(size=2, color="gray"),
                    ),
                    row=4,
                    col=1,
                )
                fig_decomp.update_layout(
                    height=700,
                    showlegend=False,
                    margin=dict(t=100),
                    template="plotly_dark",
                )
                st.plotly_chart(fig_decomp, use_container_width=True)

        with tab2:
            st.markdown(
                "This matrix shows what drives demand. A high number (e.g., 0.8) means a strong link. "
                "**(Weekend = 6, 7)**."
            )
            df_corr = demand_df.copy()
            day_map = {
                "Monday": 1,
                "Tuesday": 2,
                "Wednesday": 3,
                "Thursday": 4,
                "Friday": 5,
                "Saturday": 6,
                "Sunday": 7,
            }
            df_corr["day_of_week_num"] = df_corr["day_of_week"].map(day_map)
            weather_map = {"Rainy": 0, "Cloudy": 1, "Sunny": 2}
            df_corr["weather_num"] = df_corr["weather"].map(weather_map)

            corr_df = df_corr[
                [
                    "predicted_demand_cylinders",
                    "day_of_week_num",
                    "weather_num",
                    "is_holiday",
                ]
            ]
            corr_matrix = corr_df.corr()

            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="OrRd",
                labels=dict(color="Correlation"),
                title="Correlation: What Drives Demand?",
            )
            fig_heatmap.update_xaxes(side="top")
            fig_heatmap.update_layout(template="plotly_dark")
            st.plotly_chart(fig_heatmap, use_container_width=True)
