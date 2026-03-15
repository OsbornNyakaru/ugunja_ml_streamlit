"""Route optimization page rendering."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(route_df):
    st.title("🚚 Route Optimization Insights")
    st.markdown(
        "This is the 'Money' dashboard. It proves how our AI directly saves costs by optimizing routes."
    )

    with st.container(border=True):
        st.header("💸 Interactive ROI Calculator")
        st.markdown(
            "Use the sliders to match your business costs and see the **real-time savings** our AI generates."
        )

        savings_df = route_df.groupby("route_type").sum(numeric_only=True)
        total_km_saved = (
            savings_df.loc["Standard_Route"]["total_distance_km"]
            - savings_df.loc["Ugunja_AI_Route"]["total_distance_km"]
        )
        total_hours_saved = (
            savings_df.loc["Standard_Route"]["total_time_hours"]
            - savings_df.loc["Ugunja_AI_Route"]["total_time_hours"]
        )

        col_slider_1, col_slider_2 = st.columns(2)
        with col_slider_1:
            cost_per_km_ksh = st.slider("Fuel & Maint. Cost per KM (KSh)", 20, 100, 40)
        with col_slider_2:
            cost_per_hour_ksh = st.slider(
                "Driver Labor Cost per Hour (KSh)", 300, 1000, 500
            )

        total_fuel_savings = total_km_saved * cost_per_km_ksh
        total_labor_savings = total_hours_saved * cost_per_hour_ksh
        total_savings = total_fuel_savings + total_labor_savings

        st.divider()

        st.subheader("Estimated Total Savings (from this dataset)")
        col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
        col_metric_1.metric(
            label="Fuel/Maint. Savings (KSh)",
            value=f"{total_fuel_savings:,.0f}",
        )
        col_metric_2.metric(
            label="Labor Savings (KSh)",
            value=f"{total_labor_savings:,.0f}",
        )
        col_metric_3.metric(
            label="TOTAL SAVINGS",
            value=f"KSh {total_savings:,.0f}",
        )

    with st.container(border=True):
        st.header("How We Achieve These Savings")
        tab1, tab2 = st.tabs(
            [
                "The 'Financial Breakdown' (Waterfall)",
                "The 'Efficiency Frontier' (Scatter)",
            ]
        )

        with tab1:
            st.markdown(
                "This chart breaks down the *source* of the savings from the calculator above."
            )
            fig_waterfall = go.Figure(
                go.Waterfall(
                    name="Savings",
                    orientation="v",
                    x=[
                        "Fuel/Maint. Savings (from KM)",
                        "Labor Savings (from Hours)",
                        "Total Savings",
                    ],
                    y=[total_fuel_savings, total_labor_savings, total_savings],
                    text=[
                        f"KSh {total_fuel_savings:,.0f}",
                        f"KSh {total_labor_savings:,.0f}",
                        f"KSh {total_savings:,.0f}",
                    ],
                    textposition="auto",
                    measure=["relative", "relative", "total"],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#FFA500"}},
                    totals={"marker": {"color": "#008000"}},
                )
            )
            fig_waterfall.update_layout(
                title="Financial Breakdown of Savings",
                showlegend=False,
                template="plotly_dark",
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

        with tab2:
            st.markdown(
                "This proves our AI is smarter. 'Standard' routes are a scattered cloud (inefficient). "
                "'Ugunja AI' routes form a tight, optimal line (the 'Efficiency Frontier')."
            )
            fig_frontier = px.scatter(
                route_df,
                x="total_distance_km",
                y="total_time_hours",
                color="route_type",
                title="Ugunja AI vs. Standard Routes",
                color_discrete_map={
                    "Standard_Route": "grey",
                    "Ugunja_AI_Route": "#FFA500",
                },
                hover_data=["route_id"],
            )
            fig_frontier.update_layout(template="plotly_dark")
            st.plotly_chart(fig_frontier, use_container_width=True)
