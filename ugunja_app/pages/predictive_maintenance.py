"""Predictive maintenance page rendering."""

import plotly.express as px
import streamlit as st


def render(maintenance_df):
    st.title("🔧 Predictive Maintenance Insights")
    st.markdown(
        "Preventing costly breakdowns *before* they happen. This is our 'insurance policy' for the fleet."
    )

    with st.container(border=True):
        high_risk_vehicles = maintenance_df[
            maintenance_df["predicted_risk_score"] > 0.8
        ]["vehicle_id"].nunique()
        total_fleet_size = maintenance_df["vehicle_id"].nunique()
        avg_fleet_risk = maintenance_df["predicted_risk_score"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Fleet Size", f"{total_fleet_size} Vehicles")
        col2.metric("🔴 Vehicles at High Risk (>80%)", f"{high_risk_vehicles}")
        col3.metric("Avg. Fleet Risk Score", f"{avg_fleet_risk:.2%}")

    with st.container(border=True):
        st.header("At-a-Glance Fleet Risk Overview (Treemap)")
        st.markdown(
            "See your entire fleet in one view. **Size** = Vehicle Mileage (how much it works). "
            "**Color** = Risk Score (how likely it is to fail)."
        )

        latest_fleet_data = maintenance_df.sort_values("date").drop_duplicates(
            "vehicle_id", keep="last"
        )
        fig_tree = px.treemap(
            latest_fleet_data,
            path=[px.Constant("All Vehicles"), "predicted_failure_type", "vehicle_id"],
            values="mileage",
            color="predicted_risk_score",
            color_continuous_scale="OrRd",
            hover_data={
                "predicted_failure_type": True,
                "predicted_risk_score": ":.2f",
                "mileage": True,
            },
        )
        fig_tree.update_layout(
            margin=dict(t=25, l=25, r=25, b=25), template="plotly_dark"
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    with st.container(border=True):
        st.header("Finding the 'Needle in the Haystack'")
        st.markdown(
            "Our model spots the *one* bad truck out of the *entire* healthy fleet."
        )

        tab1, tab2 = st.tabs(
            ["Fleet-Wide Anomaly (Scatter)", "Risk Distribution (Box Plot)"]
        )

        with tab1:
            fig_scatter = px.scatter(
                maintenance_df,
                x="avg_fuel_consumption_l_100km",
                y="predicted_risk_score",
                color="vehicle_id",
                hover_data=["date", "predicted_failure_type"],
                title="Outlier Detection: Fuel Use vs. Risk",
            )
            fig_scatter.update_layout(template="plotly_dark")
            st.plotly_chart(fig_scatter, use_container_width=True)

        with tab2:
            fig_box = px.box(
                maintenance_df,
                x="vehicle_id",
                y="predicted_risk_score",
                color="vehicle_id",
                title="Vehicle Risk Score Distribution",
            )
            fig_box.update_layout(template="plotly_dark")
            st.plotly_chart(fig_box, use_container_width=True)

    with st.container(border=True):
        st.header("Vehicle-Specific Anomaly Trend (The 'Proof')")
        all_vehicles = maintenance_df["vehicle_id"].unique()
        anomalous_vehicle = (
            "KDC 456Y" if "KDC 456Y" in all_vehicles else all_vehicles[0]
        )

        selected_vehicle = st.selectbox(
            "Select a Vehicle to Analyze",
            options=all_vehicles,
            index=list(all_vehicles).index(anomalous_vehicle),
        )

        vehicle_df = maintenance_df[maintenance_df["vehicle_id"] == selected_vehicle]

        col1, col2 = st.columns(2)
        with col1:
            fig_fuel_trend = px.line(
                vehicle_df,
                x="date",
                y="avg_fuel_consumption_l_100km",
                title="Fuel Consumption Trend",
            )
            fig_fuel_trend.update_layout(template="plotly_dark")
            st.plotly_chart(fig_fuel_trend, use_container_width=True)
        with col2:
            fig_risk_trend = px.line(
                vehicle_df,
                x="date",
                y="predicted_risk_score",
                title="Predicted Risk Score Trend",
            )
            fig_risk_trend.update_yaxes(range=[0, 1])
            fig_risk_trend.update_layout(template="plotly_dark")
            st.plotly_chart(fig_risk_trend, use_container_width=True)
