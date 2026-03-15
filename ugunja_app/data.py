"""Data loading and caching utilities."""

from pathlib import Path

import pandas as pd
import streamlit as st

from .schema import validate_dataframe

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


@st.cache_data
def load_data():
    try:
        demand_df = pd.read_csv(DATA_DIR / "demand_forecasting.csv")
        maintenance_df = pd.read_csv(DATA_DIR / "predictive_maintenance.csv")
        route_df = pd.read_csv(DATA_DIR / "route_optimization.csv")

        demand_df["date"] = pd.to_datetime(demand_df["date"], errors="coerce")
        maintenance_df["date"] = pd.to_datetime(maintenance_df["date"], errors="coerce")

        results = [
            validate_dataframe(
                demand_df,
                "Demand Forecasting",
                required_columns={
                    "date",
                    "neighborhood",
                    "day_of_week",
                    "weather",
                    "is_holiday",
                    "predicted_demand_cylinders",
                },
                date_columns=["date"],
                numeric_columns=["predicted_demand_cylinders"],
            ),
            validate_dataframe(
                maintenance_df,
                "Predictive Maintenance",
                required_columns={
                    "date",
                    "vehicle_id",
                    "mileage",
                    "avg_engine_temp_c",
                    "avg_fuel_consumption_l_100km",
                    "predicted_risk_score",
                    "predicted_failure_type",
                },
                date_columns=["date"],
                numeric_columns=[
                    "mileage",
                    "avg_engine_temp_c",
                    "avg_fuel_consumption_l_100km",
                    "predicted_risk_score",
                ],
            ),
            validate_dataframe(
                route_df,
                "Route Optimization",
                required_columns={
                    "route_id",
                    "route_type",
                    "num_stops",
                    "total_distance_km",
                    "total_time_hours",
                },
                numeric_columns=["num_stops", "total_distance_km", "total_time_hours"],
            ),
        ]

        has_errors = False
        for result in results:
            for warning in result.warnings:
                st.warning(warning)
            for error in result.errors:
                st.error(error)
                has_errors = True

        if has_errors:
            return None, None, None

        return demand_df, maintenance_df, route_df
    except FileNotFoundError:
        st.error("FATAL ERROR: Make sure all three CSV files are in the data folder")
        return None, None, None
