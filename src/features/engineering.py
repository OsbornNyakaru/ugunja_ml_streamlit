"""Feature engineering utilities aligned with the notebooks."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from src.utils.geo import DEPOTS, haversine_km


def engineer_features_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for predictive maintenance + demand."""
    df = df.copy()
    df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

    df["days_since_service"] = (
        (df["timestamp"] - df["last_service_date"]).dt.days
    ).clip(0, 365)

    df["stress_index"] = (df["load_weight_kg"] * df["mileage_km"] / 1e9).round(6)

    df["mileage_per_load_cycle"] = (
        df["mileage_km"] / (df["load_cycles"] + 1)
    ).round(2)

    df["temp_vib_interaction"] = (
        df["engine_temp_c"] * df["vibration_level"]
    ).round(3)

    df["vibration_rolling_7d"] = (
        df.groupby("vehicle_id")["vibration_level"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    ).round(4)

    df["engine_temp_rolling_7d"] = (
        df.groupby("vehicle_id")["engine_temp_c"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    ).round(4)

    df["efficiency_degradation"] = (
        14.0 - df["fuel_efficiency"]
    ).clip(0, None).round(3)

    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    return df


def engineer_features_route(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for route optimisation telemetry."""
    df = df.copy()
    df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

    df["days_since_service"] = (
        (df["timestamp"] - df["last_service_date"]).dt.days
    ).clip(0, 365)

    df["stress_index"] = (df["load_weight_kg"] * df["mileage_km"] / 1e9).round(6)

    df["mileage_per_load_cycle"] = (
        df["mileage_km"] / (df["load_cycles"] + 1)
    ).round(2)

    df["temp_vib_interaction"] = (
        df["engine_temp_c"] * df["vibration_level"]
    ).round(3)

    df["vibration_rolling_7d"] = (
        df.groupby("vehicle_id")["vibration_level"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    ).round(4)

    df["engine_temp_rolling_7d"] = (
        df.groupby("vehicle_id")["engine_temp_c"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    ).round(4)

    df["efficiency_degradation"] = (14.0 - df["fuel_efficiency"]).clip(0).round(3)

    df["distance_to_depot_km"] = df.apply(
        lambda r: haversine_km(
            r["vehicle_lat"],
            r["vehicle_lon"],
            DEPOTS[r["depot_id"]]["lat"],
            DEPOTS[r["depot_id"]]["lon"],
        ),
        axis=1,
    ).round(2)

    effective_speed = (
        60
        * df["road_condition_score"]
        * (1 - 0.3 * (df["load_weight_kg"] / df["load_weight_kg"].max()))
    ).clip(15, 80)
    df["estimated_travel_time_hrs"] = (
        df["distance_to_dest_km"] / effective_speed
    ).round(3)

    df["route_congestion_index"] = (
        df["route_frequency"] * (1 - df["road_condition_score"])
    ).round(4)

    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    return df


def engineer_features(
    df: pd.DataFrame,
    mode: Literal["maintenance", "route"] = "maintenance",
) -> pd.DataFrame:
    """Unified wrapper for feature engineering."""
    if mode == "route":
        return engineer_features_route(df)
    return engineer_features_basic(df)

