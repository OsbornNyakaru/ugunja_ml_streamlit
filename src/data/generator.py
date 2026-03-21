"""Synthetic data generation for Project Ugunja."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.geo import DEPOTS, DESTINATIONS, ROAD_CONDITIONS, haversine_km

RANDOM_STATE = 42
N_ROWS = 10_000
N_VEHICLES = 80
N_REGIONS = 6


def generate_vehicle_telemetry_basic(
    n_rows: int = N_ROWS,
    n_vehicles: int = N_VEHICLES,
    n_regions: int = N_REGIONS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate synthetic telemetry for predictive maintenance + demand."""
    rng = np.random.default_rng(random_state)

    vehicle_ids = [f"KCH-{str(i).zfill(3)}" for i in range(1, n_vehicles + 1)]
    vehicle_types = ["Heavy Tanker", "Medium Tanker", "Light Delivery", "Cylinder Truck"]
    regions = [f"Region_{chr(65 + i)}" for i in range(n_regions)]

    vehicle_id_col = rng.choice(vehicle_ids, size=n_rows)
    vehicle_type_col = rng.choice(vehicle_types, size=n_rows, p=[0.3, 0.3, 0.25, 0.15])
    region_col = rng.choice(regions, size=n_rows)

    start_date = datetime(2022, 1, 1)
    day_offsets = rng.integers(0, 730, size=n_rows)
    timestamps = [start_date + timedelta(days=int(d)) for d in day_offsets]

    type_mileage_factor = {
        "Heavy Tanker": 1.4,
        "Medium Tanker": 1.1,
        "Light Delivery": 0.8,
        "Cylinder Truck": 0.7,
    }
    base_mileage = rng.uniform(5_000, 250_000, size=n_rows)
    mileage_factor = np.array([type_mileage_factor[vt] for vt in vehicle_type_col])
    mileage_km = (base_mileage * mileage_factor).clip(1_000, 400_000)

    capacity_map = {
        "Heavy Tanker": (15_000, 30_000),
        "Medium Tanker": (8_000, 15_000),
        "Light Delivery": (2_000, 8_000),
        "Cylinder Truck": (1_000, 5_000),
    }
    load_weight_kg = np.array([rng.uniform(*capacity_map[vt]) for vt in vehicle_type_col])

    mileage_norm = mileage_km / mileage_km.max()
    load_norm = load_weight_kg / load_weight_kg.max()
    engine_temp_c = (
        75
        + 40 * mileage_norm
        + 20 * load_norm
        + 15 * mileage_norm * load_norm
        + rng.normal(0, 5, size=n_rows)
    ).clip(60, 135)

    vibration_level = (
        0.5 + 2.5 * mileage_norm + 1.0 * load_norm + rng.normal(0, 0.3, size=n_rows)
    ).clip(0.1, 5.0)

    days_since_service_base = rng.integers(0, 180, size=n_rows)
    last_service_date = [
        ts - timedelta(days=int(d))
        for ts, d in zip(timestamps, days_since_service_base)
    ]

    fuel_efficiency = (
        12.0
        - 4.0 * mileage_norm
        - 2.0 * (engine_temp_c / 135)
        + rng.normal(0, 0.4, size=n_rows)
    ).clip(3.0, 14.0)

    load_cycles = rng.integers(1, 500, size=n_rows)

    risk_score = (
        0.35 * mileage_norm
        + 0.25 * (vibration_level / 5.0)
        + 0.20 * ((engine_temp_c - 60) / 75)
        + 0.15 * (days_since_service_base / 180)
        + 0.05 * (load_cycles / 500)
        + rng.normal(0, 0.05, size=n_rows)
    )
    maintenance_required = (risk_score > np.percentile(risk_score, 68)).astype(int)

    shortage_risk = (
        0.4 * maintenance_required
        + 0.3 * (load_weight_kg / load_weight_kg.max())
        + 0.3 * rng.uniform(0, 1, size=n_rows)
    )
    regional_shortage = (shortage_risk > np.percentile(shortage_risk, 60)).astype(int)

    df = pd.DataFrame(
        {
            "vehicle_id": vehicle_id_col,
            "vehicle_type": vehicle_type_col,
            "region": region_col,
            "timestamp": timestamps,
            "mileage_km": mileage_km.round(1),
            "load_weight_kg": load_weight_kg.round(1),
            "engine_temp_c": engine_temp_c.round(2),
            "vibration_level": vibration_level.round(3),
            "last_service_date": last_service_date,
            "fuel_efficiency": fuel_efficiency.round(2),
            "load_cycles": load_cycles,
            "maintenance_required": maintenance_required,
            "regional_shortage": regional_shortage,
        }
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["last_service_date"] = pd.to_datetime(df["last_service_date"])
    return df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)


def generate_vehicle_telemetry_route(
    n_rows: int = N_ROWS,
    n_vehicles: int = N_VEHICLES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate synthetic telemetry with route and geo fields."""
    rng = np.random.default_rng(random_state)

    depot_ids = list(DEPOTS.keys())
    dest_ids = list(DESTINATIONS.keys())
    vehicle_ids = [f"KDH-{str(i).zfill(3)}" for i in range(1, n_vehicles + 1)]
    vehicle_types = ["Heavy Tanker", "Medium Tanker", "Light Delivery", "Cylinder Truck"]
    regions = [f"Region_{chr(65 + i)}" for i in range(N_REGIONS)]

    vehicle_id_col = rng.choice(vehicle_ids, size=n_rows)
    vehicle_type_col = rng.choice(vehicle_types, size=n_rows, p=[0.30, 0.30, 0.25, 0.15])
    region_col = rng.choice(regions, size=n_rows)
    depot_id_col = rng.choice(depot_ids, size=n_rows)
    dest_id_col = rng.choice(dest_ids, size=n_rows)

    start_date = datetime(2022, 1, 1)
    day_offsets = rng.integers(0, 730, size=n_rows)
    timestamps = [start_date + timedelta(days=int(d)) for d in day_offsets]

    type_mileage_factor = {
        "Heavy Tanker": 1.4,
        "Medium Tanker": 1.1,
        "Light Delivery": 0.8,
        "Cylinder Truck": 0.7,
    }
    base_mileage = rng.uniform(5_000, 250_000, size=n_rows)
    mileage_factor = np.array([type_mileage_factor[vt] for vt in vehicle_type_col])
    mileage_km = (base_mileage * mileage_factor).clip(1_000, 400_000)

    capacity_map = {
        "Heavy Tanker": (15_000, 30_000),
        "Medium Tanker": (8_000, 15_000),
        "Light Delivery": (2_000, 8_000),
        "Cylinder Truck": (1_000, 5_000),
    }
    load_weight_kg = np.array([rng.uniform(*capacity_map[vt]) for vt in vehicle_type_col])

    mileage_norm = mileage_km / mileage_km.max()
    load_norm = load_weight_kg / load_weight_kg.max()
    engine_temp_c = (
        75
        + 40 * mileage_norm
        + 20 * load_norm
        + 15 * mileage_norm * load_norm
        + rng.normal(0, 5, size=n_rows)
    ).clip(60, 135)

    vibration_level = (
        0.5 + 2.5 * mileage_norm + 1.0 * load_norm + rng.normal(0, 0.3, size=n_rows)
    ).clip(0.1, 5.0)

    days_since_service_base = rng.integers(0, 180, size=n_rows)
    last_service_date = [
        ts - timedelta(days=int(d))
        for ts, d in zip(timestamps, days_since_service_base)
    ]

    fuel_efficiency = (
        12.0
        - 4.0 * mileage_norm
        - 2.0 * (engine_temp_c / 135)
        + rng.normal(0, 0.4, size=n_rows)
    ).clip(3.0, 14.0)

    load_cycles = rng.integers(1, 500, size=n_rows)

    depot_lats = np.array([DEPOTS[d]["lat"] for d in depot_id_col])
    depot_lons = np.array([DEPOTS[d]["lon"] for d in depot_id_col])
    vehicle_lat = depot_lats + rng.uniform(-0.8, 0.8, size=n_rows)
    vehicle_lon = depot_lons + rng.uniform(-0.8, 0.8, size=n_rows)

    dest_lats = np.array([DESTINATIONS[d]["lat"] for d in dest_id_col])
    dest_lons = np.array([DESTINATIONS[d]["lon"] for d in dest_id_col])

    distance_km = np.array(
        [
            haversine_km(
                DEPOTS[dep]["lat"],
                DEPOTS[dep]["lon"],
                DESTINATIONS[dst]["lat"],
                DESTINATIONS[dst]["lon"],
            )
            for dep, dst in zip(depot_id_col, dest_id_col)
        ]
    )

    road_condition_score = np.array(
        [ROAD_CONDITIONS.get(f"{dep}-{dst}", 0.7) for dep, dst in zip(depot_id_col, dest_id_col)]
    )

    delivery_window_hours = np.where(
        distance_km > 200, rng.integers(6, 14, size=n_rows), rng.integers(2, 8, size=n_rows)
    ).astype(float)

    route_frequency = rng.integers(1, 50, size=n_rows)

    risk_score = (
        0.35 * mileage_norm
        + 0.25 * (vibration_level / 5.0)
        + 0.20 * ((engine_temp_c - 60) / 75)
        + 0.15 * (days_since_service_base / 180)
        + 0.05 * (load_cycles / 500)
        + rng.normal(0, 0.05, size=n_rows)
    )
    maintenance_required = (risk_score > np.percentile(risk_score, 68)).astype(int)

    route_feasibility = (distance_km / distance_km.max()) * (1 - road_condition_score)
    shortage_risk = (
        0.30 * maintenance_required
        + 0.25 * (load_weight_kg / load_weight_kg.max())
        + 0.25 * route_feasibility
        + 0.20 * rng.uniform(0, 1, size=n_rows)
    )
    regional_shortage = (shortage_risk > np.percentile(shortage_risk, 60)).astype(int)

    delay_risk = (
        0.30 * maintenance_required
        + 0.25 * (1 - road_condition_score)
        + 0.20 * (distance_km / distance_km.max())
        + 0.15 * (1 - delivery_window_hours / delivery_window_hours.max())
        + 0.10 * rng.uniform(0, 1, size=n_rows)
    )
    delivery_delay_risk = (delay_risk > np.percentile(delay_risk, 65)).astype(int)

    df = pd.DataFrame(
        {
            "vehicle_id": vehicle_id_col,
            "vehicle_type": vehicle_type_col,
            "region": region_col,
            "depot_id": depot_id_col,
            "destination_id": dest_id_col,
            "timestamp": timestamps,
            "mileage_km": mileage_km.round(1),
            "load_weight_kg": load_weight_kg.round(1),
            "engine_temp_c": engine_temp_c.round(2),
            "vibration_level": vibration_level.round(3),
            "last_service_date": last_service_date,
            "fuel_efficiency": fuel_efficiency.round(2),
            "load_cycles": load_cycles,
            "vehicle_lat": vehicle_lat.round(5),
            "vehicle_lon": vehicle_lon.round(5),
            "dest_lat": dest_lats,
            "dest_lon": dest_lons,
            "distance_to_dest_km": distance_km.round(2),
            "road_condition_score": road_condition_score.round(3),
            "delivery_window_hours": delivery_window_hours,
            "route_frequency": route_frequency,
            "maintenance_required": maintenance_required,
            "regional_shortage": regional_shortage,
            "delivery_delay_risk": delivery_delay_risk,
        }
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["last_service_date"] = pd.to_datetime(df["last_service_date"])
    return df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)


def generate_vehicle_telemetry(
    mode: Literal["maintenance", "route"] = "maintenance",
    **kwargs,
) -> pd.DataFrame:
    """Unified wrapper for telemetry generation."""
    if mode == "route":
        return generate_vehicle_telemetry_route(**kwargs)
    return generate_vehicle_telemetry_basic(**kwargs)

