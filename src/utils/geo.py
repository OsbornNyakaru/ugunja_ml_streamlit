"""Geospatial utilities and depot/destination configuration."""

from __future__ import annotations

from typing import Dict

import numpy as np

DEFAULT_RANDOM_STATE = 42

DEPOTS: Dict[str, Dict[str, float | str]] = {
    "DEPOT_NBI": {"name": "Nairobi Hub", "lat": -1.286, "lon": 36.817},
    "DEPOT_MSA": {"name": "Mombasa Terminal", "lat": -4.043, "lon": 39.668},
    "DEPOT_KSM": {"name": "Kisumu Depot", "lat": -0.091, "lon": 34.768},
    "DEPOT_ELD": {"name": "Eldoret Station", "lat": 0.520, "lon": 35.270},
}

DESTINATIONS: Dict[str, Dict[str, float | str]] = {
    "DEST_01": {"name": "Thika", "lat": -1.033, "lon": 37.083, "depot": "DEPOT_NBI"},
    "DEST_02": {"name": "Nakuru", "lat": -0.303, "lon": 36.080, "depot": "DEPOT_NBI"},
    "DEST_03": {"name": "Machakos", "lat": -1.519, "lon": 37.263, "depot": "DEPOT_NBI"},
    "DEST_04": {"name": "Malindi", "lat": -3.218, "lon": 40.116, "depot": "DEPOT_MSA"},
    "DEST_05": {"name": "Kilifi", "lat": -3.630, "lon": 39.850, "depot": "DEPOT_MSA"},
    "DEST_06": {"name": "Voi", "lat": -3.396, "lon": 38.556, "depot": "DEPOT_MSA"},
    "DEST_07": {"name": "Kisii", "lat": -0.682, "lon": 34.766, "depot": "DEPOT_KSM"},
    "DEST_08": {"name": "Homa Bay", "lat": -0.527, "lon": 34.457, "depot": "DEPOT_KSM"},
    "DEST_09": {"name": "Bungoma", "lat": 0.564, "lon": 34.559, "depot": "DEPOT_ELD"},
    "DEST_10": {"name": "Kitale", "lat": 1.015, "lon": 35.006, "depot": "DEPOT_ELD"},
    "DEST_11": {"name": "Kericho", "lat": -0.369, "lon": 35.284, "depot": "DEPOT_ELD"},
    "DEST_12": {"name": "Nanyuki", "lat": 0.007, "lon": 37.074, "depot": "DEPOT_NBI"},
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth (km)."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def build_road_conditions(random_state: int = DEFAULT_RANDOM_STATE) -> Dict[str, float]:
    """Generate road condition scores per depot-destination pair."""
    rng = np.random.default_rng(random_state)
    return {
        f"{d}-{dest}": round(rng.uniform(0.3, 1.0), 2)
        for d in DEPOTS
        for dest in DESTINATIONS
    }


ROAD_CONDITIONS: Dict[str, float] = build_road_conditions()

