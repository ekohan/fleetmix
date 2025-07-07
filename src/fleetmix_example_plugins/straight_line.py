"""Straight-line route time estimator plugin for FleetMix.

Provides a minimal example of a custom :class:`RouteTimeEstimator` that assumes
vehicles travel in straight lines at the configuration's *avg_speed*.

This file is *only* for demonstration purposes, it shows how a user can add
custom components without modifying FleetMix's source code.

See examples/custom_route_time.py.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pandas as pd

from fleetmix.core_types import RouteTimeContext
from fleetmix.registry import register_route_time_estimator


@register_route_time_estimator("straight_line")
class StraightLineRouteTimeEstimator:
    """Route time = straight-line distance / avg_speed + service times."""

    def estimate_route_time(
        self,
        cluster_customers: pd.DataFrame,
        context: RouteTimeContext,
    ) -> Tuple[float, List[str]]:
        # Extract lat/lon as tuples – fall back to (0,0) if missing.
        coords = cluster_customers[["Latitude", "Longitude"]].fillna(0.0).to_numpy()
        depot_lat = context.depot["latitude"]
        depot_lon = context.depot["longitude"]

        def _dist(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
            return math.hypot(a_lat - b_lat, a_lon - b_lon)

        total_distance = 0.0
        last_lat, last_lon = depot_lat, depot_lon
        for lat, lon in coords:
            total_distance += _dist(last_lat, last_lon, lat, lon)
            last_lat, last_lon = lat, lon
        # Return to depot
        total_distance += _dist(last_lat, last_lon, depot_lat, depot_lon)

        # Convert distance to hours using average speed (km/h). Coordinates are
        # treated as *kilometres* for simplicity – sufficient for demonstration.
        travel_time_hr = (
            total_distance / context.avg_speed if context.avg_speed else 0.0
        )

        service_time_hr = len(coords) * context.service_time / 60.0  # min → hr

        route_time_hr = travel_time_hr + service_time_hr

        sequence = cluster_customers["Customer_ID"].astype(str).tolist()
        return route_time_hr, sequence
