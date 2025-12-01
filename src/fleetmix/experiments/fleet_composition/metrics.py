"""
Normalized metrics for alpha analysis.
"""

from collections import defaultdict
from typing import Dict

import numpy as _np

from fleetmix.core_types import FleetmixSolution

__all__ = [
    "average_visits_per_customer",
    "cost_per_drop",
    "cost_per_kg",
    "distance_ratios",
    "route_time_stats",
    "split_rate",
    "stops_stats",
]


def _count_customer_visits(solution: FleetmixSolution) -> Dict[str, int]:
    """Helper function to count visits per physical customer across all clusters."""
    customer_counts: Dict[str, int] = defaultdict(int)
    for cluster in solution.selected_clusters:
        # Count each physical customer at most once per cluster visit
        origins_in_cluster = set()
        for cust_id in cluster.customers:
            origin_id = cust_id.split("::")[0] if "::" in cust_id else cust_id
            origins_in_cluster.add(origin_id)
        for origin_id in origins_in_cluster:
            customer_counts[origin_id] += 1
    return dict(customer_counts)


def cost_per_drop(total_cost: float, num_customers: int) -> float:
    """Cost per customer drop."""
    if num_customers == 0:
        return 0.0
    return total_cost / num_customers


def cost_per_kg(total_cost: float, total_demand_kg: float) -> float:
    """Cost per kg of demand."""
    if total_demand_kg == 0:
        return 0.0
    return total_cost / total_demand_kg


def split_rate(solution: FleetmixSolution) -> float:
    """Fraction of physical customers served by multiple vehicles (split stops)."""
    customer_counts = _count_customer_visits(solution)
    if not customer_counts:
        return 0.0
    num_split = sum(1 for count in customer_counts.values() if count > 1)
    total_physical = len(customer_counts)
    return num_split / total_physical


def average_visits_per_customer(solution: FleetmixSolution) -> float:
    """Average number of vehicle visits (stops) per physical customer."""
    customer_counts = _count_customer_visits(solution)
    if not customer_counts:
        return 0.0
    return sum(customer_counts.values()) / len(customer_counts)


def route_time_stats(solution: FleetmixSolution) -> dict[str, float]:
    """Return basic statistics (avg, max, p95) of route time in **hours**.

    Assumes each ``Cluster.route_time`` is already expressed in hours.  If no
    clusters are present the function returns zeros.
    """

    rt_list = [
        float(cl.route_time)
        for cl in solution.selected_clusters
        if cl.route_time is not None
    ]
    if not rt_list:
        return {
            "avg_route_time_hr": 0.0,
            "max_route_time_hr": 0.0,
            "p95_route_time_hr": 0.0,
        }

    arr = _np.asarray(rt_list, dtype=float)
    return {
        "avg_route_time_hr": float(arr.mean()),
        "max_route_time_hr": float(arr.max()),
        "p95_route_time_hr": float(_np.percentile(arr, 95)),
    }


def stops_stats(solution: FleetmixSolution) -> dict[str, float]:
    """Return average / max number of customer stops per route."""

    stops = [len(cl.customers) for cl in solution.selected_clusters if cl.customers]
    if not stops:
        return {
            "avg_stops_per_route": 0.0,
            "max_stops_per_route": 0.0,
        }
    arr = _np.asarray(stops, dtype=float)
    return {
        "avg_stops_per_route": float(arr.mean()),
        "max_stops_per_route": float(arr.max()),
    }


def distance_ratios(
    total_distance_km: float, num_customers: int, num_routes: int
) -> dict[str, float]:
    """Compute distance per drop and per route (km)."""

    return {
        "distance_per_drop_km": float(total_distance_km / num_customers)
        if num_customers > 0
        else 0.0,
        "distance_per_route_km": float(total_distance_km / num_routes)
        if num_routes > 0
        else 0.0,
    }
