import numpy as np
import pandas as pd
import pytest

from fleetmix.core_types import Customer
from fleetmix.clustering.heuristics import (
    compute_composite_distance,
    get_cached_demand,
    estimate_num_initial_clusters,
)
from fleetmix.core_types import CapacitatedClusteringContext, DepotLocation, VehicleConfiguration


def make_customers(coords, demands, goods):
    # coords: list of (lat, lon), demands: list of dicts g->d
    df = pd.DataFrame(
        {
            "Customer_ID": [f"C{i+1}" for i in range(len(coords))],
            "Latitude": [c[0] for c in coords], 
            "Longitude": [c[1] for c in coords]
        }
    )
    for i, d in enumerate(demands):
        for g in goods:
            df.at[i, f"{g}_Demand"] = d.get(g, 0)
    return df


def test_compute_composite_distance_symmetry_and_zero_diag():
    goods = ["Dry", "Chilled", "Frozen"]
    # Two customers
    coords = [(0, 0), (0, 1)]
    demands = [
        {"Dry": 1, "Chilled": 0, "Frozen": 0},
        {"Dry": 0, "Chilled": 1, "Frozen": 0},
    ]
    df = make_customers(coords, demands, goods)

    # geo_weight=0 => only demand similarity
    dist = compute_composite_distance(df, goods, geo_weight=0.0, demand_weight=1.0)
    # Should be symmetric and zero diagonal
    assert np.allclose(np.diag(dist), 0)
    assert pytest.approx(dist[0, 1]) == dist[1, 0]
    # demand distance should be >0
    assert dist[0, 1] > 0

    # geo_weight=1 => only geo distances (normalized)
    dist2 = compute_composite_distance(df, goods, geo_weight=1.0, demand_weight=0.0)
    # distance between points is 1 deg, normalized to max 1
    assert pytest.approx(dist2[0, 1]) == 1.0


def test_estimate_num_initial_clusters_by_capacity():
    goods = ["dry"]  # only dry demand - use lowercase for consistency
    # Create 5 customers each with dry demand=2
    coords = [(0, 0)] * 5
    demands = [{"dry": 2} for _ in coords]
    df = make_customers(coords, demands, goods)

    # Build dummy config and clustering context using VehicleConfiguration
    config = VehicleConfiguration(
        config_id=1,
        vehicle_type="TestVehicle",
        capacity=3,
        fixed_cost=100,
        compartments={"dry": True, "chilled": False, "frozen": False},
        avg_speed=1,
        service_time=0,
        max_route_time=100,
    )
    depot_location = DepotLocation(latitude=0, longitude=0)
    context = CapacitatedClusteringContext(
        goods=goods,
        depot=depot_location,
        max_depth=1,
        route_time_estimation="Legacy",
        geo_weight=1.0,
        demand_weight=0.0,
    )

    num = estimate_num_initial_clusters(df, config, context)
    # total demand 5*2=10, capacity 3 => clusters_by_capacity=ceil(10/3)=4
    assert num == 4
