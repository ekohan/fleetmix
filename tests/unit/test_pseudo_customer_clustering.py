import pandas as pd
import pytest

from fleetmix.clustering.heuristics import estimate_num_initial_clusters
from fleetmix.core_types import ClusteringContext, DepotLocation, VehicleConfiguration


def test_estimate_clusters_with_pseudo_customers():
    """Test that estimate_num_initial_clusters correctly handles pseudo-customers by grouping demand by origin ID."""
    
    # Create a DataFrame that simulates exploded pseudo-customers from 1 original customer
    # Original customer has {Dry: 100, Chilled: 50, Frozen: 30}
    # After explosion, we get 7 pseudo-customers representing all non-empty subsets
    pseudo_customers_data = [
        {"Customer_ID": "C1::dry", "Origin_ID": "C1", "Dry_Demand": 100, "Chilled_Demand": 0, "Frozen_Demand": 0, "Latitude": 0, "Longitude": 0},
        {"Customer_ID": "C1::chilled", "Origin_ID": "C1", "Dry_Demand": 0, "Chilled_Demand": 50, "Frozen_Demand": 0, "Latitude": 0, "Longitude": 0},
        {"Customer_ID": "C1::frozen", "Origin_ID": "C1", "Dry_Demand": 0, "Chilled_Demand": 0, "Frozen_Demand": 30, "Latitude": 0, "Longitude": 0},
        {"Customer_ID": "C1::dry-chilled", "Origin_ID": "C1", "Dry_Demand": 100, "Chilled_Demand": 50, "Frozen_Demand": 0, "Latitude": 0, "Longitude": 0},
        {"Customer_ID": "C1::dry-frozen", "Origin_ID": "C1", "Dry_Demand": 100, "Chilled_Demand": 0, "Frozen_Demand": 30, "Latitude": 0, "Longitude": 0},
        {"Customer_ID": "C1::chilled-frozen", "Origin_ID": "C1", "Dry_Demand": 0, "Chilled_Demand": 50, "Frozen_Demand": 30, "Latitude": 0, "Longitude": 0},
        {"Customer_ID": "C1::dry-chilled-frozen", "Origin_ID": "C1", "Dry_Demand": 100, "Chilled_Demand": 50, "Frozen_Demand": 30, "Latitude": 0, "Longitude": 0},
    ]
    
    df = pd.DataFrame(pseudo_customers_data)
    
    # Without the fix, total demand would be inflated to:
    # Dry: 100*4 = 400, Chilled: 50*4 = 200, Frozen: 30*4 = 120
    # Total = 720, which would massively overestimate the number of clusters needed
    
    # With the fix, we should get the original demand:
    # Dry: 100, Chilled: 50, Frozen: 30
    # Total = 180
    
    goods = ["Dry", "Chilled", "Frozen"]
    
    # Create vehicle config that can carry all goods with capacity 200
    config = VehicleConfiguration(
        config_id=1,
        vehicle_type="TestVehicle",
        capacity=200,  # Should be sufficient for the original demand (180)
        fixed_cost=100,
        compartments={"Dry": True, "Chilled": True, "Frozen": True},
        avg_speed=30,
        service_time=25,
        max_route_time=10,
    )
    
    depot_location = DepotLocation(latitude=0, longitude=0)
    context = ClusteringContext(
        goods=goods,
        depot=depot_location,
        max_depth=1,
        route_time_estimation="Legacy",
        geo_weight=1.0,
        demand_weight=0.0,
    )
    
    num_clusters = estimate_num_initial_clusters(df, config, context)
    
    # With the fix, demand = 180, capacity = 200, so we should need 1 cluster (ceil(180/200) = 1)
    # Without the fix, demand = 720, capacity = 200, so we would need 4 clusters (ceil(720/200) = 4)
    assert num_clusters == 1, f"Expected 1 cluster but got {num_clusters}. The fix may not be working correctly."


def test_estimate_clusters_with_regular_customers():
    """Test that the function still works correctly with regular (non-pseudo) customers."""
    
    # Create regular customers without pseudo-customer IDs
    regular_customers_data = [
        {"Customer_ID": "C1", "Dry_Demand": 100, "Chilled_Demand": 50, "Frozen_Demand": 30, "Latitude": 0, "Longitude": 0},
        {"Customer_ID": "C2", "Dry_Demand": 80, "Chilled_Demand": 40, "Frozen_Demand": 20, "Latitude": 1, "Longitude": 1},
    ]
    
    df = pd.DataFrame(regular_customers_data)
    
    goods = ["Dry", "Chilled", "Frozen"]
    
    # Create vehicle config with capacity 200
    config = VehicleConfiguration(
        config_id=1,
        vehicle_type="TestVehicle",
        capacity=200,  # Total demand = 180 + 140 = 320
        fixed_cost=100,
        compartments={"Dry": True, "Chilled": True, "Frozen": True},
        avg_speed=30,
        service_time=25,
        max_route_time=10,
    )
    
    depot_location = DepotLocation(latitude=0, longitude=0)
    context = ClusteringContext(
        goods=goods,
        depot=depot_location,
        max_depth=1,
        route_time_estimation="Legacy",
        geo_weight=1.0,
        demand_weight=0.0,
    )
    
    num_clusters = estimate_num_initial_clusters(df, config, context)
    
    # Total demand = 320, capacity = 200, so we should need 2 clusters (ceil(320/200) = 2)
    assert num_clusters == 2, f"Expected 2 clusters but got {num_clusters}" 