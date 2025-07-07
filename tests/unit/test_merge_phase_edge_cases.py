"""Test edge cases in the merge phase module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fleetmix.config import load_fleetmix_params
from fleetmix.core_types import FleetmixSolution, VehicleConfiguration, Cluster
from fleetmix.merging.core import (
    _get_merged_route_time,
    generate_merge_phase_clusters,
    validate_merged_cluster,
)
from fleetmix.post_optimization.merge_phase import improve_solution


def dataframe_to_configurations(df: pd.DataFrame) -> list[VehicleConfiguration]:
    """Convert DataFrame to List[VehicleConfiguration] for testing."""
    configs = []
    for _, row in df.iterrows():
        # Determine compartments based on goods columns
        compartments = {}
        goods_cols = ["Dry", "Chilled", "Frozen"]
        for good in goods_cols:
            if good in row:
                compartments[good] = bool(row[good])

        config = VehicleConfiguration(
            config_id=row["Config_ID"],
            vehicle_type=row.get("Vehicle_Type", "Test"),
            capacity=row["Capacity"],
            fixed_cost=row["Fixed_Cost"],
            compartments=compartments,
        )
        configs.append(config)
    return configs


@pytest.fixture
def simple_params():
    """Create simple parameters for testing."""
    config_path = (
        Path(__file__).parent.parent / "_assets" / "configs" / "base_test_config.yaml"
    )
    return load_fleetmix_params(config_path)

@pytest.fixture
def params_with_post_opt():
    """Create parameters with post-optimization settings."""
    config_path = (
        Path(__file__).parent.parent
        / "_assets"
        / "configs"
        / "test_config_post_opt.yaml"
    )
    return load_fleetmix_params(config_path)


def test_improve_solution_no_clusters(simple_params):
    """Test improve_solution when solution has no clusters."""
    # Solution without clusters key
    initial_solution_obj = FleetmixSolution(
        total_cost=100,
        total_fixed_cost=50,
        total_variable_cost=50,
    )

    configs = []
    customers_df = pd.DataFrame()

    # Should return the initial solution unchanged
    # Convert DataFrame to list for new API
    from fleetmix.core_types import Customer

    customers_list = Customer.from_dataframe(customers_df)
    result = improve_solution(
        initial_solution_obj, configs, customers_list, simple_params
    )
    assert result.total_cost == initial_solution_obj.total_cost
    assert len(result.selected_clusters) == 0


def test_improve_solution_missing_goods_columns(params_with_post_opt):
    """Test improve_solution when selected clusters miss goods columns."""
    # Create selected clusters without goods columns
    selected_clusters = [
        Cluster(
            cluster_id="C1",
            config_id="Small",
            customers=["Cust1", "Cust2"],
            total_demand={"Dry": 20, "Chilled": 0, "Frozen": 0},
            centroid_latitude=0.1,
            centroid_longitude=0.1,
            goods_in_config=["Dry"],
            route_time=2.0,
            method="test",
        )
    ]

    initial_solution_obj = FleetmixSolution(
        total_cost=100, selected_clusters=selected_clusters
    )

    configs_df = pd.DataFrame(
        {
            "Config_ID": ["Small"],
            "Vehicle_Type": ["Small"],
            "Capacity": [50],
            "Fixed_Cost": [100],
            "Dry": [1],
            "Chilled": [0],
            "Frozen": [0],
        }
    )
    configurations = dataframe_to_configurations(configs_df)

    customers_df = pd.DataFrame(
        {
            "Customer_ID": ["Cust1", "Cust2"],
            "Latitude": [0.05, 0.15],
            "Longitude": [0.05, 0.15],
        }
    )

    # This should handle missing goods columns
    # Convert DataFrame to list for new API
    from fleetmix.core_types import Customer

    customers_list = Customer.from_dataframe(customers_df)
    result = improve_solution(
        initial_solution_obj, configurations, customers_list, params_with_post_opt
    )
    assert len(result.selected_clusters) > 0


def test_generate_merge_phase_clusters_with_small_clusters(simple_params):
    """Test generating merge phase clusters with small cluster size."""
    # Create small clusters that are valid candidates for merging
    selected_clusters = pd.DataFrame(
        {
            "Cluster_ID": ["C1", "C2"],
            "Customers": [["Cust1"], ["Cust2"]],  # Small clusters
            "Config_ID": ["V1", "V1"],
            "Total_Demand": [
                {"Dry": 10, "Chilled": 0, "Frozen": 0},
                {"Dry": 10, "Chilled": 0, "Frozen": 0},
            ],
            "Route_Time": [2.0, 2.0],  # Reasonable route times
            "Centroid_Latitude": [0.1, 0.2],
            "Centroid_Longitude": [0.1, 0.2],
            "Method": ["test", "test"],
            "Dry": [1, 1],
            "Chilled": [0, 0],
            "Frozen": [0, 0],
        }
    )

    configs_df = pd.DataFrame(
        {
            "Config_ID": ["V1"],
            "Capacity": [50],
            "Fixed_Cost": [100],
            "Dry": [1],
            "Chilled": [0],
            "Frozen": [0],
        }
    )
    configurations = dataframe_to_configurations(configs_df)

    customers_df = pd.DataFrame(
        {
            "Customer_ID": ["Cust1", "Cust2"],
            "Latitude": [0.1, 0.2],
            "Longitude": [0.1, 0.2],
        }
    )

    # Should generate merge candidates when small clusters exist
    result = generate_merge_phase_clusters(
        selected_clusters, configurations, customers_df, simple_params
    )
    assert not result.empty  # Should have some merge candidates


def test_validate_merged_cluster_missing_customers(simple_params):
    """Test validating merged cluster with missing customer IDs."""
    cluster1 = pd.Series(
        {
            "Cluster_ID": "C1",
            "Customers": ["Cust1", "Missing1"],  # Missing1 doesn't exist
            "Total_Demand": {"Dry": 10, "Chilled": 0, "Frozen": 0},
            "Route_Time": 1.0,
        }
    )

    cluster2 = pd.Series(
        {
            "Cluster_ID": "C2",
            "Customers": ["Cust2"],
            "Total_Demand": {"Dry": 10, "Chilled": 0, "Frozen": 0},
            "Route_Time": 1.0,
        }
    )

    config = VehicleConfiguration(
        config_id="V1",
        vehicle_type="Test",
        capacity=50,
        fixed_cost=100,
        compartments={"Dry": True, "Chilled": False, "Frozen": False},
    )

    # Only has Cust1 and Cust2, missing Missing1
    customers_df = pd.DataFrame(
        {
            "Customer_ID": ["Cust1", "Cust2"],
            "Latitude": [0.1, 0.2],
            "Longitude": [0.1, 0.2],
        }
    ).set_index("Customer_ID")

    is_valid, route_time, demands, sequence = validate_merged_cluster(
        cluster1, cluster2, config, customers_df, simple_params
    )

    assert not is_valid


def test_validate_merged_cluster_invalid_locations(simple_params):
    """Test validating merged cluster with invalid customer locations."""
    cluster1 = pd.Series(
        {
            "Cluster_ID": "C1",
            "Customers": ["Cust1"],
            "Total_Demand": {"Dry": 10, "Chilled": 0, "Frozen": 0},
            "Route_Time": 1.0,
        }
    )

    cluster2 = pd.Series(
        {
            "Cluster_ID": "C2",
            "Customers": ["Cust2"],
            "Total_Demand": {"Dry": 10, "Chilled": 0, "Frozen": 0},
            "Route_Time": 1.0,
        }
    )

    config = VehicleConfiguration(
        config_id="V1",
        vehicle_type="Test",
        capacity=50,
        fixed_cost=100,
        compartments={"Dry": True, "Chilled": False, "Frozen": False},
    )

    # Cust2 has NaN location
    customers_df = pd.DataFrame(
        {
            "Customer_ID": ["Cust1", "Cust2"],
            "Latitude": [0.1, np.nan],
            "Longitude": [0.1, np.nan],
        }
    ).set_index("Customer_ID")

    is_valid, route_time, demands, sequence = validate_merged_cluster(
        cluster1, cluster2, config, customers_df, simple_params
    )

    assert not is_valid


def test_validate_merged_cluster_capacity_exceeded(simple_params):
    """Test validating merged cluster that exceeds capacity."""
    cluster1 = pd.Series(
        {
            "Cluster_ID": "C1",
            "Customers": ["Cust1"],
            "Total_Demand": {"Dry": 30, "Chilled": 0, "Frozen": 0},
            "Route_Time": 1.0,
        }
    )

    cluster2 = pd.Series(
        {
            "Cluster_ID": "C2",
            "Customers": ["Cust2"],
            "Total_Demand": {"Dry": 30, "Chilled": 0, "Frozen": 0},
            "Route_Time": 1.0,
        }
    )

    config = VehicleConfiguration(
        config_id="V1",
        vehicle_type="Test",
        capacity=50,  # Total demand would be 60, exceeding capacity
        fixed_cost=100,
        compartments={"Dry": True, "Chilled": False, "Frozen": False},
    )

    customers_df = pd.DataFrame(
        {
            "Customer_ID": ["Cust1", "Cust2"],
            "Latitude": [0.1, 0.2],
            "Longitude": [0.1, 0.2],
        }
    ).set_index("Customer_ID")

    is_valid, route_time, demands, sequence = validate_merged_cluster(
        cluster1, cluster2, config, customers_df, simple_params
    )

    assert not is_valid


def test_get_merged_route_time_caching(simple_params):
    """Test that merged route time caching works correctly."""
    from fleetmix.merging.core import _merged_route_time_cache

    # Clear cache
    _merged_route_time_cache.clear()

    customers = pd.DataFrame(
        {"Customer_ID": ["C1", "C2"], "Latitude": [0.1, 0.2], "Longitude": [0.1, 0.2]}
    )

    # Create a vehicle configuration for the function
    from fleetmix.core_types import VehicleConfiguration

    config = VehicleConfiguration(
        config_id=1,
        vehicle_type="Test",
        capacity=100,
        fixed_cost=50,
        compartments={"Dry": True},
        avg_speed=30.0,
        service_time=25.0,
        max_route_time=10.0,
    )

    # First call should compute
    time1, seq1 = _get_merged_route_time(customers, config, simple_params)
    assert len(_merged_route_time_cache) == 1

    # Second call should use cache
    time2, seq2 = _get_merged_route_time(customers, config, simple_params)
    assert time1 == time2
    assert seq1 == seq2
    assert len(_merged_route_time_cache) == 1  # Still only one entry
