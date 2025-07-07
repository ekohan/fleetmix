"""Test post-optimization functionality."""

import pandas as pd
import pytest
import dataclasses

from fleetmix.config import load_fleetmix_params
from fleetmix.config.params import FleetmixParams
from fleetmix.core_types import Cluster, FleetmixSolution, VehicleConfiguration
from fleetmix.post_optimization import merge_phase


def make_cluster_df(cluster_id):
    goods = ["Dry", "Chilled", "Frozen"]
    # Create demand dict for Total_Demand column
    demand_dict = dict.fromkeys(goods, 1)

    return pd.DataFrame(
        [
            {
                "Cluster_ID": cluster_id,
                "Config_ID": 1,
                "Customers": [f"C{cluster_id}"],  # Add a Customers column
                "Total_Demand": demand_dict,  # Add Total_Demand with dict of goods
                "Route_Time": 10,  # Add Route_Time column
                "Centroid_Latitude": 42.0,  # Add lat/lon for centroid
                "Centroid_Longitude": -71.0,
                "Method": "test",  # Add method field
                # include all goods by default
                **dict.fromkeys(goods, 1),
            }
        ]
    )


def make_cluster_list(cluster_id):
    """Create a list of Cluster objects for testing."""
    df = make_cluster_df(cluster_id)
    return Cluster.from_dataframe(df)


# Create a minimal configurations list
def make_configs():
    return [
        VehicleConfiguration(
            config_id=1,
            vehicle_type="Test",
            capacity=1000,
            fixed_cost=100,
            compartments={"Dry": True, "Chilled": True, "Frozen": True},
        )
    ]


def make_test_customers():
    """Create minimal customer data for testing."""
    test_customers_df = pd.DataFrame({
        'Customer_ID': ['Cc1', 'Cc', 'Cm1', 'Cg1', 'Cg2', 'Cg3', 'Cc0'],  # Match cluster customer IDs
        'Latitude': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'Longitude': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'Dry_Demand': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        'Chilled_Demand': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'Frozen_Demand': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
    from fleetmix.core_types import Customer
    return Customer.from_dataframe(test_customers_df)


def test_improve_solution_basic():
    """Test that improve_solution can be called without errors."""
    # Create minimal test data
    initial_clusters = make_cluster_list("c1")
    initial_solution = FleetmixSolution(
        selected_clusters=initial_clusters, total_cost=100
    )
    configs = make_configs()
    customers = make_test_customers()
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")
    params = dataclasses.replace(params, algorithm=dataclasses.replace(params.algorithm, max_improvement_iterations=1))
    
    # This should not raise an exception
    result = merge_phase.improve_solution(initial_solution, configs, customers, params)
    
    # Basic checks
    assert isinstance(result, FleetmixSolution)
    assert result.total_cost is not None
    assert hasattr(result, 'selected_clusters')


def test_improve_solution_no_post_optimization():
    """Test that improve_solution respects post_optimization=False."""
    initial_clusters = make_cluster_list("c1")  
    initial_solution = FleetmixSolution(
        selected_clusters=initial_clusters, total_cost=100
    )
    configs = make_configs()
    customers = make_test_customers()
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")
    params = dataclasses.replace(params, algorithm=dataclasses.replace(params.algorithm, post_optimization=False))
    params = dataclasses.replace(params, algorithm=dataclasses.replace(params.algorithm, max_improvement_iterations=1))
    
    # Call improve_solution - it should still work even if post_optimization=False in params
    # because improve_solution is called directly
    result = merge_phase.improve_solution(initial_solution, configs, customers, params)
    
    assert isinstance(result, FleetmixSolution)
    assert result.total_cost is not None


def test_improve_solution_empty_clusters():
    """Test improve_solution with empty selected clusters."""
    initial_solution = FleetmixSolution(
        selected_clusters=[], total_cost=0
    )
    configs = make_configs()
    customers = make_test_customers()
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")
    params = dataclasses.replace(params, algorithm=dataclasses.replace(params.algorithm, max_improvement_iterations=1))
    
    # Should handle empty clusters gracefully
    result = merge_phase.improve_solution(initial_solution, configs, customers, params)
    
    assert isinstance(result, FleetmixSolution)
    assert len(result.selected_clusters) == 0  # Should remain empty
