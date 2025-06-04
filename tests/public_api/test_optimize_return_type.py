"""Test that optimize returns the correct public API types."""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from fleetmix import optimize, FleetmixSolution, ClusterAssignment, VehicleConfiguration
from fleetmix.internal_types import FleetmixSolution as InternalSolution


def test_optimize_returns_public_solution_type():
    """Test that optimize returns the public FleetmixSolution type."""
    # Create minimal test data
    customers_df = pd.DataFrame({
        'Customer_ID': ['C1', 'C2'],
        'Customer_Name': ['Customer 1', 'Customer 2'],
        'Latitude': [40.7128, 40.7580],
        'Longitude': [-74.0060, -73.9855],
        'Dry_Demand': [10, 20],
        'Chilled_Demand': [5, 0],
        'Frozen_Demand': [0, 15]
    })
    
    # Mock the internal optimization to return a known result
    mock_internal_solution = InternalSolution()
    mock_internal_solution.selected_clusters = pd.DataFrame({
        'Cluster_ID': [1, 2],
        'Config_ID': [1, 2],
        'Customers': [['C1'], ['C2']],
        'Route_Time': [1.5, 2.0],
        'Total_Demand': [{'Dry': 10, 'Chilled': 5, 'Frozen': 0}, 
                        {'Dry': 20, 'Chilled': 0, 'Frozen': 15}],
        'Centroid_Latitude': [40.7128, 40.7580],
        'Centroid_Longitude': [-74.0060, -73.9855],
        'Vehicle_Type': ['Small', 'Large'],
        'Capacity': [50, 100],
        'Fixed_Cost': [100.0, 200.0],
        'Dry': [1, 1],
        'Chilled': [1, 0],
        'Frozen': [0, 1]
    })
    mock_internal_solution.total_cost = 500.0
    mock_internal_solution.total_vehicles = 2
    mock_internal_solution.missing_customers = set()
    mock_internal_solution.solver_status = 'Optimal'
    mock_internal_solution.solver_runtime_sec = 0.1
    
    # Create mock clusters DataFrame
    mock_clusters_df = pd.DataFrame({
        'Cluster_ID': [1, 2],
        'Config_ID': [1, 2],
        'Customers': [['C1'], ['C2']],
        'Route_Time': [1.5, 2.0],
        'Total_Demand': [{'Dry': 10, 'Chilled': 5, 'Frozen': 0}, 
                        {'Dry': 20, 'Chilled': 0, 'Frozen': 15}],
        'Centroid_Latitude': [40.7128, 40.7580],
        'Centroid_Longitude': [-74.0060, -73.9855],
    })
    
    with patch('fleetmix.api.load_customer_demand') as mock_load:
        with patch('fleetmix.utils.vehicle_configurations._generate_vehicle_configurations_df') as mock_gen_configs:
            with patch('fleetmix.clustering.generator._generate_feasible_clusters_df') as mock_gen_clusters:
                with patch('fleetmix.api.optimize_fleet_selection') as mock_optimize:
                    with patch('fleetmix.api.Parameters') as mock_params_class:
                        # Setup mocks
                        mock_load.return_value = customers_df
                        mock_gen_configs.return_value = pd.DataFrame({
                            'Config_ID': [1, 2],
                            'Vehicle_Type': ['Small', 'Large'],
                            'Capacity': [50, 100],
                            'Fixed_Cost': [100.0, 200.0],
                            'Dry': [1, 1],
                            'Chilled': [1, 0],
                            'Frozen': [0, 1]
                        })
                        mock_gen_clusters.return_value = mock_clusters_df  # Return non-empty DataFrame
                        mock_optimize.return_value = mock_internal_solution

                        # Mock parameters
                        mock_params = MagicMock()
                        mock_params.goods = ['Dry', 'Chilled', 'Frozen']
                        mock_params.vehicles = {'Small': MagicMock(), 'Large': MagicMock()}
                        mock_params_class.from_yaml.return_value = mock_params

                        # Call optimize
                        solution = optimize(customers_df, output_dir=None)
    
    # Verify return type
    assert isinstance(solution, FleetmixSolution)
    assert not isinstance(solution, InternalSolution)
    
    # Verify structure
    assert isinstance(solution.selected_clusters, list)
    assert len(solution.selected_clusters) == 2
    assert all(isinstance(c, ClusterAssignment) for c in solution.selected_clusters)
    
    assert isinstance(solution.configurations_used, list)
    assert len(solution.configurations_used) == 2
    assert all(isinstance(c, VehicleConfiguration) for c in solution.configurations_used)
    
    # Verify data
    assert solution.total_cost == 500.0
    assert solution.total_vehicles == 2
    assert solution.missing_customers == set()
    assert solution.solver_status == 'Optimal'
    assert solution.solver_runtime_sec == 0.1
    
    # Verify cluster details
    cluster1 = solution.selected_clusters[0]
    assert cluster1.cluster_id == 1
    assert cluster1.config_id == 1
    assert cluster1.customer_ids == ['C1']
    assert cluster1.route_time == 1.5
    
    # Verify configuration details
    config1 = solution.configurations_used[0]
    assert config1.config_id == 1
    assert config1.vehicle_type == 'Small'
    assert config1.capacity == 50
    assert config1.fixed_cost == 100.0
    assert config1.compartments == {'Dry': True, 'Chilled': True, 'Frozen': False} 