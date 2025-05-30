"""Test save_results utility functions"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from fleetmix.utils.save_results import save_optimization_results
from fleetmix.config.parameters import Parameters


class TestSaveResults:
    """Test save results utility functions"""
    
    def test_save_optimization_results_fallback_columns(self, tmp_path):
        """Test save_optimization_results handles missing columns gracefully"""
        # Create minimal test data
        parameters = MagicMock(spec=Parameters)
        parameters.results_dir = tmp_path
        parameters.demand_file = "test.csv"
        parameters.variable_cost_per_hour = 50
        parameters.avg_speed = 30
        parameters.max_route_time = 10
        parameters.service_time = 15
        parameters.clustering = {
            'max_depth': 3,
            'method': 'minibatch_kmeans',
            'distance': 'euclidean',
            'geo_weight': 0.7,
            'demand_weight': 0.3,
            'route_time_estimation': 'BHH'
        }
        parameters.light_load_penalty = 1000
        parameters.light_load_threshold = 0.2
        parameters.compartment_setup_cost = 50
        parameters.vehicles = {'T1': {'capacity': 1000, 'fixed_cost': 100}}
        parameters.goods = ['Dry', 'Chilled', 'Frozen']
        parameters.depot = {'latitude': 4.65, 'longitude': -74.15}
        
        configurations_df = pd.DataFrame({
            'Config_ID': ['C1'],
            'Capacity': [1000]
        })
        
        # Create selected_clusters without 'Customers' column (line 71)
        # and without 'Vehicle_Utilization' (line 77)
        selected_clusters = pd.DataFrame({
            'Cluster_ID': ['CL1'],
            'Config_ID': ['C1'],
            'Num_Customers': [5],  # Use Num_Customers instead of Customers
            'Total_Demand': [{'Dry': 500, 'Chilled': 200, 'Frozen': 100}],
            'Route_Time': [8.5],
            'Centroid_Latitude': [4.6],
            'Centroid_Longitude': [-74.1],
            'Method': ['BHH'],
            'Estimated_Distance': [120.5]
        })
        
        vehicles_used = pd.Series({'T1': 1})
        
        # Call the function - should handle missing columns gracefully
        save_optimization_results(
            execution_time=10.5,
            solver_name='CBC',
            solver_status='Optimal',
            configurations_df=configurations_df,
            selected_clusters=selected_clusters,
            total_fixed_cost=100,
            total_variable_cost=425,
            total_light_load_penalties=0,
            total_compartment_penalties=0,
            total_penalties=0,
            vehicles_used=vehicles_used,
            missing_customers=set(),
            parameters=parameters,
            filename=tmp_path / "test_results.xlsx",
            format='excel',
            is_benchmark=False
        )
        
        # Check file was created
        assert (tmp_path / "test_results.xlsx").exists()
        
        # Check visualization was created
        assert (tmp_path / "test_results_clusters.html").exists()

    def test_save_optimization_results_total_demand_string(self, tmp_path):
        """Test save_optimization_results when Total_Demand is a string that needs parsing"""
        # Create minimal test data
        parameters = MagicMock(spec=Parameters)
        parameters.results_dir = tmp_path
        parameters.demand_file = "test.csv"
        parameters.variable_cost_per_hour = 50
        parameters.avg_speed = 30
        parameters.max_route_time = 10
        parameters.service_time = 15
        parameters.clustering = {
            'max_depth': 3,
            'method': 'minibatch_kmeans',
            'distance': 'euclidean',
            'geo_weight': 0.7,
            'demand_weight': 0.3,
            'route_time_estimation': 'BHH'
        }
        parameters.light_load_penalty = 1000
        parameters.light_load_threshold = 0.2
        parameters.compartment_setup_cost = 50
        parameters.vehicles = {'T1': {'capacity': 1000, 'fixed_cost': 100}}
        parameters.goods = ['Dry', 'Chilled', 'Frozen']
        parameters.depot = {'latitude': 4.65, 'longitude': -74.15}
        
        configurations_df = pd.DataFrame({
            'Config_ID': ['C1'],
            'Capacity': [1000]
        })
        
        # Create selected_clusters with Total_Demand as string (to trigger line 77)
        selected_clusters = pd.DataFrame({
            'Cluster_ID': ['CL1'],
            'Config_ID': ['C1'],
            'Num_Customers': [5],
            'Total_Demand': ["{'Dry': 500, 'Chilled': 200, 'Frozen': 100}"],  # String format
            'Route_Time': [8.5],
            'Centroid_Latitude': [4.6],
            'Centroid_Longitude': [-74.1],
            'Method': ['BHH'],
            'Estimated_Distance': [120.5]
        })
        
        vehicles_used = pd.Series({'T1': 1})
        
        # Call the function - should handle string Total_Demand
        save_optimization_results(
            execution_time=10.5,
            solver_name='CBC',
            solver_status='Optimal',
            configurations_df=configurations_df,
            selected_clusters=selected_clusters,
            total_fixed_cost=100,
            total_variable_cost=425,
            total_light_load_penalties=0,
            total_compartment_penalties=0,
            total_penalties=0,
            vehicles_used=vehicles_used,
            missing_customers=set(),
            parameters=parameters,
            filename=tmp_path / "test_string_demand.xlsx",
            format='excel',
            is_benchmark=False
        )
        
        # Check file was created
        assert (tmp_path / "test_string_demand.xlsx").exists()
    
    def test_save_optimization_results_json_format(self, tmp_path):
        """Test save_optimization_results with JSON format"""
        # Create minimal test data
        parameters = MagicMock(spec=Parameters)
        parameters.results_dir = tmp_path
        parameters.demand_file = "test.csv"
        parameters.variable_cost_per_hour = 50
        parameters.avg_speed = 30
        parameters.max_route_time = 10
        parameters.service_time = 15
        parameters.clustering = {
            'max_depth': 3,
            'method': 'minibatch_kmeans',
            'distance': 'euclidean',
            'geo_weight': 0.7,
            'demand_weight': 0.3,
            'route_time_estimation': 'BHH'
        }
        parameters.light_load_penalty = 1000
        parameters.light_load_threshold = 0.2
        parameters.compartment_setup_cost = 50
        parameters.vehicles = {'T1': {'capacity': 1000, 'fixed_cost': 100}}
        parameters.goods = ['Dry', 'Chilled', 'Frozen']
        parameters.depot = {'latitude': 4.65, 'longitude': -74.15}
        
        configurations_df = pd.DataFrame({
            'Config_ID': ['C1'],
            'Capacity': [1000]
        })
        
        # Create selected_clusters with different data types
        selected_clusters = pd.DataFrame({
            'Cluster_ID': ['CL1'],
            'Config_ID': ['C1'],
            'Num_Customers': [5],
            'Total_Demand': [{'Dry': 500, 'Chilled': 200, 'Frozen': 100}],
            'Route_Time': [8.5],
            'Centroid_Latitude': [4.6],
            'Centroid_Longitude': [-74.1],
            'Method': ['BHH'],
            'Estimated_Distance': [120.5],
            'TSP_Sequence': [['Depot', 'C1', 'C2', 'C3', 'C4', 'C5', 'Depot']]
        })
        
        vehicles_used = pd.Series({'T1': 1})
        
        # Call with JSON format
        save_optimization_results(
            execution_time=10.5,
            solver_name='CBC',
            solver_status='Optimal',
            configurations_df=configurations_df,
            selected_clusters=selected_clusters,
            total_fixed_cost=100,
            total_variable_cost=425,
            total_light_load_penalties=0,
            total_compartment_penalties=0,
            total_penalties=0,
            vehicles_used=vehicles_used,
            missing_customers=set(),
            parameters=parameters,
            filename=tmp_path / "test_json_output.json",
            format='json',
            is_benchmark=True,
            expected_vehicles=10
        )
        
        # Check file was created
        assert (tmp_path / "test_json_output.json").exists() 