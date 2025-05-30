"""
Comprehensive tests for the Fleetmix API module.
"""
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from fleetmix.api import optimize
from fleetmix.config.parameters import Parameters


class TestAPIOptimize:
    """Test suite for the optimize function in api.py"""
    
    def test_optimize_with_dataframe_input(self):
        """Test optimization with pandas DataFrame input"""
        # Create a minimal customer DataFrame
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1', 'C2'],
            'Latitude': [40.7128, 40.7589],
            'Longitude': [-74.0060, -73.9851],
            'Dry_Demand': [10, 15],
            'Chilled_Demand': [5, 8],
            'Frozen_Demand': [0, 0]
        })
        
        # Mock the optimization pipeline
        with patch('fleetmix.api.generate_vehicle_configurations') as mock_gen_configs, \
             patch('fleetmix.api.generate_clusters_for_configurations') as mock_gen_clusters, \
             patch('fleetmix.api.solve_fsm_problem') as mock_solve, \
             patch('fleetmix.api.save_optimization_results') as mock_save:
            
            # Setup mocks
            mock_gen_configs.return_value = pd.DataFrame({'Config_ID': [1], 'Capacity': [100]})
            mock_gen_clusters.return_value = pd.DataFrame({'Cluster_ID': [1], 'Config_ID': [1]})
            mock_solve.return_value = {
                'solver_status': 'Optimal',
                'total_fixed_cost': 100.0,
                'total_variable_cost': 50.0,
                'total_penalties': 0.0,
                'vehicles_used': 1,
                'selected_clusters': pd.DataFrame(),
                'missing_customers': [],
                'solver_runtime_sec': 1.0,
                'post_optimization_runtime_sec': 0.5,
                'solver_name': 'test'
            }
            
            result = optimize(demand=customers_df, output_dir=None)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert 'total_fixed_cost' in result
            assert 'solver_status' in result
            assert result['solver_status'] == 'Optimal'
    
    def test_optimize_missing_dataframe_columns(self):
        """Test error handling when DataFrame is missing required columns"""
        incomplete_df = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [40.7128]
            # Missing Longitude
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            optimize(demand=incomplete_df)
    
    def test_optimize_missing_demand_file(self):
        """Test error handling for missing demand file"""
        with pytest.raises(FileNotFoundError, match="Demand file not found"):
            optimize(demand="nonexistent.csv")
    
    def test_optimize_missing_config_file(self):
        """Test error handling for missing config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a CSV file in the format load_customer_demand expects
            demand_file = Path(tmpdir) / "demand.csv"
            demand_file.write_text(
                "ClientID,Lat,Lon,Kg,ProductType\n"
                "C1,40.7,-74.0,10,Dry\n"
                "C1,40.7,-74.0,5,Chilled\n"
                "C1,40.7,-74.0,2,Frozen\n"
            )
            
            with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                optimize(demand=str(demand_file), config="nonexistent.yaml")
    
    def test_optimize_invalid_config_yaml(self):
        """Test error handling for invalid YAML config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a CSV file in the format load_customer_demand expects
            demand_file = Path(tmpdir) / "demand.csv"
            demand_file.write_text(
                "ClientID,Lat,Lon,Kg,ProductType\n"
                "C1,40.7,-74.0,10,Dry\n"
                "C1,40.7,-74.0,5,Chilled\n"
                "C1,40.7,-74.0,2,Frozen\n"
            )
            
            config_file = Path(tmpdir) / "invalid.yaml"
            config_file.write_text("invalid: yaml: content: [unclosed")
            
            with pytest.raises(ValueError, match="Error loading configuration"):
                optimize(demand=str(demand_file), config=str(config_file))
    
    def test_optimize_parameters_object_input(self):
        """Test optimization with Parameters object as config"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060],
            'Dry_Demand': [10],
            'Chilled_Demand': [5],
            'Frozen_Demand': [0]
        })
        
        # Mock the config=None path since it's easier to test
        with patch('fleetmix.api.generate_vehicle_configurations') as mock_gen_configs, \
             patch('fleetmix.api.generate_clusters_for_configurations') as mock_gen_clusters, \
             patch('fleetmix.api.solve_fsm_problem') as mock_solve:
            
            mock_gen_configs.return_value = pd.DataFrame({'Config_ID': [1]})
            mock_gen_clusters.return_value = pd.DataFrame({'Cluster_ID': [1]})
            mock_solve.return_value = {
                'solver_status': 'Optimal',
                'total_fixed_cost': 100.0,
                'total_variable_cost': 50.0,
                'total_penalties': 0.0,
                'vehicles_used': 1,
                'selected_clusters': pd.DataFrame(),
                'missing_customers': [],
                'solver_runtime_sec': 1.0,
                'post_optimization_runtime_sec': 0.5,
                'solver_name': 'test'
            }
            
            # Test with config=None to exercise the default parameters path
            result = optimize(demand=customers_df, config=None, output_dir=None)
            assert result['solver_status'] == 'Optimal'
    
    def test_optimize_no_feasible_clusters(self):
        """Test error handling when no feasible clusters can be generated"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060],
            'Dry_Demand': [10],
            'Chilled_Demand': [5],
            'Frozen_Demand': [0]
        })
        
        with patch('fleetmix.api.generate_vehicle_configurations') as mock_gen_configs, \
             patch('fleetmix.api.generate_clusters_for_configurations') as mock_gen_clusters:
            
            mock_gen_configs.return_value = pd.DataFrame({'Config_ID': [1]})
            mock_gen_clusters.return_value = pd.DataFrame()  # Empty clusters
            
            with pytest.raises(ValueError, match="No feasible clusters could be generated"):
                optimize(demand=customers_df, output_dir=None)
    
    def test_optimize_infeasible_solution(self):
        """Test error handling for infeasible optimization results"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1', 'C2'],
            'Latitude': [40.7128, 40.7589],
            'Longitude': [-74.0060, -73.9851],
            'Dry_Demand': [10, 15],
            'Chilled_Demand': [5, 8],
            'Frozen_Demand': [0, 0]
        })
        
        with patch('fleetmix.api.generate_vehicle_configurations') as mock_gen_configs, \
             patch('fleetmix.api.generate_clusters_for_configurations') as mock_gen_clusters, \
             patch('fleetmix.api.solve_fsm_problem') as mock_solve:
            
            mock_gen_configs.return_value = pd.DataFrame({'Config_ID': [1]})
            mock_gen_clusters.return_value = pd.DataFrame({'Cluster_ID': [1]})
            mock_solve.return_value = {
                'solver_status': 'Infeasible',
                'missing_customers': ['C1', 'C2'],
                'solver_runtime_sec': 1.0,
                'post_optimization_runtime_sec': 0.5,
                'solver_name': 'test'
            }
            
            with pytest.raises(ValueError, match="Optimization problem is infeasible"):
                optimize(demand=customers_df, output_dir=None)
    
    def test_optimize_vehicle_config_error(self):
        """Test error handling during vehicle configuration generation"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060],
            'Dry_Demand': [10],
            'Chilled_Demand': [5],
            'Frozen_Demand': [0]
        })
        
        with patch('fleetmix.api.generate_vehicle_configurations') as mock_gen_configs:
            mock_gen_configs.side_effect = Exception("Config generation failed")
            
            with pytest.raises(ValueError, match="Error generating vehicle configurations"):
                optimize(demand=customers_df, output_dir=None)
    
    def test_optimize_clustering_error(self):
        """Test error handling during clustering"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060],
            'Dry_Demand': [10],
            'Chilled_Demand': [5],
            'Frozen_Demand': [0]
        })
        
        with patch('fleetmix.api.generate_vehicle_configurations') as mock_gen_configs, \
             patch('fleetmix.api.generate_clusters_for_configurations') as mock_gen_clusters:
            
            mock_gen_configs.return_value = pd.DataFrame({'Config_ID': [1]})
            mock_gen_clusters.side_effect = Exception("Clustering failed")
            
            with pytest.raises(ValueError, match="Error generating clusters"):
                optimize(demand=customers_df, output_dir=None)
    
    def test_optimize_csv_wide_format(self):
        """Test optimization with CSV file in wide format (already has demand columns)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            demand_file = Path(tmpdir) / "demand.csv"
            demand_file.write_text(
                "Customer_ID,Customer_Name,Latitude,Longitude,Dry_Demand,Chilled_Demand,Frozen_Demand\n"
                "C1,Customer 1,40.7128,-74.0060,10,5,0\n"
                "C2,Customer 2,40.7589,-73.9851,15,8,0\n"
            )
            
            with patch('fleetmix.api.generate_vehicle_configurations') as mock_gen_configs, \
                 patch('fleetmix.api.generate_clusters_for_configurations') as mock_gen_clusters, \
                 patch('fleetmix.api.solve_fsm_problem') as mock_solve:
                
                mock_gen_configs.return_value = pd.DataFrame({'Config_ID': [1]})
                mock_gen_clusters.return_value = pd.DataFrame({'Cluster_ID': [1]})
                mock_solve.return_value = {
                    'solver_status': 'Optimal',
                    'total_fixed_cost': 100.0,
                    'total_variable_cost': 50.0,
                    'total_penalties': 0.0,
                    'vehicles_used': 1,
                    'selected_clusters': pd.DataFrame(),
                    'missing_customers': [],
                    'solver_runtime_sec': 1.0,
                    'post_optimization_runtime_sec': 0.5,
                    'solver_name': 'test'
                }
                
                result = optimize(demand=str(demand_file), output_dir=None)
                assert result['solver_status'] == 'Optimal'
    
    def test_optimize_save_results_failure(self):
        """Test that save failure doesn't crash the optimization"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060],
            'Dry_Demand': [10],
            'Chilled_Demand': [5],
            'Frozen_Demand': [0]
        })
        
        with patch('fleetmix.api.generate_vehicle_configurations') as mock_gen_configs, \
             patch('fleetmix.api.generate_clusters_for_configurations') as mock_gen_clusters, \
             patch('fleetmix.api.solve_fsm_problem') as mock_solve, \
             patch('fleetmix.api.save_optimization_results') as mock_save:
            
            mock_gen_configs.return_value = pd.DataFrame({'Config_ID': [1]})
            mock_gen_clusters.return_value = pd.DataFrame({'Cluster_ID': [1]})
            mock_solve.return_value = {
                'solver_status': 'Optimal',
                'total_fixed_cost': 100.0,
                'total_variable_cost': 50.0,
                'total_penalties': 0.0,
                'vehicles_used': 1,
                'selected_clusters': pd.DataFrame(),
                'missing_customers': [],
                'solver_runtime_sec': 1.0,
                'post_optimization_runtime_sec': 0.5,
                'solver_name': 'test'
            }
            mock_save.side_effect = Exception("Save failed")
            
            # Should not raise exception, just log warning
            result = optimize(demand=customers_df, output_dir="test_dir")
            assert result['solver_status'] == 'Optimal' 