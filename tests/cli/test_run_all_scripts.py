"""
Unit tests for run_all_* batch processing scripts.

These tests mock the expensive operations (conversion, optimization, saving)
to focus on testing the script logic flow without running actual computations.
"""
import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import warnings
import time
import pandas as pd

from fleetmix.cli import run_all_cvrp, run_all_mcvrp
from fleetmix.pipeline.vrp_interface import VRPType
from fleetmix.benchmarking.converters.cvrp import CVRPBenchmarkType
from fleetmix.core_types import FleetmixSolution


class TestRunAllCVRP(unittest.TestCase):
    """Test cases for run_all_cvrp.py script."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock solution data
        self.mock_solution = {
            "solver_name": "test_solver",
            "solver_status": "optimal", 
            "solver_runtime_sec": 1.5,
            "post_optimization_runtime_sec": 0.5,
            "selected_clusters": pd.DataFrame(),
            "total_fixed_cost": 100.0,
            "total_variable_cost": 50.0,
            "total_light_load_penalties": 0.0,
            "total_compartment_penalties": 0.0,
            "total_penalties": 0.0,
            "vehicles_used": {"SmallTruck": 2},
            "missing_customers": []
        }
        
        self.mock_configs_df = pd.DataFrame([{
            'Config_ID': 1,
            'Vehicle_Type': 'SmallTruck',
            'Capacity': 100,
            'Fixed_Cost': 100
        }])
        
        self.mock_customers_df = pd.DataFrame([{
            'Customer_ID': 'C1',
            'Dry_Demand': 10,
            'Chilled_Demand': 0,
            'Frozen_Demand': 0
        }])
        
        # Mock parameters
        self.mock_params = MagicMock()
        self.mock_params.results_dir = Path("/tmp/results")
        self.mock_params.expected_vehicles = 2

    @patch('fleetmix.cli.run_all_cvrp.save_optimization_results')
    @patch('fleetmix.cli.run_all_cvrp.run_optimization')
    @patch('fleetmix.cli.run_all_cvrp.convert_to_fsm')
    @patch('fleetmix.cli.run_all_cvrp.setup_logging')
    @patch('fleetmix.cli.run_all_cvrp.time')
    def test_main_with_mock_instances(self, mock_time_module, mock_setup_logging, 
                                     mock_convert, mock_run_opt, mock_save):
        """Test main function with mocked VRP instances."""
        # Create mock VRP files
        mock_vrp_files = [
            Path("/fake/path/X-n101-k25.vrp"),
            Path("/fake/path/X-n106-k14.vrp")
        ]
        
        # Setup mocks
        mock_time_module.time.side_effect = [0.0, 2.0, 5.0, 7.0]  # start times and end times
        mock_convert.return_value = (self.mock_customers_df, self.mock_params)
        mock_run_opt.return_value = (self.mock_solution, self.mock_configs_df)
        
        # Mock the directory glob
        with patch.object(Path, 'glob') as mock_glob:
            mock_glob.return_value = sorted(mock_vrp_files)
            
            # Mock Path.parent chain to return our fake directory
            with patch.object(Path, 'parent', new_callable=lambda: property(
                lambda self: Path("/fake/benchmarking/datasets/cvrp")
            )):
                # Test the function
                run_all_cvrp.main()
        
        # Verify logging setup was called
        mock_setup_logging.assert_called_once()
        
        # Verify convert_to_fsm was called for each instance
        expected_convert_calls = [
            call(VRPType.CVRP, instance_names=["X-n101-k25"], benchmark_type=CVRPBenchmarkType.NORMAL),
            call(VRPType.CVRP, instance_names=["X-n106-k14"], benchmark_type=CVRPBenchmarkType.NORMAL)
        ]
        mock_convert.assert_has_calls(expected_convert_calls)
        
        # Verify run_optimization was called for each instance
        expected_opt_calls = [
            call(customers_df=self.mock_customers_df, params=self.mock_params, verbose=False),
            call(customers_df=self.mock_customers_df, params=self.mock_params, verbose=False)
        ]
        mock_run_opt.assert_has_calls(expected_opt_calls)
        
        # Verify save_optimization_results was called for each instance
        self.assertEqual(mock_save.call_count, 2)
        
        # Check that the correct filenames were used
        save_calls = mock_save.call_args_list
        self.assertIn("cvrp_X-n101-k25_normal.json", str(save_calls[0]))
        self.assertIn("cvrp_X-n106-k14_normal.json", str(save_calls[1]))

    @patch('fleetmix.cli.run_all_cvrp.save_optimization_results')
    @patch('fleetmix.cli.run_all_cvrp.run_optimization')
    @patch('fleetmix.cli.run_all_cvrp.convert_to_fsm')
    @patch('fleetmix.cli.run_all_cvrp.setup_logging')
    def test_main_with_no_instances(self, mock_setup_logging, mock_convert, mock_run_opt, mock_save):
        """Test main function when no VRP instances are found."""
        # Mock empty glob result
        with patch.object(Path, 'glob') as mock_glob:
            mock_glob.return_value = []
            
            with patch.object(Path, 'parent', new_callable=lambda: property(
                lambda self: Path("/fake/benchmarking/datasets/cvrp")
            )):
                run_all_cvrp.main()
        
        # Verify setup was called but no processing occurred
        mock_setup_logging.assert_called_once()
        mock_convert.assert_not_called()
        mock_run_opt.assert_not_called()
        mock_save.assert_not_called()

    @patch('fleetmix.cli.run_all_cvrp.setup_logging')
    def test_main_deprecation_warning(self, mock_setup_logging):
        """Test that deprecation warning is raised."""
        with patch.object(Path, 'glob') as mock_glob:
            mock_glob.return_value = []
            
            with patch.object(Path, 'parent', new_callable=lambda: property(
                lambda self: Path("/fake/benchmarking/datasets/cvrp")
            )):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    run_all_cvrp.main()
                    
                    # Check that warning was raised
                    self.assertEqual(len(w), 1)
                    self.assertTrue(issubclass(w[0].category, FutureWarning))
                    self.assertIn("Direct script execution is deprecated", str(w[0].message))
                    self.assertIn("fleetmix benchmark cvrp", str(w[0].message))

    @patch('fleetmix.cli.run_all_cvrp.save_optimization_results')
    @patch('fleetmix.cli.run_all_cvrp.run_optimization')
    @patch('fleetmix.cli.run_all_cvrp.convert_to_fsm')
    @patch('fleetmix.cli.run_all_cvrp.setup_logging')
    def test_convert_to_fsm_parameters(self, mock_setup_logging, mock_convert, mock_run_opt, mock_save):
        """Test that convert_to_fsm is called with correct parameters."""
        mock_vrp_files = [Path("/fake/path/X-n101-k25.vrp")]
        mock_convert.return_value = (self.mock_customers_df, self.mock_params)
        mock_run_opt.return_value = (self.mock_solution, self.mock_configs_df)
        
        with patch.object(Path, 'glob') as mock_glob:
            mock_glob.return_value = mock_vrp_files
            
            with patch.object(Path, 'parent', new_callable=lambda: property(
                lambda self: Path("/fake/benchmarking/datasets/cvrp")
            )):
                run_all_cvrp.main()
        
        # Verify convert_to_fsm was called with correct VRP type and benchmark type
        mock_convert.assert_called_once_with(
            VRPType.CVRP,
            instance_names=["X-n101-k25"],
            benchmark_type=CVRPBenchmarkType.NORMAL
        )

    @patch('fleetmix.cli.run_all_cvrp.save_optimization_results')
    @patch('fleetmix.cli.run_all_cvrp.run_optimization')
    @patch('fleetmix.cli.run_all_cvrp.convert_to_fsm')
    @patch('fleetmix.cli.run_all_cvrp.setup_logging')
    def test_run_optimization_parameters(self, mock_setup_logging, mock_convert, mock_run_opt, mock_save):
        """Test that run_optimization is called with correct parameters."""
        mock_vrp_files = [Path("/fake/path/X-n101-k25.vrp")]
        mock_convert.return_value = (self.mock_customers_df, self.mock_params)
        mock_run_opt.return_value = (self.mock_solution, self.mock_configs_df)
        
        with patch.object(Path, 'glob') as mock_glob:
            mock_glob.return_value = mock_vrp_files
            
            with patch.object(Path, 'parent', new_callable=lambda: property(
                lambda self: Path("/fake/benchmarking/datasets/cvrp")
            )):
                run_all_cvrp.main()
        
        # Verify run_optimization was called with correct parameters
        mock_run_opt.assert_called_once_with(
            customers_df=self.mock_customers_df,
            params=self.mock_params,
            verbose=False
        )


class TestRunAllMCVRP(unittest.TestCase):
    """Test cases for run_all_mcvrp.py script."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock solution data (same as CVRP)
        self.mock_solution = {
            "solver_name": "test_solver",
            "solver_status": "optimal",
            "solver_runtime_sec": 1.5,
            "post_optimization_runtime_sec": 0.5,
            "selected_clusters": pd.DataFrame(),
            "total_fixed_cost": 100.0,
            "total_variable_cost": 50.0,
            "total_light_load_penalties": 0.0,
            "total_compartment_penalties": 0.0,
            "total_penalties": 0.0,
            "vehicles_used": {"MCVRP": 2},
            "missing_customers": []
        }
        
        self.mock_configs_df = pd.DataFrame([{
            'Config_ID': 1,
            'Vehicle_Type': 'MCVRP',
            'Capacity': 100,
            'Fixed_Cost': 100
        }])
        
        self.mock_customers_df = pd.DataFrame([{
            'Customer_ID': 'C1',
            'Dry_Demand': 10,
            'Chilled_Demand': 5,
            'Frozen_Demand': 3
        }])
        
        # Mock parameters
        self.mock_params = MagicMock()
        self.mock_params.results_dir = Path("/tmp/results")
        self.mock_params.expected_vehicles = 2

    @patch('fleetmix.cli.run_all_mcvrp.save_optimization_results')
    @patch('fleetmix.cli.run_all_mcvrp.run_optimization')
    @patch('fleetmix.cli.run_all_mcvrp.convert_to_fsm')
    @patch('fleetmix.cli.run_all_mcvrp.setup_logging')
    @patch('fleetmix.cli.run_all_mcvrp.time')
    def test_main_with_mock_instances(self, mock_time_module, mock_setup_logging,
                                     mock_convert, mock_run_opt, mock_save):
        """Test main function with mocked MCVRP instances."""
        # Create mock DAT files
        mock_dat_files = [
            Path("/fake/path/10_3_3_3_(01).dat"),
            Path("/fake/path/15_3_3_3_(01).dat")
        ]
        
        # Setup mocks
        mock_time_module.time.side_effect = [0.0, 2.0, 5.0, 7.0]  # start times and end times
        mock_convert.return_value = (self.mock_customers_df, self.mock_params)
        mock_run_opt.return_value = (self.mock_solution, self.mock_configs_df)
        
        # Mock the directory glob
        with patch.object(Path, 'glob') as mock_glob:
            mock_glob.return_value = sorted(mock_dat_files)
            
            # Mock Path.parent chain to return our fake directory
            with patch.object(Path, 'parent', new_callable=lambda: property(
                lambda self: Path("/fake/benchmarking/datasets/mcvrp")
            )):
                # Test the function
                run_all_mcvrp.main()
        
        # Verify logging setup was called
        mock_setup_logging.assert_called_once()
        
        # Verify convert_to_fsm was called for each instance
        expected_convert_calls = [
            call(VRPType.MCVRP, instance_path=mock_dat_files[0]),
            call(VRPType.MCVRP, instance_path=mock_dat_files[1])
        ]
        mock_convert.assert_has_calls(expected_convert_calls)
        
        # Verify run_optimization was called for each instance
        expected_opt_calls = [
            call(customers_df=self.mock_customers_df, params=self.mock_params, verbose=False),
            call(customers_df=self.mock_customers_df, params=self.mock_params, verbose=False)
        ]
        mock_run_opt.assert_has_calls(expected_opt_calls)
        
        # Verify save_optimization_results was called for each instance
        self.assertEqual(mock_save.call_count, 2)
        
        # Check that the correct filenames were used
        save_calls = mock_save.call_args_list
        self.assertIn("mcvrp_10_3_3_3_(01).json", str(save_calls[0]))
        self.assertIn("mcvrp_15_3_3_3_(01).json", str(save_calls[1]))

    @patch('fleetmix.cli.run_all_mcvrp.save_optimization_results')
    @patch('fleetmix.cli.run_all_mcvrp.run_optimization')
    @patch('fleetmix.cli.run_all_mcvrp.convert_to_fsm')
    @patch('fleetmix.cli.run_all_mcvrp.setup_logging')
    def test_main_with_no_instances(self, mock_setup_logging, mock_convert, mock_run_opt, mock_save):
        """Test main function when no MCVRP instances are found."""
        # Mock empty glob result
        with patch.object(Path, 'glob') as mock_glob:
            mock_glob.return_value = []
            
            with patch.object(Path, 'parent', new_callable=lambda: property(
                lambda self: Path("/fake/benchmarking/datasets/mcvrp")
            )):
                run_all_mcvrp.main()
        
        # Verify setup was called but no processing occurred
        mock_setup_logging.assert_called_once()
        mock_convert.assert_not_called()
        mock_run_opt.assert_not_called()
        mock_save.assert_not_called()

    @patch('fleetmix.cli.run_all_mcvrp.setup_logging')
    def test_main_deprecation_warning(self, mock_setup_logging):
        """Test that deprecation warning is raised."""
        with patch.object(Path, 'glob') as mock_glob:
            mock_glob.return_value = []
            
            with patch.object(Path, 'parent', new_callable=lambda: property(
                lambda self: Path("/fake/benchmarking/datasets/mcvrp")
            )):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    run_all_mcvrp.main()
                    
                    # Check that warning was raised
                    self.assertEqual(len(w), 1)
                    self.assertTrue(issubclass(w[0].category, FutureWarning))
                    self.assertIn("Direct script execution is deprecated", str(w[0].message))
                    self.assertIn("fleetmix benchmark mcvrp", str(w[0].message))

    @patch('fleetmix.cli.run_all_mcvrp.save_optimization_results')
    @patch('fleetmix.cli.run_all_mcvrp.run_optimization')
    @patch('fleetmix.cli.run_all_mcvrp.convert_to_fsm')
    @patch('fleetmix.cli.run_all_mcvrp.setup_logging')
    def test_convert_to_fsm_parameters(self, mock_setup_logging, mock_convert, mock_run_opt, mock_save):
        """Test that convert_to_fsm is called with correct parameters."""
        mock_dat_files = [Path("/fake/path/10_3_3_3_(01).dat")]
        mock_convert.return_value = (self.mock_customers_df, self.mock_params)
        mock_run_opt.return_value = (self.mock_solution, self.mock_configs_df)
        
        with patch.object(Path, 'glob') as mock_glob:
            mock_glob.return_value = mock_dat_files
            
            with patch.object(Path, 'parent', new_callable=lambda: property(
                lambda self: Path("/fake/benchmarking/datasets/mcvrp")
            )):
                run_all_mcvrp.main()
        
        # Verify convert_to_fsm was called with correct VRP type and instance path
        mock_convert.assert_called_once_with(
            VRPType.MCVRP,
            instance_path=mock_dat_files[0]
        )

    @patch('fleetmix.cli.run_all_mcvrp.save_optimization_results')
    @patch('fleetmix.cli.run_all_mcvrp.run_optimization')
    @patch('fleetmix.cli.run_all_mcvrp.convert_to_fsm')
    @patch('fleetmix.cli.run_all_mcvrp.setup_logging')
    def test_run_optimization_parameters(self, mock_setup_logging, mock_convert, mock_run_opt, mock_save):
        """Test that run_optimization is called with correct parameters."""
        mock_dat_files = [Path("/fake/path/10_3_3_3_(01).dat")]
        mock_convert.return_value = (self.mock_customers_df, self.mock_params)
        mock_run_opt.return_value = (self.mock_solution, self.mock_configs_df)
        
        with patch.object(Path, 'glob') as mock_glob:
            mock_glob.return_value = mock_dat_files
            
            with patch.object(Path, 'parent', new_callable=lambda: property(
                lambda self: Path("/fake/benchmarking/datasets/mcvrp")
            )):
                run_all_mcvrp.main()
        
        # Verify run_optimization was called with correct parameters
        mock_run_opt.assert_called_once_with(
            customers_df=self.mock_customers_df,
            params=self.mock_params,
            verbose=False
        )


if __name__ == "__main__":
    unittest.main() 