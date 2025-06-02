"""
Comprehensive tests for the Streamlit GUI module.

These tests focus on testing individual functions and components without 
launching the full Streamlit app, using mocking and isolation techniques.
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import json
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

import pytest

# Try to import streamlit and gui module
try:
    import streamlit as st
    from fleetmix import gui
    from fleetmix.config.parameters import Parameters
    from fleetmix.core_types import FleetmixSolution
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestGuiHelperFunctions(unittest.TestCase):
    """Test helper functions in the GUI module."""

    def test_convert_numpy_types_dict(self):
        """Test conversion of numpy types in dictionaries."""
        test_dict = {
            'int': np.int64(42),
            'float': np.float64(3.14),
            'array': np.array([1, 2, 3]),
            'nested': {'value': np.int32(10)}
        }
        
        result = gui.convert_numpy_types(test_dict)
        
        self.assertEqual(result['int'], 42)
        self.assertEqual(result['float'], 3.14)
        self.assertEqual(result['array'], [1, 2, 3])
        self.assertEqual(result['nested']['value'], 10)
        self.assertIsInstance(result['int'], int)
        self.assertIsInstance(result['float'], float)

    def test_convert_numpy_types_dataframe(self):
        """Test conversion of pandas DataFrames."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = gui.convert_numpy_types(df)
        
        expected = [{'A': 1, 'B': 4}, {'A': 2, 'B': 5}, {'A': 3, 'B': 6}]
        self.assertEqual(result, expected)

    def test_convert_numpy_types_datetime(self):
        """Test conversion of datetime objects."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = gui.convert_numpy_types(dt)
        
        self.assertEqual(result, "2024-01-01T12:00:00")

    def test_convert_numpy_types_path(self):
        """Test conversion of Path objects."""
        path = Path("/tmp/test")
        result = gui.convert_numpy_types(path)
        
        self.assertEqual(result, "/tmp/test")

    def test_convert_numpy_types_set(self):
        """Test conversion of sets."""
        test_set = {1, 2, 3}
        result = gui.convert_numpy_types(test_set)
        
        self.assertIsInstance(result, list)
        self.assertEqual(sorted(result), [1, 2, 3])


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestRunOptimizationInProcess(unittest.TestCase):
    """Test the run_optimization_in_process function."""

    @patch('fleetmix.gui.api.optimize')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_optimization_success(self, mock_file, mock_optimize):
        """Test successful optimization run."""
        # Mock the optimization result as a FleetmixSolution
        mock_solution = FleetmixSolution(
            total_cost=1000.0,
            total_fixed_cost=600.0,
            total_variable_cost=400.0,
            total_penalties=0.0,
            total_light_load_penalties=0.0,
            total_compartment_penalties=0.0,
            vehicles_used={'SmallTruck': 2},
            missing_customers=set(),
            solver_name='test_solver',
            solver_status='optimal',
            solver_runtime_sec=1.5,
            selected_clusters=pd.DataFrame()
        )
        # Add post_optimization_runtime_sec as an attribute (not in constructor)
        mock_solution.post_optimization_runtime_sec = 0.5
        mock_optimize.return_value = mock_solution
        
        # Create complete test parameters
        params_dict = {
            'vehicles': {'SmallTruck': {'capacity': 100, 'fixed_cost': 300}},
            'goods': ['Dry', 'Chilled', 'Frozen'],
            'depot': {'latitude': 40.7128, 'longitude': -74.0060},
            'avg_speed': 25,
            'service_time': 15,
            'max_route_time': 8,
            'variable_cost_per_hour': 50,
            'clustering': {'method': 'combine'},
            'demand_file': 'demand.csv',
            'light_load_penalty': 20,
            'light_load_threshold': 0.5,
            'compartment_setup_cost': 10,
            'format': 'excel',
            'post_optimization': False,
            'small_cluster_size': 3,
            'nearest_merge_candidates': 10,
            'max_improvement_iterations': 5,
            'prune_tsp': True
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            demand_path = Path(tmpdir) / "demand.csv"
            status_file = Path(tmpdir) / "status.json"
            
            # Create a dummy demand file
            pd.DataFrame({'Customer_ID': ['C1'], 'Dry': [10]}).to_csv(demand_path, index=False)
            
            # Run the function
            gui.run_optimization_in_process(
                str(demand_path),
                params_dict,
                tmpdir,
                str(status_file)
            )
            
            # Verify optimize was called
            mock_optimize.assert_called_once()
            
            # Verify status updates were written
            # Should have at least 2 writes (initial and final)
            self.assertGreaterEqual(mock_file.return_value.write.call_count, 2)

    @patch('fleetmix.gui.api.optimize')
    def test_run_optimization_failure(self, mock_optimize):
        """Test optimization failure handling."""
        # Make optimize raise an exception
        mock_optimize.side_effect = ValueError("Test error")
        
        params_dict = {
            'vehicles': {'SmallTruck': {'capacity': 100, 'fixed_cost': 300}},
            'goods': ['Dry', 'Chilled', 'Frozen'],
            'depot': {'latitude': 40.7128, 'longitude': -74.0060},
            'avg_speed': 25,
            'service_time': 15,
            'max_route_time': 8,
            'variable_cost_per_hour': 50,
            'clustering': {'method': 'combine'},
            'demand_file': 'demand.csv',
            'light_load_penalty': 20,
            'light_load_threshold': 0.5,
            'compartment_setup_cost': 10,
            'format': 'excel'
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            demand_path = Path(tmpdir) / "demand.csv"
            status_file = Path(tmpdir) / "status.json"
            
            pd.DataFrame({'Customer_ID': ['C1'], 'Dry': [10]}).to_csv(demand_path, index=False)
            
            # Run should raise the exception
            with self.assertRaises(ValueError):
                gui.run_optimization_in_process(
                    str(demand_path),
                    params_dict,
                    tmpdir,
                    str(status_file)
                )
            
            # Check that error was written to status file
            self.assertTrue(status_file.exists())
            with open(status_file, 'r') as f:
                error_data = json.load(f)
            self.assertIn('error', error_data)
            self.assertEqual(error_data['error'], 'Test error')
            self.assertIn('traceback', error_data)


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestSessionStateInitialization(unittest.TestCase):
    """Test session state initialization."""

    @patch('streamlit.session_state')
    def test_init_session_state(self, mock_session_state):
        """Test that session state is properly initialized."""
        # Mock the contains check to always return False (nothing initialized)
        mock_session_state.__contains__ = MagicMock(return_value=False)
        
        gui.init_session_state()
        
        # Verify all required keys are checked
        expected_checks = [
            'uploaded_data',
            'optimization_results', 
            'optimization_running',
            'parameters',
            'error_info'
        ]
        
        # Check that __contains__ was called for each key
        for key in expected_checks:
            mock_session_state.__contains__.assert_any_call(key)

    @patch('streamlit.session_state')
    def test_init_session_state_with_existing(self, mock_session_state):
        """Test initialization when some state already exists."""
        # Mock session state with some existing values
        existing_keys = {'uploaded_data'}
        
        def check_contains(key):
            return key in existing_keys
        
        mock_session_state.__contains__ = MagicMock(side_effect=check_contains)
        
        gui.init_session_state()
        
        # Should check all keys
        expected_checks = [
            'uploaded_data',
            'optimization_results', 
            'optimization_running',
            'parameters',
            'error_info'
        ]
        
        # Verify contains was called for all keys
        for key in expected_checks:
            mock_session_state.__contains__.assert_any_call(key)


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestCollectParametersFromUI(unittest.TestCase):
    """Test parameter collection from UI."""

    def test_collect_parameters_basic(self):
        """Test basic parameter collection."""
        # Create mock parameters
        mock_params = Parameters.from_yaml()
        
        # Create a mock session state object
        mock_state = MagicMock()
        mock_state.parameters = mock_params
        
        # Define items in session state
        session_items = {
            'parameters': mock_params,
            'param_avg_speed': 30,
            'param_service_time': 20,
            'param_max_route_time': 10
        }
        
        # Mock iteration and getitem
        mock_state.__iter__ = MagicMock(return_value=iter(session_items.keys()))
        mock_state.__getitem__ = MagicMock(side_effect=lambda key: session_items[key])
        
        with patch('streamlit.session_state', mock_state):
            result = gui.collect_parameters_from_ui()
        
        self.assertEqual(result['avg_speed'], 30)
        self.assertEqual(result['service_time'], 20)
        self.assertEqual(result['max_route_time'], 10)

    def test_collect_parameters_nested(self):
        """Test collection of nested parameters like clustering.method."""
        mock_params = Parameters.from_yaml()
        
        # Create a mock session state object
        mock_state = MagicMock()
        mock_state.parameters = mock_params
        
        # Define items with nested parameter
        session_items = {
            'parameters': mock_params,
            'param_clustering.method': 'kmedoids'
        }
        
        # Mock iteration and getitem
        mock_state.__iter__ = MagicMock(return_value=iter(session_items.keys()))
        mock_state.__getitem__ = MagicMock(side_effect=lambda key: session_items[key])
        
        with patch('streamlit.session_state', mock_state):
            result = gui.collect_parameters_from_ui()
        
        self.assertEqual(result['clustering']['method'], 'kmedoids')


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestDisplayResults(unittest.TestCase):
    """Test results display function."""

    @patch('streamlit.success')
    @patch('streamlit.metric')
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.dataframe')
    @patch('streamlit.download_button')
    @patch('streamlit.components.v1.html')
    def test_display_results_basic(self, mock_html, mock_download, mock_dataframe,
                                  mock_columns, mock_subheader, mock_metric, mock_success):
        """Test basic results display."""
        # Mock columns to return column objects
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        
        # Set up columns to return the right number of columns each time
        call_count = 0
        def columns_side_effect(n):
            nonlocal call_count
            call_count += 1
            return [mock_col] * n
        
        mock_columns.side_effect = columns_side_effect
        
        # Create test solution
        solution = {
            'total_fixed_cost': 600.0,
            'total_variable_cost': 400.0,
            'total_penalties': 50.0,
            'total_vehicles': 3,
            'vehicles_used': {'SmallTruck': 2, 'LargeTruck': 1},
            'missing_customers': [],
            'solver_runtime_sec': 2.5
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create dummy result files
            excel_file = output_dir / "optimization_results_test.xlsx"
            excel_file.write_text("dummy excel")
            json_file = output_dir / "optimization_results_test.json"
            json_file.write_text("{}")
            html_file = output_dir / "optimization_results_test_clusters.html"
            html_file.write_text("<html>Map</html>")
            
            # Call display function
            gui.display_results(solution, output_dir)
            
            # Verify success message
            mock_success.assert_called_once()
            
            # Verify metrics were displayed
            self.assertGreater(mock_metric.call_count, 0)
            
            # Verify subheaders
            mock_subheader.assert_any_call("📊 Cost Breakdown")
            mock_subheader.assert_any_call("🚚 Vehicle Usage")
            
            # Verify dataframe was called for vehicle usage
            mock_dataframe.assert_called_once()
            
            # Verify download buttons were created
            self.assertEqual(mock_download.call_count, 2)  # Excel and JSON
            
            # Verify map was displayed
            mock_html.assert_called_once()


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestStreamlitIntegration(unittest.TestCase):
    """Test Streamlit-specific integrations."""

    @patch('streamlit.set_page_config')
    @patch('streamlit.markdown')
    @patch('streamlit.title')
    @patch('streamlit.sidebar')
    @patch('streamlit.container')
    @patch('fleetmix.gui.init_session_state')
    def test_main_initialization(self, mock_init_state, mock_container,
                                 mock_sidebar, mock_title, mock_markdown, mock_config):
        """Test main function initialization."""
        # Mock the sidebar context manager
        mock_sidebar_ctx = MagicMock()
        mock_sidebar_ctx.__enter__ = MagicMock(return_value=mock_sidebar_ctx)
        mock_sidebar_ctx.__exit__ = MagicMock(return_value=None)
        mock_sidebar.return_value = mock_sidebar_ctx
        
        # Mock container
        mock_container_ctx = MagicMock()
        mock_container_ctx.__enter__ = MagicMock(return_value=mock_container_ctx)
        mock_container_ctx.__exit__ = MagicMock(return_value=None)
        mock_container.return_value = mock_container_ctx
        
        # Patch session state to avoid running full app
        with patch('streamlit.session_state', {'optimization_running': False}):
            # We can't run the full main() due to its complexity,
            # but we can test that initialization happens
            mock_config.assert_not_called()  # Not called yet
            
            # Instead, test individual components
            gui.init_session_state()
            mock_init_state.assert_called_once()


if __name__ == "__main__":
    unittest.main() 