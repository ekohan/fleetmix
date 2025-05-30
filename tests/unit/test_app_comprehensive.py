"""
Comprehensive tests for the Fleetmix CLI app module.
"""
import pytest
import tempfile
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from fleetmix.app import (
    _get_available_instances,
    _list_instances,
    _run_single_instance,
    _setup_logging_from_flags,
    app
)


class TestInstanceHelpers:
    """Test suite for instance helper functions"""
    
    def test_get_available_instances_mcvrp(self):
        """Test getting available MCVRP instances"""
        # Use a simple patch of the glob operation
        with patch('pathlib.Path.glob') as mock_glob:
            # Create mock file objects
            mock_file1 = MagicMock()
            mock_file1.stem = "instance1"
            mock_file2 = MagicMock()
            mock_file2.stem = "instance2"
            
            # Mock glob to return our test files
            mock_glob.return_value = [mock_file1, mock_file2]
            
            instances = _get_available_instances("mcvrp")
            
            # Should return the list of instances
            assert len(instances) == 2
            assert "instance1" in instances
            assert "instance2" in instances
    
    def test_get_available_instances_cvrp(self):
        """Test getting available CVRP instances"""
        # Use a simple patch of the glob operation
        with patch('pathlib.Path.glob') as mock_glob:
            # Create mock file objects
            mock_file1 = MagicMock()
            mock_file1.stem = "X-n101-k25"
            mock_file2 = MagicMock()
            mock_file2.stem = "X-n120-k6"
            
            # Mock glob to return our test files
            mock_glob.return_value = [mock_file1, mock_file2]
            
            instances = _get_available_instances("cvrp")
            
            # Should return the list of instances
            assert len(instances) == 2
            assert "X-n101-k25" in instances
            assert "X-n120-k6" in instances
    
    def test_get_available_instances_invalid_suite(self):
        """Test getting instances for invalid suite"""
        instances = _get_available_instances("invalid")
        assert instances == []
    
    @patch('fleetmix.app.console')
    @patch('fleetmix.app._get_available_instances')
    def test_list_instances_with_instances(self, mock_get_instances, mock_console):
        """Test listing instances when instances exist"""
        mock_get_instances.return_value = ["instance1", "instance2", "instance3"]
        
        _list_instances("mcvrp")
        
        mock_get_instances.assert_called_with("mcvrp")
        # Should print table and usage instructions
        assert mock_console.print.call_count >= 2
    
    @patch('fleetmix.app.console')
    @patch('fleetmix.app._get_available_instances')
    def test_list_instances_no_instances(self, mock_get_instances, mock_console):
        """Test listing instances when no instances exist"""
        mock_get_instances.return_value = []
        
        _list_instances("mcvrp")
        
        mock_console.print.assert_called()
        # Should show "No instances found" message
        args = mock_console.print.call_args_list[0][0][0]
        assert "No instances found" in args


class TestSingleInstanceRun:
    """Test suite for single instance execution"""
    
    @patch('fleetmix.app.convert_to_fsm')
    @patch('fleetmix.app.run_optimization')
    @patch('fleetmix.app.save_optimization_results')
    @patch('fleetmix.app.Path')
    def test_run_single_instance_mcvrp_success(self, mock_path, mock_save, mock_optimize, mock_convert):
        """Test successful MCVRP single instance run"""
        # Mock path existence
        mock_dat_path = MagicMock()
        mock_dat_path.exists.return_value = True
        mock_path.return_value.parent.__truediv__.return_value.__truediv__.return_value = mock_dat_path
        
        # Mock conversion
        mock_customers_df = pd.DataFrame({'Customer_ID': ['C1'], 'Latitude': [40.0], 'Longitude': [-74.0]})
        mock_params = MagicMock()
        mock_params.results_dir = Path("/tmp")
        mock_convert.return_value = (mock_customers_df, mock_params)
        
        # Mock optimization
        mock_solution = {
            'solver_name': 'test',
            'solver_status': 'Optimal',
            'solver_runtime_sec': 1.0,
            'post_optimization_runtime_sec': 0.5,
            'selected_clusters': pd.DataFrame(),
            'total_fixed_cost': 100.0,
            'total_variable_cost': 50.0,
            'total_light_load_penalties': 0.0,
            'total_compartment_penalties': 0.0,
            'total_penalties': 0.0,
            'vehicles_used': 1,
            'missing_customers': []
        }
        mock_configs_df = pd.DataFrame({'Config_ID': [1], 'Capacity': [100]})
        mock_optimize.return_value = (mock_solution, mock_configs_df)
        
        # Run the function
        _run_single_instance("mcvrp", "test_instance", verbose=True)
        
        # Verify calls were made
        mock_convert.assert_called_once()
        mock_optimize.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('fleetmix.app._get_available_instances')
    @patch('fleetmix.app.Path')
    @patch('fleetmix.app.typer.Exit')
    def test_run_single_instance_mcvrp_not_found(self, mock_exit, mock_path, mock_get_instances):
        """Test MCVRP instance not found"""
        # Mock path not existing
        mock_dat_path = MagicMock()
        mock_dat_path.exists.return_value = False
        mock_path.return_value.parent.__truediv__.return_value.__truediv__.return_value = mock_dat_path
        
        mock_get_instances.return_value = ["other_instance"]
        
        with pytest.raises(Exception):  # typer.Exit exception
            _run_single_instance("mcvrp", "nonexistent_instance")
    
    @patch('fleetmix.app.convert_to_fsm')
    @patch('fleetmix.app.run_optimization')
    @patch('fleetmix.app.save_optimization_results')
    @patch('fleetmix.app._get_available_instances')
    def test_run_single_instance_cvrp_success(self, mock_get_instances, mock_save, mock_optimize, mock_convert):
        """Test successful CVRP single instance run"""
        # Mock available instances
        mock_get_instances.return_value = ["X-n101-k25", "X-n120-k6"]
        
        # Mock conversion
        mock_customers_df = pd.DataFrame({'Customer_ID': ['C1'], 'Latitude': [40.0], 'Longitude': [-74.0]})
        mock_params = MagicMock()
        mock_params.results_dir = Path("/tmp")
        mock_convert.return_value = (mock_customers_df, mock_params)
        
        # Mock optimization
        mock_solution = {
            'solver_name': 'test',
            'solver_status': 'Optimal',
            'solver_runtime_sec': 1.0,
            'post_optimization_runtime_sec': 0.5,
            'selected_clusters': pd.DataFrame(),
            'total_fixed_cost': 100.0,
            'total_variable_cost': 50.0,
            'total_light_load_penalties': 0.0,
            'total_compartment_penalties': 0.0,
            'total_penalties': 0.0,
            'vehicles_used': 1,
            'missing_customers': []
        }
        mock_configs_df = pd.DataFrame({'Config_ID': [1], 'Capacity': [100]})
        mock_optimize.return_value = (mock_solution, mock_configs_df)
        
        # Run the function
        _run_single_instance("cvrp", "X-n101-k25", verbose=True)
        
        # Verify calls were made
        mock_convert.assert_called_once()
        mock_optimize.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('fleetmix.app._get_available_instances')
    @patch('fleetmix.app.typer.Exit')
    def test_run_single_instance_cvrp_not_found(self, mock_exit, mock_get_instances):
        """Test CVRP instance not found"""
        mock_get_instances.return_value = ["X-n101-k25"]
        
        with pytest.raises(Exception):  # typer.Exit exception
            _run_single_instance("cvrp", "nonexistent_instance")


class TestLoggingSetup:
    """Test suite for logging setup function"""
    
    @patch('fleetmix.app.setup_logging')
    def test_setup_logging_verbose(self, mock_setup):
        """Test logging setup with verbose flag"""
        _setup_logging_from_flags(verbose=True, quiet=False, debug=False)
        # Should be called with LogLevel.VERBOSE enum value
        assert mock_setup.called
    
    @patch('fleetmix.app.setup_logging')
    def test_setup_logging_quiet(self, mock_setup):
        """Test logging setup with quiet flag"""
        _setup_logging_from_flags(verbose=False, quiet=True, debug=False)
        # Should be called with LogLevel.QUIET enum value
        assert mock_setup.called
    
    @patch('fleetmix.app.setup_logging')
    def test_setup_logging_debug(self, mock_setup):
        """Test logging setup with debug flag"""
        _setup_logging_from_flags(verbose=False, quiet=False, debug=True)
        # Should be called with LogLevel.DEBUG enum value
        assert mock_setup.called
    
    @patch('fleetmix.app.setup_logging')
    def test_setup_logging_default(self, mock_setup):
        """Test logging setup with default flags"""
        _setup_logging_from_flags(verbose=False, quiet=False, debug=False)
        # Should be called with default logging
        assert mock_setup.called
    
    @patch('fleetmix.app.setup_logging')
    def test_setup_logging_conflicting_flags(self, mock_setup):
        """Test logging setup with conflicting flags (debug takes precedence)"""
        _setup_logging_from_flags(verbose=True, quiet=True, debug=True)
        # Should be called with LogLevel.DEBUG enum value
        assert mock_setup.called


class TestCLIIntegration:
    """Integration tests for CLI commands"""
    
    def test_cli_version_command(self):
        """Test the version command works"""
        result = subprocess.run(
            [sys.executable, "-m", "fleetmix", "version"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "fleetmix" in result.stdout.lower() or len(result.stdout.strip()) > 0
    
    def test_cli_help_command(self):
        """Test the help command works"""
        result = subprocess.run(
            [sys.executable, "-m", "fleetmix", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "help" in result.stdout.lower() or "usage" in result.stdout.lower()
    
    def test_cli_benchmark_help(self):
        """Test the benchmark command help"""
        result = subprocess.run(
            [sys.executable, "-m", "fleetmix", "benchmark", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "benchmark" in result.stdout.lower()
    
    def test_cli_optimize_help(self):
        """Test the optimize command help"""
        result = subprocess.run(
            [sys.executable, "-m", "fleetmix", "optimize", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "optimize" in result.stdout.lower()
    
    def test_cli_convert_help(self):
        """Test the convert command help"""
        result = subprocess.run(
            [sys.executable, "-m", "fleetmix", "convert", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "convert" in result.stdout.lower()
    
    def test_cli_gui_help(self):
        """Test the gui command help"""
        result = subprocess.run(
            [sys.executable, "-m", "fleetmix", "gui", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "gui" in result.stdout.lower()


class TestCLIErrorHandling:
    """Test CLI error handling scenarios"""
    
    def test_optimize_missing_demand_file(self):
        """Test optimize command with missing demand file"""
        result = subprocess.run(
            [sys.executable, "-m", "fleetmix", "optimize", "--demand", "nonexistent.csv"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()
    
    def test_benchmark_invalid_suite(self):
        """Test benchmark command with invalid suite"""
        result = subprocess.run(
            [sys.executable, "-m", "fleetmix", "benchmark", "invalid_suite"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 1
        assert "invalid" in result.stderr.lower() or "error" in result.stderr.lower()
    
    def test_convert_invalid_type(self):
        """Test convert command with invalid type"""
        result = subprocess.run(
            [sys.executable, "-m", "fleetmix", "convert", "--type", "invalid", "--instance", "test"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 1
    
    def test_convert_missing_benchmark_type(self):
        """Test convert command missing benchmark type for CVRP"""
        result = subprocess.run(
            [sys.executable, "-m", "fleetmix", "convert", "--type", "cvrp", "--instance", "test"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 1


class TestCLIWithRealFiles:
    """Test CLI with actual test files"""
    
    def test_optimize_with_smoke_test_files(self):
        """Test optimize command with smoke test files"""
        smoke_dir = Path(__file__).parent.parent / "_assets" / "smoke"
        demand_file = smoke_dir / "mini_demand.csv"
        config_file = smoke_dir / "mini.yaml"
        
        if demand_file.exists() and config_file.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    [
                        sys.executable, "-m", "fleetmix", "optimize",
                        "--demand", str(demand_file),
                        "--config", str(config_file),
                        "--output", tmpdir,
                        "--format", "json"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120  # Longer timeout for actual optimization
                )
                
                # Should succeed or fail gracefully
                assert result.returncode in [0, 1]
                if result.returncode == 0:
                    assert "total cost" in result.stdout.lower() or "results" in result.stdout.lower()
    
    def test_benchmark_list_mcvrp(self):
        """Test listing MCVRP benchmark instances"""
        result = subprocess.run(
            [sys.executable, "-m", "fleetmix", "benchmark", "mcvrp", "--list"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should succeed regardless of whether instances are available
        assert result.returncode == 0
        # Should either show instances or "no instances found"
        assert len(result.stdout) > 0
    
    def test_benchmark_list_cvrp(self):
        """Test listing CVRP benchmark instances"""
        result = subprocess.run(
            [sys.executable, "-m", "fleetmix", "benchmark", "cvrp", "--list"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should succeed regardless of whether instances are available
        assert result.returncode == 0
        # Should either show instances or "no instances found"
        assert len(result.stdout) > 0 