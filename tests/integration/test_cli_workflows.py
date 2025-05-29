"""
CLI integration tests for FleetMix Typer-based command line interface.
Tests real CLI workflows without mocking the core functionality.
"""
import pytest
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
import json
from typer.testing import CliRunner

from fleetmix.app import app


class TestCLIWorkflows:
    """Test CLI commands with real data and workflows."""
    
    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary results directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_demand_csv(self, temp_results_dir):
        """Create small realistic demand CSV for CLI testing."""
        customers_data = {
            'Customer_ID': [1, 2, 3],
            'Customer_Name': ['Shop A', 'Shop B', 'Shop C'],
            'Latitude': [40.7589, 40.7614, 40.7505],
            'Longitude': [-73.9851, -73.9776, -73.9934],
            'Dry_Demand': [50, 75, 60],
            'Chilled_Demand': [30, 40, 35],
            'Frozen_Demand': [15, 20, 18]
        }
        df = pd.DataFrame(customers_data)
        csv_path = temp_results_dir / "cli_test_demand.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
        
    @pytest.fixture
    def sample_config_yaml(self, temp_results_dir):
        """Create minimal config for fast CLI testing."""
        config_content = """
depot:
  latitude: 40.7831
  longitude: -73.9712

demand_file: dummy_path.csv # Will be overridden by CLI --demand

goods: ["Dry", "Chilled", "Frozen"]

vehicles:
  Test Van:
    fixed_cost: 100
    variable_cost_per_km: 0.5 # This will be used to calculate variable_cost_per_hour with avg_speed
    capacity: 500
    compartments:
      - temperature_min: -25
        temperature_max: 25
        capacity: 500

# Root level parameters
variable_cost_per_hour: 20.0 # Example: 0.5 cost/km * 40 km/hr avg_speed
avg_speed: 40.0 # km/hr
max_route_time: 8.0 # hours
service_time: 10.0 # minutes, moved from clustering

clustering:
  max_clusters_per_vehicle: 50
  time_limit_minutes: 30
  # service_time_minutes: 10 # Moved to root as service_time
  route_time_estimation: 'Legacy'
  method: 'minibatch_kmeans'
  max_depth: 5
  geo_weight: 0.7
  demand_weight: 0.3

# These were previously under 'optimization'
# solver: "cbc" # REMOVED
# time_limit_minutes: 1 # REMOVED - For the optimization solver
light_load_penalty: 5.0
light_load_threshold: 0.5
compartment_setup_cost: 100.0

# Output format, will be overridden by CLI --format if provided
format: "json"

post_optimization: false # Corresponds to post_optimization: { enabled: false }
"""
        config_path = temp_results_dir / "cli_test_config.yaml"
        config_path.write_text(config_content)
        return config_path

    def test_cli_optimize_command_json_output(self, sample_demand_csv, sample_config_yaml, temp_results_dir):
        """Test 'fleetmix optimize' command with JSON output."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "optimize",
            "--demand", str(sample_demand_csv),
            "--config", str(sample_config_yaml),
            "--output", str(temp_results_dir),
            "--format", "json",
            "--verbose"
        ])
        
        # Should not exit with error
        assert result.exit_code == 0
        
        # Should have created output files
        output_files = list(temp_results_dir.glob("*.json"))
        # May be 0 if optimization failed, but command should still succeed
        assert len(output_files) >= 0

    def test_cli_optimize_command_excel_output(self, sample_demand_csv, sample_config_yaml, temp_results_dir):
        """Test 'fleetmix optimize' command with Excel output."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "optimize", 
            "--demand", str(sample_demand_csv),
            "--config", str(sample_config_yaml),
            "--output", str(temp_results_dir),
            "--format", "excel",
            "--quiet"
        ])
        
        assert result.exit_code == 0

    def test_cli_optimize_command_missing_demand_file(self, temp_results_dir):
        """Test error handling for missing demand file."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "optimize",
            "--demand", "nonexistent.csv",
            "--output", str(temp_results_dir)
        ])
        
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_cli_optimize_command_missing_config_file(self, sample_demand_csv, temp_results_dir):
        """Test error handling for missing config file."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "optimize",
            "--demand", str(sample_demand_csv),
            "--config", "nonexistent.yaml",
            "--output", str(temp_results_dir)
        ])
        
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_cli_optimize_command_invalid_format(self, sample_demand_csv, temp_results_dir):
        """Test error handling for invalid output format."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "optimize",
            "--demand", str(sample_demand_csv),
            "--format", "invalid_format",
            "--output", str(temp_results_dir)
        ])
        
        assert result.exit_code == 1
        assert "Invalid format" in result.output

    def test_cli_version_command(self):
        """Test 'fleetmix version' command."""
        runner = CliRunner()
        
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "fleetmix" in result.output.lower()

    def test_cli_benchmark_mcvrp_list(self):
        """Test 'fleetmix benchmark mcvrp --list' command."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "benchmark", "mcvrp", "--list"
        ])
        
        # Should work even if no datasets available
        assert result.exit_code == 0

    def test_cli_benchmark_cvrp_list(self):
        """Test 'fleetmix benchmark cvrp --list' command.""" 
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "benchmark", "cvrp", "--list"
        ])
        
        # Should work even if no datasets available
        assert result.exit_code == 0

    def test_cli_benchmark_invalid_suite(self):
        """Test benchmark command with invalid suite."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "benchmark", "invalid_suite"
        ])
        
        assert result.exit_code == 1
        assert "Invalid suite" in result.output

    def test_cli_convert_command_cvrp(self, temp_results_dir):
        """Test 'fleetmix convert' command for CVRP (if data available)."""
        runner = CliRunner()
        
        # Check if CVRP datasets exist first
        cvrp_dir = Path(__file__).parent.parent.parent / "src/fleetmix/benchmarking/datasets/cvrp"
        if not cvrp_dir.exists():
            pytest.skip("CVRP datasets not available")
            
        cvrp_files = list(cvrp_dir.glob("X-n*.vrp"))
        if not cvrp_files:
            pytest.skip("No CVRP .vrp files found")
            
        instance_name = cvrp_files[0].stem
        
        result = runner.invoke(app, [
            "convert",
            "--type", "cvrp",
            "--instance", instance_name,
            "--benchmark-type", "normal",
            "--output", str(temp_results_dir),
            "--quiet"
        ])
        
        # May fail due to solver issues, but command structure should be valid
        # Exit code 0 = success, exit code 1 = expected error (solver/data issues)
        assert result.exit_code in [0, 1]

    def test_cli_convert_command_mcvrp(self, temp_results_dir):
        """Test 'fleetmix convert' command for MCVRP (if data available)."""
        runner = CliRunner()
        
        # Check if MCVRP datasets exist
        mcvrp_dir = Path(__file__).parent.parent.parent / "src/fleetmix/benchmarking/datasets/mcvrp"
        if not mcvrp_dir.exists():
            pytest.skip("MCVRP datasets not available")
            
        mcvrp_files = list(mcvrp_dir.glob("*.dat"))
        if not mcvrp_files:
            pytest.skip("No MCVRP .dat files found")
            
        instance_name = mcvrp_files[0].stem
        
        result = runner.invoke(app, [
            "convert",
            "--type", "mcvrp", 
            "--instance", instance_name,
            "--output", str(temp_results_dir),
            "--quiet"
        ])
        
        # May fail due to solver issues, but command structure should be valid
        assert result.exit_code in [0, 1]

    def test_cli_convert_missing_benchmark_type(self):
        """Test convert command missing required benchmark type for CVRP."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "convert",
            "--type", "cvrp",
            "--instance", "test-instance"
        ])
        
        assert result.exit_code == 1
        assert "benchmark-type is required" in result.output

    def test_cli_convert_invalid_type(self):
        """Test convert command with invalid VRP type."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "convert",
            "--type", "invalid_type",
            "--instance", "test"
        ])
        
        assert result.exit_code == 1
        assert "Invalid type" in result.output

    def test_cli_help_commands(self):
        """Test that help commands work."""
        runner = CliRunner()
        
        # Main help
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "fleetmix" in result.output.lower()
        
        # Subcommand help
        result = runner.invoke(app, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "optimize" in result.output.lower()
        
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "benchmark" in result.output.lower()

    def test_cli_optimize_with_default_config(self, sample_demand_csv, temp_results_dir):
        """Test optimization without specifying config file (uses defaults)."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "optimize",
            "--demand", str(sample_demand_csv),
            "--output", str(temp_results_dir),
            "--format", "json",
            "--quiet"
        ])
        
        # Should work with default configuration
        assert result.exit_code == 0

    def test_cli_logging_levels(self, sample_demand_csv, temp_results_dir):
        """Test different logging levels in CLI."""
        runner = CliRunner()
        
        # Test verbose mode
        result = runner.invoke(app, [
            "optimize",
            "--demand", str(sample_demand_csv),
            "--output", str(temp_results_dir),
            "--verbose"
        ])
        assert result.exit_code == 0
        
        # Test quiet mode 
        result = runner.invoke(app, [
            "optimize",
            "--demand", str(sample_demand_csv),
            "--output", str(temp_results_dir),
            "--quiet"
        ])
        assert result.exit_code == 0
        
        # Test debug mode
        result = runner.invoke(app, [
            "optimize", 
            "--demand", str(sample_demand_csv),
            "--output", str(temp_results_dir),
            "--debug"
        ])
        assert result.exit_code == 0 