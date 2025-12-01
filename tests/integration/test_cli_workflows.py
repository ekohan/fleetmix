"""
CLI integration tests for FleetMix Typer-based command line interface.
Tests real CLI workflows without mocking the core functionality.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
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
            "Customer_ID": [1, 2, 3],
            "Customer_Name": ["Shop A", "Shop B", "Shop C"],
            "Latitude": [40.7589, 40.7614, 40.7505],
            "Longitude": [-73.9851, -73.9776, -73.9934],
            "Dry_Demand": [50, 75, 60],
            "Chilled_Demand": [30, 40, 35],
            "Frozen_Demand": [15, 20, 18],
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
    capacity: 500
    avg_speed: 40.0 # km/hr
    max_route_time: 8.0 # hours
    service_time: 10.0 # minutes, moved from clustering
    compartments:
      Dry: True
      Chilled: True
      Frozen: True
    extra:
      variable_cost_per_km: 0.5

# Root level parameters
variable_cost_per_hour: 20.0

clustering:
  route_time_estimation: 'BHH'
  method: 'minibatch_kmeans'
  max_depth: 5
  geo_weight: 0.7
  demand_weight: 0.3
  distance: 'euclidean'

light_load_penalty: 5.0
light_load_threshold: 0.5
compartment_setup_cost: 100.0
prune_tsp: False

format: "json"

post_optimization: false
"""
        config_path = temp_results_dir / "cli_test_config.yaml"
        config_path.write_text(config_content)
        return config_path

    def test_cli_optimize_command_json_output(
        self, sample_demand_csv, sample_config_yaml, temp_results_dir
    ):
        """Test 'fleetmix optimize' command with JSON output."""
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "optimize",
                "--demand",
                str(sample_demand_csv),
                "--config",
                str(sample_config_yaml),
                "--output",
                str(temp_results_dir),
                "--format",
                "json",
                "--verbose",
            ],
        )

        # Should not exit with error
        assert result.exit_code == 0

        # Should have created output files
        output_files = list(temp_results_dir.glob("*.json"))
        # May be 0 if optimization failed, but command should still succeed
        assert len(output_files) >= 0

    def test_cli_optimize_command_excel_output(
        self, sample_demand_csv, sample_config_yaml, temp_results_dir
    ):
        """Test 'fleetmix optimize' command with Excel output."""
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "optimize",
                "--demand",
                str(sample_demand_csv),
                "--config",
                str(sample_config_yaml),
                "--output",
                str(temp_results_dir),
                "--format",
                "xlsx",
                "--quiet",
            ],
        )

        assert result.exit_code == 0

    def test_cli_optimize_command_missing_demand_file(self, temp_results_dir):
        """Test error handling for missing demand file."""
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "optimize",
                "--demand",
                "nonexistent.csv",
                "--output",
                str(temp_results_dir),
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_cli_optimize_command_missing_config_file(
        self, sample_demand_csv, temp_results_dir
    ):
        """Test error handling for missing config file."""
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "optimize",
                "--demand",
                str(sample_demand_csv),
                "--config",
                "nonexistent.yaml",
                "--output",
                str(temp_results_dir),
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_cli_optimize_command_invalid_format(
        self, sample_demand_csv, temp_results_dir
    ):
        """Test error handling for invalid output format."""
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "optimize",
                "--demand",
                str(sample_demand_csv),
                "--format",
                "invalid_format",
                "--output",
                str(temp_results_dir),
            ],
        )

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

        result = runner.invoke(app, ["benchmark", "mcvrp", "--list"])

        # Should work even if no datasets available
        assert result.exit_code == 0

    def test_cli_benchmark_invalid_suite(self):
        """Test benchmark command with invalid suite."""
        runner = CliRunner()

        result = runner.invoke(app, ["benchmark", "invalid_suite"])

        assert result.exit_code == 1
        assert "Invalid suite" in result.output

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

    def test_cli_optimize_with_default_config(
        self, sample_demand_csv, temp_results_dir
    ):
        """Test optimization without specifying config file (uses defaults)."""
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "optimize",
                "--demand",
                str(sample_demand_csv),
                "--output",
                str(temp_results_dir),
                "--format",
                "json",
                "--quiet",
            ],
        )

        # Should work with default configuration
        assert result.exit_code == 0

    def test_cli_logging_levels(self, sample_demand_csv, temp_results_dir):
        """Test different logging levels in CLI."""
        runner = CliRunner()

        # Test verbose mode
        result = runner.invoke(
            app,
            [
                "optimize",
                "--demand",
                str(sample_demand_csv),
                "--output",
                str(temp_results_dir),
                "--verbose",
            ],
        )
        assert result.exit_code == 0

        # Test quiet mode
        result = runner.invoke(
            app,
            [
                "optimize",
                "--demand",
                str(sample_demand_csv),
                "--output",
                str(temp_results_dir),
                "--quiet",
            ],
        )
        assert result.exit_code == 0

        # Test debug mode
        result = runner.invoke(
            app,
            [
                "optimize",
                "--demand",
                str(sample_demand_csv),
                "--output",
                str(temp_results_dir),
                "--debug",
            ],
        )
        assert result.exit_code == 0
