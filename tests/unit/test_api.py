"""Test the API module."""

from pathlib import Path

import pandas as pd
import pytest

from fleetmix.api import optimize
from fleetmix.config.parameters import Parameters


@pytest.fixture
def simple_demand_df():
    """Create a simple demand DataFrame for testing."""
    return pd.DataFrame(
        {
            "Customer_ID": ["C1", "C2", "C3"],
            "Customer_Name": ["Customer 1", "Customer 2", "Customer 3"],
            "Latitude": [0.0, 0.1, 0.2],
            "Longitude": [0.0, 0.1, 0.2],
            "Dry_Demand": [10, 15, 20],
            "Chilled_Demand": [0, 5, 0],
            "Frozen_Demand": [0, 0, 10],
        }
    )


@pytest.fixture
def base_config_path():
    """Path to base test configuration."""
    return (
        Path(__file__).parent.parent / "_assets" / "configs" / "base_test_config.yaml"
    )


@pytest.fixture
def minimal_config_path():
    """Path to minimal test configuration."""
    return (
        Path(__file__).parent.parent
        / "_assets"
        / "configs"
        / "test_config_minimal.yaml"
    )


def test_optimize_with_dataframe(simple_demand_df, base_config_path):
    """Test optimization with DataFrame input."""
    # Run optimization
    result = optimize(
        demand=simple_demand_df,
        config=str(base_config_path),
        output_dir=None,  # Don't save results
        verbose=False,
    )

    # Check result structure using attribute access
    assert result.total_fixed_cost is not None
    assert result.total_variable_cost is not None
    assert result.total_penalties is not None
    assert result.vehicles_used is not None
    assert result.selected_clusters is not None
    assert result.missing_customers is not None
    assert result.solver_status is not None
    assert result.solver_runtime_sec is not None

    # Check that solution is optimal
    assert result.solver_status == "Optimal"

    # Check that all customers are served
    assert len(result.missing_customers) == 0


def test_optimize_with_csv_file(simple_demand_df, base_config_path, tmp_path):
    """Test optimization with CSV file input."""
    # Create temporary demand CSV
    demand_path = tmp_path / "demand.csv"
    simple_demand_df.to_csv(demand_path, index=False)

    # Run optimization
    result = optimize(
        demand=str(demand_path),
        config=str(base_config_path),
        output_dir=None,
        verbose=False,
    )

    # Check result
    assert result.solver_status == "Optimal"
    assert len(result.missing_customers) == 0


def test_optimize_with_parameters_object(simple_demand_df):
    """Test optimization with Parameters object input."""
    # Load parameters from YAML file
    config_path = (
        Path(__file__).parent.parent / "_assets" / "configs" / "base_test_config.yaml"
    )
    params = Parameters.from_yaml(str(config_path))

    # Run optimization
    result = optimize(
        demand=simple_demand_df, config=params, output_dir=None, verbose=False
    )

    # Check result
    assert result.solver_status == "Optimal"


def test_optimize_with_missing_columns():
    """Test optimization with DataFrame missing required columns."""
    bad_df = pd.DataFrame(
        {
            "Customer_ID": ["C1"],
            "Latitude": [0.0],
            # Missing Longitude and demand columns
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        optimize(demand=bad_df, output_dir=None)


def test_optimize_with_nonexistent_demand_file():
    """Test optimization with non-existent demand file."""
    with pytest.raises(FileNotFoundError, match="Demand file not found"):
        optimize(demand="nonexistent.csv", output_dir=None)


def test_optimize_with_nonexistent_config_file(simple_demand_df):
    """Test optimization with non-existent config file."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        optimize(demand=simple_demand_df, config="nonexistent.yaml", output_dir=None)


def test_optimize_with_default_config(simple_demand_df):
    """Test optimization with default configuration."""
    # This should use the default configuration
    result = optimize(
        demand=simple_demand_df, config=None, output_dir=None, verbose=False
    )

    # Check result
    assert result.solver_status is not None


def test_optimize_saves_results(simple_demand_df, base_config_path, tmp_path):
    """Test that optimization saves results when output_dir is specified."""
    # Create output directory
    output_dir = tmp_path / "results"

    # Run optimization with output
    result = optimize(
        demand=simple_demand_df,
        config=str(base_config_path),
        output_dir=str(output_dir),
        format="excel",
        verbose=False,
    )

    # Check that results were saved
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.xlsx"))) > 0


def test_optimize_with_infeasible_problem(minimal_config_path):
    """Test optimization with an infeasible problem."""
    # Create demand that exceeds all vehicle capacities
    large_demand_df = pd.DataFrame(
        {
            "Customer_ID": ["C1"],
            "Customer_Name": ["Customer 1"],
            "Latitude": [0.0],
            "Longitude": [0.0],
            "Dry_Demand": [1000],  # Exceeds all capacities in minimal config
            "Chilled_Demand": [0],
            "Frozen_Demand": [0],
        }
    )

    with pytest.raises(ValueError, match="No feasible clusters could be generated"):
        optimize(
            demand=large_demand_df,
            config=str(minimal_config_path),
            output_dir=None,
            verbose=False,
        )


def test_optimize_with_json_format(simple_demand_df, tmp_path):
    """Test optimization with JSON output format."""
    # Use BHH config which has JSON format
    config_path = (
        Path(__file__).parent.parent / "_assets" / "configs" / "test_config_bhh.yaml"
    )

    # Create output directory
    output_dir = tmp_path / "results"

    # Run optimization with JSON output
    result = optimize(
        demand=simple_demand_df,
        config=str(config_path),
        output_dir=str(output_dir),
        format="json",
        verbose=False,
    )

    # Check that JSON results were saved
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.json"))) > 0


def test_optimize_verbose_mode(simple_demand_df, base_config_path, capsys):
    """Test optimization in verbose mode."""
    # Run optimization in verbose mode
    result = optimize(
        demand=simple_demand_df,
        config=str(base_config_path),
        output_dir=None,
        verbose=True,
    )

    # Check that verbose output was produced
    captured = capsys.readouterr()
    assert len(captured.out) > 0  # Some output should be printed
