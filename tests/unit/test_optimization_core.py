"""Test the optimization core module."""

from pathlib import Path

import pandas as pd
import pulp
import pytest

from fleetmix.config.parameters import Parameters
from fleetmix.core_types import VehicleConfiguration
from fleetmix.optimization.core import (
    _calculate_cluster_cost,
    _calculate_solution_statistics,
    _create_model,
    _extract_solution,
    _validate_solution,
    solve_fsm_problem,
)


@pytest.fixture
def simple_clusters_df():
    """Create simple clusters DataFrame for testing."""
    return pd.DataFrame(
        {
            "Cluster_ID": ["C1", "C2"],
            "Customers": [["Cust1", "Cust2"], ["Cust3"]],
            "Config_ID": ["V1", "V1"],
            "Total_Demand": [
                {"Dry": 20, "Chilled": 0, "Frozen": 0},
                {"Dry": 15, "Chilled": 5, "Frozen": 0},
            ],
            "Route_Time": [2.5, 1.8],
            "Centroid_Latitude": [0.1, 0.2],
            "Centroid_Longitude": [0.1, 0.2],
            "Method": ["minibatch_kmeans", "minibatch_kmeans"],
        }
    )


@pytest.fixture
def simple_configs_df():
    """Create simple vehicle configurations DataFrame."""
    return pd.DataFrame(
        {
            "Config_ID": ["V1", "V2"],
            "Vehicle_Type": ["Small", "Large"],
            "Capacity": [50, 100],
            "Fixed_Cost": [100, 200],
            "Dry": [1, 1],
            "Chilled": [0, 1],
            "Frozen": [0, 1],
        }
    )


@pytest.fixture
def simple_customers_df():
    """Create simple customers DataFrame."""
    return pd.DataFrame(
        {
            "Customer_ID": ["Cust1", "Cust2", "Cust3"],
            "Customer_Name": ["Customer 1", "Customer 2", "Customer 3"],
            "Latitude": [0.05, 0.15, 0.2],
            "Longitude": [0.05, 0.15, 0.2],
            "Dry_Demand": [10, 10, 15],
            "Chilled_Demand": [0, 0, 5],
            "Frozen_Demand": [0, 0, 0],
        }
    )


@pytest.fixture
def simple_params():
    """Create simple parameters for testing."""
    config_path = (
        Path(__file__).parent.parent / "_assets" / "configs" / "base_test_config.yaml"
    )
    return Parameters.from_yaml(str(config_path))


@pytest.fixture
def params_with_post_opt():
    """Create parameters with post-optimization enabled."""
    config_path = (
        Path(__file__).parent.parent
        / "_assets"
        / "configs"
        / "test_config_post_opt.yaml"
    )
    return Parameters.from_yaml(str(config_path))


def dataframe_to_configurations(df: pd.DataFrame) -> list[VehicleConfiguration]:
    """Convert DataFrame to List[VehicleConfiguration] for testing."""
    configs = []
    for _, row in df.iterrows():
        # Determine compartments based on goods columns
        compartments = {}
        goods_cols = ["Dry", "Chilled", "Frozen"]
        for good in goods_cols:
            if good in row:
                compartments[good] = bool(row[good])

        config = VehicleConfiguration(
            config_id=row["Config_ID"],
            vehicle_type=row["Vehicle_Type"],
            capacity=row["Capacity"],
            fixed_cost=row["Fixed_Cost"],
            compartments=compartments,
        )
        configs.append(config)
    return configs


def test_solve_fsm_problem_basic(
    simple_clusters_df, simple_configs_df, simple_customers_df, simple_params
):
    """Test basic FSM problem solving."""
    configurations = dataframe_to_configurations(simple_configs_df)
    # Convert DataFrames to lists for new API
    from fleetmix.core_types import Cluster, Customer

    clusters_list = Cluster.from_dataframe(simple_clusters_df)
    customers_list = Customer.from_dataframe(simple_customers_df)

    result = solve_fsm_problem(
        clusters=clusters_list,
        configurations=configurations,
        customers=customers_list,
        parameters=simple_params,
        verbose=False,
    )

    # Validate result structure
    assert hasattr(result, "total_cost")
    assert hasattr(result, "selected_clusters")
    assert hasattr(result, "vehicles_used")
    assert hasattr(result, "total_vehicles")

    # Check that result is reasonable
    assert result.total_cost > 0
    assert not result.selected_clusters.empty
    assert result.total_vehicles > 0


def test_solve_fsm_problem_with_post_optimization(
    simple_clusters_df, simple_configs_df, simple_customers_df, params_with_post_opt
):
    """Test FSM problem solving with post-optimization enabled."""
    configurations = dataframe_to_configurations(simple_configs_df)
    # Convert DataFrames to lists for new API
    from fleetmix.core_types import Cluster, Customer

    clusters_list = Cluster.from_dataframe(simple_clusters_df)
    customers_list = Customer.from_dataframe(simple_customers_df)

    result = solve_fsm_problem(
        clusters=clusters_list,
        configurations=configurations,
        customers=customers_list,
        parameters=params_with_post_opt,
        verbose=False,
    )

    # Validate result structure
    assert hasattr(result, "total_cost")
    assert hasattr(result, "selected_clusters")
    assert hasattr(result, "post_optimization_runtime_sec")


def test_create_model(simple_clusters_df, simple_configs_df, simple_params):
    """Test model creation."""
    configurations = dataframe_to_configurations(simple_configs_df)
    model, y_vars, x_vars, c_vk = _create_model(
        clusters_df=simple_clusters_df,
        configurations=configurations,
        parameters=simple_params,
    )

    # Check that model was created
    assert isinstance(model, pulp.LpProblem)
    assert len(y_vars) > 0
    assert len(x_vars) > 0
    assert len(c_vk) > 0


def test_extract_solution(simple_clusters_df):
    """Test solution extraction."""
    # Create mock variables
    y_vars = {
        "C1": type("MockVar", (), {"varValue": 1})(),
        "C2": type("MockVar", (), {"varValue": 0})(),
    }

    x_vars = {
        ("V1", "C1"): type("MockVar", (), {"varValue": 1})(),
        ("V1", "C2"): type("MockVar", (), {"varValue": 0})(),
    }

    selected_clusters = _extract_solution(simple_clusters_df, y_vars, x_vars)

    # Check that only selected clusters are returned
    assert len(selected_clusters) == 1
    assert selected_clusters.iloc[0]["Cluster_ID"] == "C1"


def test_validate_solution(simple_customers_df, simple_configs_df, simple_params):
    """Test solution validation."""
    # Create selected clusters that miss one customer
    selected_clusters = pd.DataFrame(
        {
            "Cluster_ID": ["C1"],
            "Customers": [["Cust1", "Cust2"]],  # Missing Cust3
            "Config_ID": ["V1"],
        }
    )

    missing_customers = _validate_solution(
        selected_clusters, simple_customers_df, simple_configs_df, simple_params
    )

    # Check that missing customer is detected
    assert len(missing_customers) == 1
    assert "Cust3" in missing_customers


def test_calculate_cluster_cost(simple_params):
    """Test cluster cost calculation."""
    cluster = pd.Series(
        {"Route_Time": 2.0, "Total_Demand": {"Dry": 20, "Chilled": 10, "Frozen": 0}}
    )

    config = VehicleConfiguration(
        config_id="V1",
        vehicle_type="Small",
        capacity=50,
        fixed_cost=100,
        compartments={"Dry": True, "Chilled": True, "Frozen": False},
    )

    cost = _calculate_cluster_cost(cluster, config, simple_params)

    # Cost should include fixed cost + variable cost + compartment cost
    # Fixed: 100, Variable: 2.0 * 20.0 = 40, Compartment: 5 * (2-1) = 5
    expected_cost = 100 + 40 + 5  # = 145
    assert cost == expected_cost


def test_solve_with_infeasible_clusters():
    """Test solving with clusters that have no feasible vehicles."""
    # Create clusters with very high demand
    clusters_df = pd.DataFrame(
        {
            "Cluster_ID": ["C1"],
            "Customers": [["Cust1"]],
            "Config_ID": ["V1"],
            "Total_Demand": [
                {"Dry": 1000, "Chilled": 0, "Frozen": 0}
            ],  # Exceeds all capacities
            "Route_Time": [1.0],
            "Method": ["minibatch_kmeans"],
        }
    )

    configs_df = pd.DataFrame(
        {
            "Config_ID": ["V1"],
            "Vehicle_Type": ["Small"],
            "Capacity": [50],  # Too small for demand
            "Fixed_Cost": [100],
            "Dry": [1],
            "Chilled": [0],
            "Frozen": [0],
        }
    )

    customers_df = pd.DataFrame(
        {
            "Customer_ID": ["Cust1"],
            "Customer_Name": ["Customer 1"],
            "Dry_Demand": [1000],
            "Chilled_Demand": [0],
            "Frozen_Demand": [0],
        }
    )

    # Load minimal config
    config_path = (
        Path(__file__).parent.parent
        / "_assets"
        / "configs"
        / "test_config_minimal.yaml"
    )
    params = Parameters.from_yaml(str(config_path))

    configurations = dataframe_to_configurations(configs_df)

    # This should exit with an error
    # Convert DataFrames to lists for new API
    from fleetmix.core_types import Cluster, Customer

    clusters_list = Cluster.from_dataframe(clusters_df)
    customers_list = Customer.from_dataframe(customers_df)

    with pytest.raises(RuntimeError, match="Optimization failed with status: Not Solved"):
        solve_fsm_problem(
            clusters=clusters_list,
            configurations=configurations,
            customers=customers_list,
            parameters=params,
            verbose=False,
        )


def test_calculate_solution_statistics(
    simple_clusters_df, simple_configs_df, simple_params
):
    """Test solution statistics calculation."""
    # Create a mock model and variables
    model = type(
        "MockModel",
        (),
        {"objective": type("MockObjective", (), {"value": lambda: 300})()},
    )()

    x_vars = {
        ("V1", "C1"): type("MockVar", (), {"varValue": 1})(),
        ("V1", "C2"): type("MockVar", (), {"varValue": 1})(),
    }

    c_vk = {("V1", "C1"): 150, ("V1", "C2"): 150}

    # Merge configs into clusters
    selected_clusters = simple_clusters_df.copy()
    selected_clusters["Dry"] = 1
    selected_clusters["Chilled"] = 0
    selected_clusters["Frozen"] = 0

    configurations = dataframe_to_configurations(simple_configs_df)

    stats = _calculate_solution_statistics(
        selected_clusters, configurations, simple_params, model, x_vars, c_vk
    )

    # Validate statistics
    assert hasattr(stats, "total_cost")
    assert hasattr(stats, "total_fixed_cost")
    assert hasattr(stats, "total_variable_cost")
    assert stats.total_cost > 0
