"""Test the optimization core module."""

import pytest
import pandas as pd
import pulp
from pathlib import Path

from fleetmix.optimization.core import (
    solve_fsm_problem,
    _create_model,
    _extract_solution,
    _validate_solution,
    _calculate_solution_statistics,
    _calculate_cluster_cost
)
from fleetmix.config.parameters import Parameters


@pytest.fixture
def simple_clusters_df():
    """Create simple clusters DataFrame for testing."""
    return pd.DataFrame({
        'Cluster_ID': ['C1', 'C2'],
        'Customers': [['Cust1', 'Cust2'], ['Cust3']],
        'Config_ID': ['V1', 'V1'],
        'Total_Demand': [{'Dry': 20, 'Chilled': 0, 'Frozen': 0}, 
                        {'Dry': 15, 'Chilled': 5, 'Frozen': 0}],
        'Route_Time': [2.5, 1.8],
        'Centroid_Latitude': [0.1, 0.2],
        'Centroid_Longitude': [0.1, 0.2],
        'Method': ['minibatch_kmeans', 'minibatch_kmeans']
    })


@pytest.fixture
def simple_configs_df():
    """Create simple vehicle configurations DataFrame."""
    return pd.DataFrame({
        'Config_ID': ['V1', 'V2'],
        'Vehicle_Type': ['Small', 'Large'],
        'Capacity': [50, 100],
        'Fixed_Cost': [100, 200],
        'Dry': [1, 1],
        'Chilled': [0, 1],
        'Frozen': [0, 1]
    })


@pytest.fixture
def simple_customers_df():
    """Create simple customers DataFrame."""
    return pd.DataFrame({
        'Customer_ID': ['Cust1', 'Cust2', 'Cust3'],
        'Customer_Name': ['Customer 1', 'Customer 2', 'Customer 3'],
        'Latitude': [0.05, 0.15, 0.2],
        'Longitude': [0.05, 0.15, 0.2],
        'Dry_Demand': [10, 10, 15],
        'Chilled_Demand': [0, 0, 5],
        'Frozen_Demand': [0, 0, 0]
    })


@pytest.fixture
def simple_params():
    """Create simple parameters for testing."""
    config_path = Path(__file__).parent.parent / "_assets" / "configs" / "base_test_config.yaml"
    return Parameters.from_yaml(str(config_path))


@pytest.fixture
def params_with_post_opt():
    """Create parameters with post-optimization enabled."""
    config_path = Path(__file__).parent.parent / "_assets" / "configs" / "test_config_post_opt.yaml"
    return Parameters.from_yaml(str(config_path))


def test_solve_fsm_problem_basic(simple_clusters_df, simple_configs_df, simple_customers_df, simple_params):
    """Test basic FSM problem solving."""
    result = solve_fsm_problem(
        clusters_df=simple_clusters_df,
        configurations_df=simple_configs_df,
        customers_df=simple_customers_df,
        parameters=simple_params,
        verbose=False
    )
    
    # Check result structure
    assert result.total_cost is not None
    assert result.total_fixed_cost is not None
    assert result.total_variable_cost is not None
    assert result.total_penalties is not None
    assert result.selected_clusters is not None
    assert result.vehicles_used is not None
    assert result.missing_customers is not None
    assert result.solver_status is not None
    assert result.solver_runtime_sec is not None
    
    # Check that solution is optimal
    assert result.solver_status == 'Optimal'


def test_solve_fsm_problem_with_post_optimization(simple_clusters_df, simple_configs_df, simple_customers_df, params_with_post_opt):
    """Test FSM problem solving with post-optimization enabled."""
    result = solve_fsm_problem(
        clusters_df=simple_clusters_df,
        configurations_df=simple_configs_df,
        customers_df=simple_customers_df,
        parameters=params_with_post_opt,
        verbose=False
    )
    
    # Check that post-optimization runtime is recorded
    assert result.post_optimization_runtime_sec is not None


def test_create_model(simple_clusters_df, simple_configs_df, simple_params):
    """Test model creation."""
    model, y_vars, x_vars, c_vk = _create_model(
        clusters_df=simple_clusters_df,
        configurations_df=simple_configs_df,
        parameters=simple_params
    )
    
    # Check model type
    assert isinstance(model, pulp.LpProblem)
    
    # Check that variables were created
    assert len(y_vars) > 0
    assert len(x_vars) > 0
    assert len(c_vk) > 0
    
    # Check that constraints were added
    assert len(model.constraints) > 0


def test_extract_solution(simple_clusters_df):
    """Test solution extraction."""
    # Create mock variables
    y_vars = {
        'C1': type('MockVar', (), {'varValue': 1})(),
        'C2': type('MockVar', (), {'varValue': 0})()
    }
    
    x_vars = {
        ('V1', 'C1'): type('MockVar', (), {'varValue': 1})(),
        ('V1', 'C2'): type('MockVar', (), {'varValue': 0})()
    }
    
    selected_clusters = _extract_solution(simple_clusters_df, y_vars, x_vars)
    
    # Check that only selected clusters are returned
    assert len(selected_clusters) == 1
    assert selected_clusters.iloc[0]['Cluster_ID'] == 'C1'


def test_validate_solution(simple_customers_df, simple_configs_df):
    """Test solution validation."""
    # Create selected clusters that miss one customer
    selected_clusters = pd.DataFrame({
        'Cluster_ID': ['C1'],
        'Customers': [['Cust1', 'Cust2']],  # Missing Cust3
        'Config_ID': ['V1']
    })
    
    missing_customers = _validate_solution(
        selected_clusters,
        simple_customers_df,
        simple_configs_df
    )
    
    # Check that missing customer is detected
    assert len(missing_customers) == 1
    assert 'Cust3' in missing_customers


def test_calculate_cluster_cost(simple_params):
    """Test cluster cost calculation."""
    cluster = pd.Series({
        'Route_Time': 2.0,
        'Total_Demand': {'Dry': 20, 'Chilled': 10, 'Frozen': 0}
    })
    
    config = pd.Series({
        'Fixed_Cost': 100,
        'Dry': 1,
        'Chilled': 1,
        'Frozen': 0
    })
    
    cost = _calculate_cluster_cost(cluster, config, simple_params)
    
    # Expected: Fixed(100) + Variable(2*20) + Compartment(10*(2-1)) = 150
    # Using actual params values: Fixed(100) + Variable(2*20) + Compartment(5*(2-1)) = 145
    assert cost == 145


def test_solve_with_infeasible_clusters():
    """Test solving with clusters that have no feasible vehicles."""
    # Create clusters with very high demand
    clusters_df = pd.DataFrame({
        'Cluster_ID': ['C1'],
        'Customers': [['Cust1']],
        'Config_ID': ['V1'],
        'Total_Demand': [{'Dry': 1000, 'Chilled': 0, 'Frozen': 0}],  # Exceeds all capacities
        'Route_Time': [1.0],
        'Method': ['minibatch_kmeans']
    })
    
    configs_df = pd.DataFrame({
        'Config_ID': ['V1'],
        'Vehicle_Type': ['Small'],
        'Capacity': [50],  # Too small for demand
        'Fixed_Cost': [100],
        'Dry': [1],
        'Chilled': [0],
        'Frozen': [0]
    })
    
    customers_df = pd.DataFrame({
        'Customer_ID': ['Cust1'],
        'Customer_Name': ['Customer 1'],
        'Dry_Demand': [1000],
        'Chilled_Demand': [0],
        'Frozen_Demand': [0]
    })
    
    # Load minimal config
    config_path = Path(__file__).parent.parent / "_assets" / "configs" / "test_config_minimal.yaml"
    params = Parameters.from_yaml(str(config_path))
    
    # This should exit with an error
    with pytest.raises(SystemExit):
        solve_fsm_problem(
            clusters_df=clusters_df,
            configurations_df=configs_df,
            customers_df=customers_df,
            parameters=params,
            verbose=False
        )


def test_calculate_solution_statistics(simple_clusters_df, simple_configs_df, simple_params):
    """Test solution statistics calculation."""
    # Create a mock model and variables
    model = type('MockModel', (), {'objective': type('MockObjective', (), {'value': lambda: 300})()})()
    
    x_vars = {
        ('V1', 'C1'): type('MockVar', (), {'varValue': 1})(),
        ('V1', 'C2'): type('MockVar', (), {'varValue': 1})()
    }
    
    c_vk = {
        ('V1', 'C1'): 150,
        ('V1', 'C2'): 150
    }
    
    # Merge configs into clusters
    selected_clusters = simple_clusters_df.copy()
    selected_clusters['Dry'] = 1
    selected_clusters['Chilled'] = 0
    selected_clusters['Frozen'] = 0
    
    stats = _calculate_solution_statistics(
        selected_clusters,
        simple_configs_df,
        simple_params,
        model,
        x_vars,
        c_vk
    )
    
    assert stats.total_fixed_cost is not None
    assert stats.total_variable_cost is not None
    assert stats.total_penalties is not None
    assert stats.vehicles_used is not None
    assert stats.total_vehicles is not None 