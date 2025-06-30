import logging

import pandas as pd
import pytest

from fleetmix.optimization.core import _create_model, _extract_solution, _solve_internal
from fleetmix.core_types import VehicleConfiguration


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
            vehicle_type=row.get("Vehicle_Type", "Test"),
            capacity=row["Capacity"],
            fixed_cost=row["Fixed_Cost"],
            compartments=compartments,
        )
        configs.append(config)
    return configs


def test_create_model_counts_traditional_mode(toy_fsm_core_data):
    """Test model creation in traditional mode (split-stops disabled)."""
    clusters_df, config_df, customers_df, params = toy_fsm_core_data
    # Explicitly disable split-stops to ensure consistent constraint naming
    params.allow_split_stops = False

    configurations = dataframe_to_configurations(config_df)
    model, y_vars, x_vars, c_vk = _create_model(
        clusters_df, configurations, customers_df, params
    )
    # Exactly one cluster variable and one assignment x-var
    assert len(y_vars) == 1, "Should create one y var"
    assert len(x_vars) == 1, "Should create one x var per vehicle-cluster"
    # Constraint names include customer coverage and vehicle assignment
    cons_names = list(model.constraints.keys())
    assert any("Customer_Coverage_" in name for name in cons_names), (
        f"Expected Customer_Coverage_ constraint, got: {cons_names}"
    )
    assert any("Vehicle_Assignment_" in name for name in cons_names), (
        f"Expected Vehicle_Assignment_ constraint, got: {cons_names}"
    )


def test_create_model_counts_split_stop_mode(toy_fsm_core_data):
    """Test model creation in split-stop mode (split-stops enabled)."""
    clusters_df, config_df, customers_df, params = toy_fsm_core_data
    # Explicitly enable split-stops to test the split-stop constraint naming
    params.allow_split_stops = True

    # Modify the test data to use pseudo-customer IDs for split-stop mode
    # This simulates what happens when customers are exploded into pseudo-customers
    clusters_df.at[0, "Customers"] = [
        "C1::dry",
        "C1::chilled",
    ]  # Two pseudo-customers for same physical customer
    clusters_df.at[0, "Total_Demand"] = {"Dry": 1, "Chilled": 1, "Frozen": 0}
    config_df.at[0, "Chilled"] = 1  # Enable chilled compartment
    
    # Update customers_df to include the pseudo-customers
    customers_df = pd.DataFrame({
        "Customer_ID": ["C1::dry", "C1::chilled"],
        "Dry_Demand": [1, 0],
        "Chilled_Demand": [0, 1],
        "Frozen_Demand": [0, 0],
        "Latitude": [0.0, 0.0],
        "Longitude": [0.0, 0.0]
    })

    configurations = dataframe_to_configurations(config_df)
    model, y_vars, x_vars, c_vk = _create_model(
        clusters_df, configurations, customers_df, params
    )
    # Exactly one cluster variable and one assignment x-var
    assert len(y_vars) == 1, "Should create one y var"
    assert len(x_vars) == 1, "Should create one x var per vehicle-cluster"
    # Constraint names should include coverage per good and vehicle assignment
    cons_names = list(model.constraints.keys())
    # In split-stop mode with pseudo-customers, we should have Cover_ constraints
    assert any("Cover_" in name for name in cons_names), (
        f"Expected Cover_ constraint in split-stop mode, got: {cons_names}"
    )
    assert any("Vehicle_Assignment_" in name for name in cons_names), (
        f"Expected Vehicle_Assignment_ constraint, got: {cons_names}"
    )
    # Should have coverage constraints for both goods
    assert any("Cover_C1_dry" in name for name in cons_names), (
        f"Expected Cover_C1_dry constraint, got: {cons_names}"
    )
    assert any("Cover_C1_chilled" in name for name in cons_names), (
        f"Expected Cover_C1_chilled constraint, got: {cons_names}"
    )


def test_create_model_counts(toy_fsm_core_data):
    """Legacy test name - defaults to traditional mode for backward compatibility."""
    clusters_df, config_df, customers_df, params = toy_fsm_core_data
    # Explicitly disable split-stops to ensure consistent constraint naming
    params.allow_split_stops = False

    configurations = dataframe_to_configurations(config_df)
    model, y_vars, x_vars, c_vk = _create_model(
        clusters_df, configurations, customers_df, params
    )
    # Exactly one cluster variable and one assignment x-var
    assert len(y_vars) == 1, "Should create one y var"
    assert len(x_vars) == 1, "Should create one x var per vehicle-cluster"
    # Constraint names include customer coverage and vehicle assignment
    cons_names = list(model.constraints.keys())
    assert any("Customer_Coverage_" in name for name in cons_names)
    assert any("Vehicle_Assignment_" in name for name in cons_names)


def test_extract_solution():
    import pulp

    # Build clusters DataFrame
    clusters_df = pd.DataFrame({"Cluster_ID": [1, 2], "Customers": [["C1"], ["C2"]]})
    # Create y-vars: only cluster 1 selected
    y1 = pulp.LpVariable("y_1", cat="Binary")
    y2 = pulp.LpVariable("y_2", cat="Binary")
    y1.varValue = 1
    y2.varValue = 0
    y_vars = {1: y1, 2: y2}
    # Create x-vars: assign vehicle 10 to cluster 1, vehicle 20 to cluster 2
    xA1 = pulp.LpVariable("x_10_1", cat="Binary")
    xB2 = pulp.LpVariable("x_20_2", cat="Binary")
    xA1.varValue = 1
    xB2.varValue = 1
    x_vars = {(10, 1): xA1, (20, 2): xB2}
    selected = _extract_solution(clusters_df, y_vars, x_vars)
    # Only cluster 1 should be selected, with Config_ID mapped to 10
    assert list(selected["Cluster_ID"]) == [1]
    assert list(selected["Config_ID"]) == [10]


def test_capacity_violation_model_warning_traditional_mode(toy_fsm_core_data, caplog):
    """Test capacity violation handling in traditional mode."""
    clusters_df, config_df, customers_df, params = toy_fsm_core_data
    # Explicitly disable split-stops to ensure consistent behavior
    params.allow_split_stops = False

    configurations = dataframe_to_configurations(config_df)
    # Build base data and violate capacity so no config is feasible
    clusters_df.at[0, "Total_Demand"] = {"Dry": 100, "Chilled": 0, "Frozen": 0}
    # Capture warnings/debug from model construction for the specific logger
    caplog.set_level(logging.DEBUG, logger="fleetmix.optimization.core")
    # Create model
    model, y_vars, x_vars, c_vk = _create_model(
        clusters_df, configurations, customers_df, params
    )
    # Assert that 'NoVehicle' variable was injected for unserviceable cluster
    assert any(v == "NoVehicle" for v, k in x_vars.keys()), (
        "Should inject NoVehicle for infeasible cluster"
    )
    # Check warning about unserviceable cluster (now expecting DEBUG level)
    assert any(
        rec[0] == "fleetmix.optimization.core"
        and rec[1] == logging.DEBUG  # Changed to DEBUG
        and "cannot be served by any vehicle"
        in rec[2].lower()  # Made message check more specific
        for rec in caplog.record_tuples
    ), "Expected debug message about unserviceable cluster"

    # Use pytest's raises to catch the expected error for infeasible status
    # Different solvers report infeasibility differently:
    # - CBC reports "Infeasible" (raises ValueError)
    # - Gurobi reports "Not Solved" (raises RuntimeError)
    with pytest.raises((ValueError, RuntimeError), match=r"Optimization failed with status: (Infeasible|Not Solved)"):
        _solve_internal(clusters_df, configurations, customers_df, params)


def test_capacity_violation_model_warning_split_stop_mode(toy_fsm_core_data, caplog):
    """Test capacity violation handling in split-stop mode."""
    clusters_df, config_df, customers_df, params = toy_fsm_core_data
    # Explicitly enable split-stops to test split-stop behavior
    params.allow_split_stops = True

    configurations = dataframe_to_configurations(config_df)
    # Build base data and violate capacity so no config is feasible
    clusters_df.at[0, "Total_Demand"] = {"Dry": 100, "Chilled": 0, "Frozen": 0}
    # Capture warnings/debug from model construction for the specific logger
    caplog.set_level(logging.DEBUG, logger="fleetmix.optimization.core")
    # Create model
    model, y_vars, x_vars, c_vk = _create_model(
        clusters_df, configurations, customers_df, params
    )
    # Assert that 'NoVehicle' variable was injected for unserviceable cluster
    assert any(v == "NoVehicle" for v, k in x_vars.keys()), (
        "Should inject NoVehicle for infeasible cluster"
    )
    # Check warning about unserviceable cluster (now expecting DEBUG level)
    assert any(
        rec[0] == "fleetmix.optimization.core"
        and rec[1] == logging.DEBUG  # Changed to DEBUG
        and "cannot be served by any vehicle"
        in rec[2].lower()  # Made message check more specific
        for rec in caplog.record_tuples
    ), "Expected debug message about unserviceable cluster"

    # In split-stop mode, the optimization should succeed but return an empty solution
    # rather than raising an exception (this is the correct behavior)
    solution = _solve_internal(
        clusters_df, configurations, customers_df, params
    )

    # Verify that the solution has no selected clusters (empty solution)
    assert len(solution.selected_clusters) == 0, (
        "Should have no selected clusters for infeasible problem"
    )
    assert solution.total_vehicles == 0, "Should have no vehicles used"
    assert solution.missing_customers == set(), (
        "Split-stop mode skips customer validation"
    )


def test_capacity_violation_model_warning(toy_fsm_core_data, caplog):
    """Legacy test name - defaults to traditional mode for backward compatibility."""
    clusters_df, config_df, customers_df, params = toy_fsm_core_data
    # Explicitly disable split-stops to ensure consistent behavior
    params.allow_split_stops = False

    configurations = dataframe_to_configurations(config_df)
    # Build base data and violate capacity so no config is feasible
    clusters_df.at[0, "Total_Demand"] = {"Dry": 100, "Chilled": 0, "Frozen": 0}
    # Capture warnings/debug from model construction for the specific logger
    caplog.set_level(logging.DEBUG, logger="fleetmix.optimization.core")
    # Create model
    model, y_vars, x_vars, c_vk = _create_model(
        clusters_df, configurations, customers_df, params
    )
    # Assert that 'NoVehicle' variable was injected for unserviceable cluster
    assert any(v == "NoVehicle" for v, k in x_vars.keys()), (
        "Should inject NoVehicle for infeasible cluster"
    )
    # Check warning about unserviceable cluster (now expecting DEBUG level)
    assert any(
        rec[0] == "fleetmix.optimization.core"
        and rec[1] == logging.DEBUG  # Changed to DEBUG
        and "cannot be served by any vehicle"
        in rec[2].lower()  # Made message check more specific
        for rec in caplog.record_tuples
    ), "Expected debug message about unserviceable cluster"

    # Use pytest's raises to catch the expected error for infeasible status
    # Different solvers report infeasibility differently:
    # - CBC reports "Infeasible" (raises ValueError)
    # - Gurobi reports "Not Solved" (raises RuntimeError)
    with pytest.raises((ValueError, RuntimeError), match=r"Optimization failed with status: (Infeasible|Not Solved)"):
        _solve_internal(clusters_df, configurations, customers_df, params)

    # Check stdout for the infeasible message
    # We don't need to check logs as the warning is printed to stdout, not logged
    # The test is successful if we reach this point (SystemExit was raised with code 1)
