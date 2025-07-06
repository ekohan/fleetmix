import logging

import pandas as pd
import pulp

from fleetmix.core_types import VehicleConfiguration
from fleetmix.optimization.core import _create_model, _extract_solution, _validate_solution


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


def test_create_model_constraints(toy_fsm_edge_data):
    """Test that model has correct constraint structure."""
    clusters_df, config_df, params = toy_fsm_edge_data
    
    # Create a simple customers_df for the test
    customers_df = pd.DataFrame({
        "Customer_ID": ["C1", "C2"],
        "Dry_Demand": [1, 1],
        "Chilled_Demand": [0, 0],
        "Frozen_Demand": [0, 0],
    })
    
    configurations = dataframe_to_configurations(config_df)
    model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations, customers_df, params)
    # Each customer coverage constraint exists
    for cid in ["C1", "C2"]:
        cname = f"Customer_Coverage_{cid}"
        assert cname in model.constraints
    # There is exactly one x_var for (1,1)
    assert (1, 1) in x_vars


def test_light_load_threshold_monotonicity(toy_fsm_edge_data):
    """Test that light load penalty is applied correctly based on threshold."""
    clusters_df, config_df, params = toy_fsm_edge_data
    
    # Create a simple customers_df for the test
    customers_df = pd.DataFrame({
        "Customer_ID": ["C1", "C2"],
        "Dry_Demand": [1, 1],
        "Chilled_Demand": [0, 0],
        "Frozen_Demand": [0, 0],
    })
    
    configurations = dataframe_to_configurations(config_df)

    # Test with light load penalty enabled
    from fleetmix.config.params import ProblemParams
    base_problem = params.problem
    
    costs = []
    for thr in [0.0, 0.5, 0.9]:
        new_problem = ProblemParams(
            vehicles=base_problem.vehicles,
            depot=base_problem.depot,
            goods=base_problem.goods,
            variable_cost_per_hour=base_problem.variable_cost_per_hour,
            light_load_penalty=100,  # Set penalty to 100
            light_load_threshold=thr,  # Vary threshold
            compartment_setup_cost=base_problem.compartment_setup_cost,
            allow_split_stops=base_problem.allow_split_stops,
        )
        params.problem = new_problem
        model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations, customers_df, params)
        costs.append(c_vk[(1, 1)])
    # Objective cost non-decreasing as threshold increases
    assert costs[0] <= costs[1] <= costs[2]


def test_capacity_infeasibility_injects_NoVehicle(toy_fsm_edge_data, caplog):
    """Test that NoVehicle is injected when cluster exceeds all vehicle capacities."""
    clusters_df, config_df, params = toy_fsm_edge_data
    
    # Create a simple customers_df for the test
    customers_df = pd.DataFrame({
        "Customer_ID": ["C1", "C2"],
        "Dry_Demand": [1, 1],
        "Chilled_Demand": [0, 0],
        "Frozen_Demand": [0, 0],
    })
    
    configurations = dataframe_to_configurations(config_df)
    # Make cluster demand exceed vehicle capacity
    clusters_df.at[0, "Total_Demand"] = {"Dry": 10, "Chilled": 0, "Frozen": 0}
    # Set caplog to capture DEBUG messages from the specific logger
    caplog.set_level(logging.DEBUG, logger="fleetmix.optimization.core")
    model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations, customers_df, params)
    # 'NoVehicle' var should be present and y_1==0 forced
    assert any(v == "NoVehicle" for (v, k) in x_vars)
    # There should be an unserviceable-cluster constraint
    assert "Unserviceable_Cluster_1" in model.constraints
    # Use record_tuples for more robust log checking
    assert any(
        rec[0] == "fleetmix.optimization.core"  # Check specific logger name
        and rec[1] == logging.DEBUG  # Expect DEBUG level
        and "cannot be served by any vehicle" in rec[2].lower()  # More specific message
        for rec in caplog.record_tuples
    ), "Expected debug message about unserviceable cluster"


def test_extract_and_validate_solution(toy_fsm_edge_data):
    clusters_df, config_df, params = toy_fsm_edge_data
    # Explicitly disable split-stops to ensure consistent behavior
    from fleetmix.config.params import ProblemParams
    new_problem = ProblemParams(
        vehicles=params.problem.vehicles,
        depot=params.problem.depot,
        goods=params.problem.goods,
        variable_cost_per_hour=params.problem.variable_cost_per_hour,
        light_load_penalty=params.problem.light_load_penalty,
        light_load_threshold=params.problem.light_load_threshold,
        compartment_setup_cost=params.problem.compartment_setup_cost,
        allow_split_stops=False,
    )
    params.problem = new_problem

    configurations = dataframe_to_configurations(config_df)
    # Build y_vars: cluster 1 selected
    y = pulp.LpVariable("y_1", cat="Binary")
    y.varValue = 1
    y_vars = {1: y}
    # Build x_vars: assign config 1 to cluster 1
    x = pulp.LpVariable("x_1_1", cat="Binary")
    x.varValue = 1
    x_vars = {(1, 1): x}
    selected = _extract_solution(clusters_df, y_vars, x_vars)
    # The selected DataFrame should have Config_ID=1
    assert list(selected["Config_ID"]) == [1]
    # Validate solution: no missing customers
    customers_df = pd.DataFrame(
        [
            {
                "Customer_ID": "C1",
                "Dry_Demand": 0,
                "Chilled_Demand": 0,
                "Frozen_Demand": 0,
            },
            {
                "Customer_ID": "C2",
                "Dry_Demand": 0,
                "Chilled_Demand": 0,
                "Frozen_Demand": 0,
            },
        ]
    )
    missing = _validate_solution(selected, customers_df, configurations, params)
    assert missing == set()
