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
    clusters_df, config_df, params = toy_fsm_edge_data
    # Explicitly disable split-stops to ensure consistent constraint naming
    params.allow_split_stops = False

    configurations = dataframe_to_configurations(config_df)
    model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations, params)
    # Each customer coverage constraint exists
    for cid in ["C1", "C2"]:
        cname = f"Customer_Coverage_{cid}"
        assert cname in model.constraints
    # There is exactly one x_var for (1,1)
    assert (1, 1) in x_vars


def test_light_load_threshold_monotonicity(toy_fsm_edge_data):
    clusters_df, config_df, params = toy_fsm_edge_data
    # Explicitly disable split-stops to ensure consistent behavior
    params.allow_split_stops = False

    configurations = dataframe_to_configurations(config_df)
    # small cluster demand -> light-load penalty applies
    params.light_load_penalty = 100
    costs = []
    for thr in [0.0, 0.5, 0.9]:
        params.light_load_threshold = thr
        model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations, params)
        costs.append(c_vk[(1, 1)])
    # Objective cost non-decreasing as threshold increases
    assert costs[0] <= costs[1] <= costs[2]


def test_capacity_infeasibility_injects_NoVehicle(toy_fsm_edge_data, caplog):
    clusters_df, config_df, params = toy_fsm_edge_data
    # Explicitly disable split-stops to ensure consistent behavior
    params.allow_split_stops = False

    configurations = dataframe_to_configurations(config_df)
    # Make demand exceed capacity
    clusters_df.at[0, "Total_Demand"] = {"Dry": 100, "Chilled": 0, "Frozen": 0}
    # Set caplog to capture DEBUG messages from the specific logger
    caplog.set_level(logging.DEBUG, logger="fleetmix.optimization.core")
    model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations, params)
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
    params.allow_split_stops = False

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
