import pandas as pd
import pulp

from fleetmix.core_types import VehicleConfiguration
from fleetmix.optimization import _create_model


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


def test_create_model_basic(toy_fsm_model_build_data):
    """Test that _create_model creates appropriate variables and constraints."""
    clusters_df, config_df, params = toy_fsm_model_build_data
    
    # Create a simple customers_df for the test
    customers_df = pd.DataFrame({
        "Customer_ID": [1, 2],
        "Dry_Demand": [3, 2],
        "Chilled_Demand": [0, 0],
        "Frozen_Demand": [0, 0],
        "Latitude": [0.0, 0.0],
        "Longitude": [0.0, 0.0]
    })

    # Convert config DataFrame to list of VehicleConfiguration objects
    configurations = [
        VehicleConfiguration(
            config_id="v1", 
            vehicle_type="A", 
            capacity=10, 
            fixed_cost=100,
            compartments={"Dry": True, "Chilled": False, "Frozen": False}
        )
    ]

    model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations, customers_df, params)

    # Model should be a pulp problem
    assert isinstance(model, pulp.LpProblem)
    # y_vars contains our cluster
    assert "k1" in y_vars
    # x_vars contains decision for (v1,k1)
    assert ("v1", "k1") in x_vars
    # c_vk has a cost entry
    assert ("v1", "k1") in c_vk

    # Check that customer coverage constraints exist for both customers
    cons_names = list(model.constraints.keys())
    assert any("Customer_Coverage_1" in name for name in cons_names)
    assert any("Customer_Coverage_2" in name for name in cons_names)

    # Objective should include x_v1_k1
    obj_str = str(model.objective)
    assert "x_v1_k1" in obj_str
