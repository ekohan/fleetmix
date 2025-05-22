import pandas as pd
import pulp
import pytest

from fleetmix.optimization import _create_model
from fleetmix.config.parameters import Parameters


def test_create_model_basic(toy_fsm_model_build_data):
    # Use shared toy data fixture for model build tests
    clusters_df, configurations_df, params = toy_fsm_model_build_data

    model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations_df, params)

    # Model should be a pulp problem
    assert isinstance(model, pulp.LpProblem)
    # y_vars contains our cluster
    assert 'k1' in y_vars
    # x_vars contains decision for (v1,k1)
    assert ('v1', 'k1') in x_vars
    # c_vk has a cost entry
    assert ('v1', 'k1') in c_vk

    # Check that customer coverage constraints exist for both customers
    cons_names = list(model.constraints.keys())
    assert any('Customer_Coverage_1' in name for name in cons_names)
    assert any('Customer_Coverage_2' in name for name in cons_names)

    # Objective should include x_v1_k1
    obj_str = str(model.objective)
    assert 'x_v1_k1' in obj_str 