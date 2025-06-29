import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from fleetmix.benchmarking.solvers.vrp_solver import VRPSolver
from fleetmix.config.parameters import Parameters


def _load_minimal_params() -> Parameters:
    """Helper that loads the minimal YAML shipped with tests assets."""
    cfg_path = (
        Path(__file__).resolve().parent.parent / "_assets" / "configs" / "test_config_minimal.yaml"
    )
    return Parameters.from_yaml(cfg_path)


def _make_customers() -> pd.DataFrame:
    """Two customers positioned ~15 km apart with positive demand for exactly one good."""
    return pd.DataFrame(
        {
            "Latitude": [0.05, 0.135],  # >0 difference from depot for both
            "Longitude": [0.0, 0.0],
            "Dry_Demand": [5, 5],
            "Chilled_Demand": [0, 0],
            "Frozen_Demand": [0, 0],
        }
    )


@pytest.fixture
def solver(monkeypatch):
    """Return a *lightweight* VRPSolver with heavy PyVRP machinery skipped."""
    # Patch the heavy model-building to keep tests fast and independent of PyVRP internals
    monkeypatch.setattr(VRPSolver, "_prepare_model", lambda self: None)
    params = _load_minimal_params()
    customers = _make_customers()
    return VRPSolver(customers=customers, params=params, time_limit=1)


def test_distance_matrix_symmetry(solver):
    # Act
    dm = solver._calculate_distance_matrix(2)

    # Assert basic structure
    assert dm.shape == (3, 3)  # 2 clients + depot
    # Zero diagonal
    assert np.allclose(np.diag(dm), 0.0)
    # Symmetric property
    assert np.allclose(dm, dm.T)
    # Non-zero distance from depot to first client and second client
    assert dm[0, 1] > 0 and dm[0, 2] > 0
    # Distances positive and sensible (< 50 km for this toy set)
    assert dm[0, 1] < 50 and dm[0, 2] < 50 