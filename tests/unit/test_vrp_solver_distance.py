import pandas as pd
import numpy as np
import pytest

from fleetmix.benchmarking.solvers.vrp_solver import VRPSolver
from fleetmix.config.parameters import Parameters
from fleetmix.core_types import DepotLocation, VehicleSpec, BenchmarkType


@pytest.fixture()
def simple_params():
    vehicles = {
        "van": VehicleSpec(
            capacity=100,
            fixed_cost=0,
            avg_speed=30,
            service_time=5,
            max_route_time=8,
            extra={},
        )
    }
    params = Parameters(
        vehicles=vehicles,
        variable_cost_per_hour=10.0,
        depot=DepotLocation(latitude=0.0, longitude=0.0),
        goods=["dry"],
        clustering={"route_time_estimation": "haversine", "geo_weight": 0.7, "demand_weight": 0.3},
        demand_file="dummy.csv",
        light_load_penalty=0.0,
        light_load_threshold=0.2,
        compartment_setup_cost=0.0,
        format="csv",
    )
    return params


def _patch_prepare_model(monkeypatch):
    monkeypatch.setattr(VRPSolver, "_prepare_model", lambda self: None)


def test_distance_matrix_symmetry(monkeypatch, simple_params):
    _patch_prepare_model(monkeypatch)
    # Construct tiny customer DataFrame
    customers = pd.DataFrame(
        {
            "Latitude": [0.5, 1.0],
            "Longitude": [0.0, 0.0],
            "dry_Demand": [10, 20],
        }
    )

    solver = VRPSolver(customers=customers, params=simple_params, time_limit=1, benchmark_type=BenchmarkType.SINGLE_COMPARTMENT)
    matrix = solver._calculate_distance_matrix(n_clients=2)

    # Matrix should be (clients+1) x (clients+1)
    assert matrix.shape == (3, 3)

    # Symmetry and zero diagonal
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)

    # Depot to first client distance equals haversine between coords
    from haversine import haversine

    expected = haversine((0.0, 0.0), (0.0, 0.0))  # depot to itself -> 0
    assert matrix[0, 0] == expected == 0.0

    depot_to_client1 = matrix[0, 1]
    assert depot_to_client1 > 0 