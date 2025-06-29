import pandas as pd
import numpy as np
import pytest

from fleetmix.benchmarking.solvers.vrp_solver import VRPSolver, BenchmarkType
from fleetmix.config.parameters import Parameters
from fleetmix.core_types import DepotLocation, VehicleSpec


@pytest.fixture()
def simple_params_two_goods():
    vehicles = {
        "truck": VehicleSpec(
            capacity=100,
            fixed_cost=0,
            avg_speed=30,
            service_time=5,
            max_route_time=8,
        )
    }
    return Parameters(
        vehicles=vehicles,
        variable_cost_per_hour=10.0,
        depot=DepotLocation(latitude=0.0, longitude=0.0),
        goods=["dry", "chilled"],
        clustering={"route_time_estimation": "haversine", "geo_weight": 0.7, "demand_weight": 0.3},
        demand_file="dummy.csv",
        light_load_penalty=0.0,
        light_load_threshold=0.2,
        compartment_setup_cost=0.0,
        format="csv",
    )


@pytest.fixture()
def solver(monkeypatch, simple_params_two_goods):
    # small customer DF
    df = pd.DataFrame(
        {
            "Latitude": [0.2, 0.3],
            "Longitude": [0.1, 0.2],
            "dry_Demand": [10, 0],
            "chilled_Demand": [0, 20],
        }
    )

    # Patch heavy _prepare_model to skip pyvrp
    monkeypatch.setattr(VRPSolver, "_prepare_model", lambda self: None)

    solver = VRPSolver(df, simple_params_two_goods, time_limit=1, benchmark_type=BenchmarkType.MULTI_COMPARTMENT)
    # populate original_demands
    solver._prepare_multi_compartment_data()
    return solver


def test_prepare_multi_compartment_data(solver):
    mc = solver._prepare_multi_compartment_data()
    # Total_Demand column should be sum of goods
    assert (mc["Total_Demand"] == [10, 20]).all()
    # original_demands mapping size matches customers
    assert len(solver.original_demands) == 2
    # Dry demand preserved
    first_id = list(solver.original_demands.keys())[0]
    assert solver.original_demands[first_id]["dry"] == 10


def test_determine_compartment_configuration(solver):
    # Prepare route with both customers
    route_customers = list(solver.original_demands.keys())
    compartments = solver._determine_compartment_configuration(route_customers, 0)
    # Should allocate 0.1 and 0.2 relative to vehicle capacity 100
    assert pytest.approx(compartments["dry"], rel=1e-3) == 0.1
    assert pytest.approx(compartments["chilled"], rel=1e-3) == 0.2


def test_print_solution_branches(monkeypatch, simple_params_two_goods):
    # minimal solver with SINGLE_COMPARTMENT to exercise vehicle breakdown
    df = pd.DataFrame({
        "Latitude": [0.2, 0.25],
        "Longitude": [0.1, 0.15],
        "dry_Demand": [5, 3],
    })
    monkeypatch.setattr(VRPSolver, "_prepare_model", lambda self: None)
    solv = VRPSolver(df, simple_params_two_goods, time_limit=1)
    solv.client_products = ["dry", "dry"]

    # Patch log functions to no-op to silence output
    import fleetmix.utils.logging as flog
    for fname in ("log_progress", "log_error", "log_detail"):
        monkeypatch.setattr(flog, fname, lambda *a, **k: None)

    # Infeasible branch
    solv._print_solution(float("inf"), 0.0, 0, [], 0.5, [], BenchmarkType.SINGLE_COMPARTMENT)

    # Feasible branch
    solv._print_solution(100.0, 10.0, 1, [[1, 2]], 0.1, [0.5], BenchmarkType.SINGLE_COMPARTMENT) 