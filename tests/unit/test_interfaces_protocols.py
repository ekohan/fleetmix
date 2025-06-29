# Tests for protocol declarations in fleetmix.interfaces
from fleetmix.interfaces import Clusterer, RouteTimeEstimator, SolverAdapter


def test_clusterer_protocol_present():
    assert hasattr(Clusterer, "fit")


def test_route_time_estimator_protocol_present():
    assert hasattr(RouteTimeEstimator, "estimate_route_time")


def test_solver_adapter_protocol_descriptors():
    # properties exist and are property objects
    assert isinstance(SolverAdapter.name, property)
    assert isinstance(SolverAdapter.available, property)
    # method exists and is callable
    assert callable(SolverAdapter.get_pulp_solver) 