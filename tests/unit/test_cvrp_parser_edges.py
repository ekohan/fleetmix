"""Tests for edge cases in the CVRP parser."""

import logging

import pytest

from fleetmix.benchmarking.models import CVRPInstance, CVRPSolution
from fleetmix.benchmarking.parsers.cvrp import CVRPParser


# Stub out vrplib
class DummyVRPLib:
    @staticmethod
    def read_instance(path):
        return {
            "node_coord": [(0.0, 0.0), (1.0, 1.0)],
            "demand": [10, 20],
            "capacity": 50,
            "depot": [0],
            # no edge_weight_type provided to test default
        }

    @staticmethod
    def read_solution(path):
        return {"routes": [[0, 1], [2]], "cost": 123.45}


@pytest.fixture(autouse=True)
def patch_vrplib(monkeypatch):
    # Monkey-patch the vrplib module in cvrp_parser
    import fleetmix.benchmarking.parsers.cvrp as parser_mod

    monkeypatch.setattr(parser_mod, "vrplib", DummyVRPLib)
    yield


def test_parse_fills_defaults(tmp_path):
    # Create dummy files
    vrp_file = tmp_path / "X-k2.vrp"
    sol_file = tmp_path / "X-k2.sol"
    vrp_file.write_text("")
    sol_file.write_text("")

    parser = CVRPParser(str(vrp_file))
    inst = parser.parse()
    # Name, dimension, capacity
    assert isinstance(inst, CVRPInstance)
    assert inst.name == "X-k2"
    assert inst.dimension == 2
    assert inst.capacity == 50
    # Coordinates and demands
    assert inst.coordinates == {1: (0.0, 0.0), 2: (1.0, 1.0)}
    assert inst.demands == {1: 10.0, 2: 20.0}
    # Default edge weight type
    assert inst.edge_weight_type == "EUC_2D"
    # Depot id one-based
    assert inst.depot_id == 1
    # num_vehicles from name
    assert inst.num_vehicles == 2


def test_parse_solution_logs_info(tmp_path, caplog):
    # Create dummy files
    vrp_file = tmp_path / "A-k3.vrp"
    sol_file = tmp_path / "A-k3.sol"
    vrp_file.write_text("")
    sol_file.write_text("")

    parser = CVRPParser(str(vrp_file))
    # instance_name has '-k3', but BKS has 2 routes
    caplog.set_level(logging.INFO)
    sol = parser.parse_solution()
    assert isinstance(sol, CVRPSolution)
    # actual routes = 2 (BKS)
    assert sol.num_vehicles == 2
    # Should log info about BKS vs k difference
    # Use record_tuples to check for info messages more robustly
    assert any(
        rec[0] == "fleetmix.benchmarking.parsers.cvrp"
        and rec[1] == logging.INFO
        and "differs" in rec[2].lower()
        for rec in caplog.record_tuples
    ), "Expected info message about BKS vs k mismatch"
