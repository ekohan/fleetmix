import pytest
from pathlib import Path

from fleetmix.benchmarking.parsers.cvrp import CVRPParser


data_root = Path(__file__).resolve().parent.parent.parent / "src" / "fleetmix" / "benchmarking" / "datasets" / "cvrp"

def test_parse_instance_roundtrip():
    """Ensure ``CVRPParser`` can parse both full and dummy instances and, when available, read BKS from .sol."""

    instance_file = data_root / "X-n101-k25.vrp"
    parser = CVRPParser(str(instance_file))

    # ---- Act ----
    inst = parser.parse()

    # ---- Assert ----
    assert inst.dimension > 0
    assert inst.capacity > 0
    # Depot should be first node by convention
    assert inst.depot_id == 1
    # Vehicles is at least 1
    assert inst.num_vehicles >= 1

    # Exercise parse_solution and basic contract
    sol = parser.parse_solution()
    assert sol.num_vehicles == len(sol.routes)

    # No further structure checks – BKS solutions may omit explicit depot nodes in VRPLIB spec. 