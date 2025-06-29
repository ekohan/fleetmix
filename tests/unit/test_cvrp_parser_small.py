from pathlib import Path
import pytest

from fleetmix.benchmarking.parsers.cvrp import CVRPParser

DATA_DIR = Path(__file__).parents[2] / "src" / "fleetmix" / "benchmarking" / "datasets" / "cvrp"


def test_parse_instance_with_bks():
    """Instance with matching .sol file should use _get_bks_vehicles()."""
    fname = "X-n101-k25.vrp"  # .sol exists next to it
    parser = CVRPParser(str(DATA_DIR / fname))
    instance = parser.parse()
    # ensure vehicles taken from .sol (not default 1)
    assert instance.num_vehicles > 1
    assert instance.dimension > 0
    assert instance.capacity > 0


def test_parse_instance_without_sol(monkeypatch):
    """Force _get_bks_vehicles to fail so fallback path uses k-value from name."""
    fname = "X-n120-k6.vrp"

    # Ensure .vrp file exists
    path = DATA_DIR / fname
    assert path.exists(), "Test VRP file missing in dataset"

    # Monkeypatch _get_bks_vehicles to simulate missing .sol
    monkeypatch.setattr(CVRPParser, "_get_bks_vehicles", lambda self: (_ for _ in ()).throw(FileNotFoundError()))

    parser = CVRPParser(str(path))
    instance = parser.parse()
    # expect k-value from filename
    assert instance.num_vehicles == 6


@pytest.mark.parametrize("filename", ["X-n101-k25.vrp", "dummy.vrp"])
def test_parse_solution_roundtrip(filename):
    path = DATA_DIR / filename
    parser = CVRPParser(str(path))
    if (path.with_suffix(".sol")).exists():
        # Should parse solution OK
        sol = parser.parse_solution()
        assert sol.num_vehicles == len(sol.routes)
        for route in sol.routes:
            assert route, "Route should not be empty"
    else:
        # Expect FileNotFoundError when .sol missing
        with pytest.raises(FileNotFoundError):
            parser.parse_solution()


def test_parse_instance_no_demand(monkeypatch, tmp_path):
    """Monkeypatch vrplib.read_instance to omit 'demand' key to cover branch."""
    dummy_file = tmp_path / "fake-k3.vrp"
    dummy_file.write_text("dummy")

    def fake_read_instance(path):
        return {
            "node_coord": [(0, 0), (10, 10), (20, 20)],
            "capacity": 50,
            "depot": [0],
            # no 'demand' key triggers else branch
        }

    monkeypatch.setattr(CVRPParser, "_get_bks_vehicles", lambda self: 3)
    import vrplib
    monkeypatch.setattr(vrplib, "read_instance", fake_read_instance)

    parser = CVRPParser(str(dummy_file))
    instance = parser.parse()
    assert instance.demands == {}
    assert instance.num_vehicles == 3 