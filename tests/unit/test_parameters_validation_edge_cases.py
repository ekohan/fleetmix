import pytest

from fleetmix.config.parameters import Parameters
from fleetmix.core_types import DepotLocation, VehicleSpec


@pytest.fixture
def _base_kwargs(tmp_path):
    """Return a kwargs dict with required minimal fields for Parameters."""
    return {
        "vehicles": {
            "Truck": VehicleSpec(capacity=100, fixed_cost=100.0)
        },
        "variable_cost_per_hour": 50.0,
        "depot": DepotLocation(latitude=0.0, longitude=0.0),
        "goods": ["Dry"],
        "clustering": {"geo_weight": 0.7, "demand_weight": 0.3},
        "demand_file": "dummy.csv",
        "light_load_penalty": 10.0,
        "light_load_threshold": 0.5,
        "compartment_setup_cost": 5.0,
        "format": "json",
        "results_dir": tmp_path,
    }


def test_invalid_clustering_weights(_base_kwargs):
    """geo_weight + demand_weight must equal 1.0"""
    bad_kwargs = _base_kwargs.copy()
    bad_kwargs["clustering"] = {"geo_weight": 0.8, "demand_weight": 0.3}
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        Parameters(**bad_kwargs)


def test_negative_small_cluster_size(_base_kwargs):
    """small_cluster_size must be positive."""
    bad_kwargs = _base_kwargs.copy()
    bad_kwargs["small_cluster_size"] = -3
    with pytest.raises(ValueError, match="small_cluster_size"):
        Parameters(**bad_kwargs) 