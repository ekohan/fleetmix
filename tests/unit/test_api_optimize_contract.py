import pandas as pd
import pytest
from pathlib import Path

from fleetmix.config import load_fleetmix_params
from fleetmix.config.params import FleetmixParams
from fleetmix.api import optimize
from fleetmix.core_types import FleetmixSolution


@pytest.fixture(scope="module")
def demand_df():
    """Minimal in-memory demand DataFrame accepted by :func:`optimize`."""
    return pd.DataFrame(
        {
            "Customer_ID": ["C1"],
            "Latitude": [0.0],
            "Longitude": [0.0],
            "Dry_Demand": [1.0],
            "Chilled_Demand": [0.0],
            "Frozen_Demand": [0.0],
        }
    )


@pytest.fixture(scope="module")
def params() -> FleetmixParams:
    cfg = (
        Path(__file__).resolve().parent.parent / "_assets" / "configs" / "test_config_minimal.yaml"
    )
    return load_fleetmix_params(cfg)


class DummySolution(FleetmixSolution):
    def __init__(self):
        super().__init__(
            total_fixed_cost=100.0,
            total_variable_cost=23.0,
            total_penalties=0.0,
            total_vehicles=1
        )


# ----------------------- successful path -----------------------

def test_optimize_happy_path(monkeypatch, demand_df, params):
    """`optimize` returns the object produced by `optimize_fleet` and preserves basic fields."""

    # Arrange – stub out heavy internals
    monkeypatch.setattr(
        "fleetmix.api.generate_vehicle_configurations",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "fleetmix.api.generate_feasible_clusters",
        lambda **k: ["cluster"],
    )
    monkeypatch.setattr(
        "fleetmix.api.optimize_fleet",
        lambda **k: DummySolution(),
    )
    monkeypatch.setattr(
        "fleetmix.api.save_optimization_results", lambda **k: None
    )

    # Act
    result = optimize(demand_df, params, output_dir="", verbose=False)

    # Assert
    assert isinstance(result, FleetmixSolution)
    assert result.total_cost == 123.0
    assert result.total_vehicles == 1


# ----------------------- failure path -----------------------

def test_optimize_invalid_demand(monkeypatch, params):
    """Missing required demand column should raise ``ValueError`` early, before heavy internals."""

    # DataFrame without demand columns
    bad_df = pd.DataFrame({"Customer_ID": ["1"], "Latitude": [0.0], "Longitude": [0.0]})

    with pytest.raises(ValueError):
        optimize(bad_df, params, output_dir="", verbose=False) 