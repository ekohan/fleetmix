"""Validation tests for configuration parameter dataclasses."""

import pytest

from fleetmix.config.params import ProblemParams
from fleetmix.core_types import DepotLocation, VehicleSpec


def _base_vehicle_specs():
    return {
        "Type1": VehicleSpec(
            capacity=1000,
            fixed_cost=100,
            compartments={"Dry": True},
        )
    }


def _base_depot():
    return DepotLocation(latitude=0.0, longitude=0.0)


def _base_goods():
    return ["Dry", "Chilled"]


def test_problem_params_rejects_duplicate_goods():
    with pytest.raises(ValueError, match="goods contains duplicate"):
        ProblemParams(
            vehicles=_base_vehicle_specs(),
            depot=_base_depot(),
            goods=["Dry", "Dry"],
            variable_cost_per_hour=10.0,
        )


def test_problem_params_rejects_invalid_allowed_goods():
    vehicles = {
        "Type1": VehicleSpec(
            capacity=1000,
            fixed_cost=100,
            compartments={"Dry": True},
            allowed_goods=["Dry", "Chilled", "Frozen"],
        )
    }

    with pytest.raises(ValueError, match="allowed_goods contains goods"):
        ProblemParams(
            vehicles=vehicles,
            depot=_base_depot(),
            goods=_base_goods(),
            variable_cost_per_hour=10.0,
        )


