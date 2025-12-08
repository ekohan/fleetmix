"""Unit tests for helper utilities in `fleetmix.app`."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest

from fleetmix.app import (
    _find_config_by_id,
    _setup_logging_from_flags,
)
from fleetmix.config import load_fleetmix_params
from fleetmix.config.params import FleetmixParams
from fleetmix.core_types import VehicleConfiguration


@pytest.fixture(scope="module")
def default_params() -> FleetmixParams:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fleetmix"
        / "config"
        / "default_config.yaml"
    )
    return load_fleetmix_params(config_path)


def make_params(base: FleetmixParams, tmp_path: Path, **problem_overrides) -> FleetmixParams:
    problem = dataclasses.replace(base.problem, **problem_overrides)
    return dataclasses.replace(
        base,
        problem=problem,
        io=dataclasses.replace(base.io, results_dir=tmp_path),
    )


def test_find_config_by_id_success() -> None:
    configs = [
        VehicleConfiguration(
            config_id="1",
            vehicle_type="Truck",
            capacity=100,
            fixed_cost=100.0,
            compartments={"Dry": True},
        ),
        VehicleConfiguration(
            config_id="2",
            vehicle_type="Van",
            capacity=80,
            fixed_cost=75.0,
            compartments={"Dry": True},
        ),
    ]
    result = _find_config_by_id(configs, "2")
    assert result.vehicle_type == "Van"


def test_find_config_by_id_missing() -> None:
    with pytest.raises(KeyError):
        _find_config_by_id([], "missing")









@patch("fleetmix.app.setup_logging")
@pytest.mark.parametrize(
    "verbose,quiet,debug,expected",
    [
        (False, False, False, None),
        (True, False, False, "VERBOSE"),
        (False, True, False, "QUIET"),
        (False, False, True, "DEBUG"),
    ],
)
def test_setup_logging_from_flags(mock_setup_logging, verbose, quiet, debug, expected):
    _setup_logging_from_flags(verbose=verbose, quiet=quiet, debug=debug)
    if expected is None:
        mock_setup_logging.assert_called_once_with()
    else:
        from fleetmix.utils.logging import LogLevel

        mock_setup_logging.assert_called_once_with(getattr(LogLevel, expected))


@patch("fleetmix.app.setup_logging")
def test_setup_logging_priority(mock_setup_logging):
    from fleetmix.utils.logging import LogLevel

    _setup_logging_from_flags(verbose=True, quiet=True, debug=True)
    mock_setup_logging.assert_called_with(LogLevel.DEBUG)

    _setup_logging_from_flags(verbose=True, quiet=True, debug=False)
    mock_setup_logging.assert_called_with(LogLevel.VERBOSE)








