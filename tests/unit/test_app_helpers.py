import pytest

from fleetmix.app import _find_config_by_id, _setup_logging_from_flags
from fleetmix.core_types import VehicleConfiguration
from fleetmix.utils.logging import LogLevel


def test_find_config_by_id_success():
    configs = [
        VehicleConfiguration(
            config_id=1,
            vehicle_type="Truck",
            capacity=100,
            fixed_cost=100.0,
            compartments={"Dry": True},
        ),
        VehicleConfiguration(
            config_id=2,
            vehicle_type="Van",
            capacity=50,
            fixed_cost=50.0,
            compartments={"Dry": True},
        ),
    ]
    cfg = _find_config_by_id(configs, "2")
    assert cfg.vehicle_type == "Van"


def test_find_config_by_id_not_found():
    with pytest.raises(KeyError):
        _find_config_by_id([], "99")


def test_setup_logging_flag_precedence(monkeypatch):
    called_levels = []

    def fake_setup(level: LogLevel | None = None):
        called_levels.append(level)

    monkeypatch.setattr("fleetmix.app.setup_logging", fake_setup)

    # debug overrides verbose/quiet
    _setup_logging_from_flags(verbose=True, quiet=True, debug=True)
    assert called_levels[-1] == LogLevel.DEBUG

    # verbose overrides quiet when debug not set
    _setup_logging_from_flags(verbose=True, quiet=True, debug=False)
    assert called_levels[-1] == LogLevel.VERBOSE

    # quiet when only quiet=True
    _setup_logging_from_flags(verbose=False, quiet=True, debug=False)
    assert called_levels[-1] == LogLevel.QUIET

    # default path (no flags) passes None
    _setup_logging_from_flags(False, False, False)
    assert called_levels[-1] is None 