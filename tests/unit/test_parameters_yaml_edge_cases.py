import pytest
import yaml
from pathlib import Path

from fleetmix.config.parameters import Parameters


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    f = tmp_path / "params.yaml"
    with open(f, "w") as fp:
        yaml.dump(data, fp)
    return f


def _minimal_yaml_dict():
    return {
        "vehicles": {
            "Truck": {
                "capacity": 100,
                "fixed_cost": 100,
                "avg_speed": 30,
                "service_time": 25,
                "max_route_time": 10,
                "compartments": {"Dry": True},
            }
        },
        "variable_cost_per_hour": 50,
        "depot": {"latitude": 0.0, "longitude": 0.0},
        "goods": ["Dry"],
        "clustering": {"geo_weight": 0.7, "demand_weight": 0.3},
        "demand_file": "dummy.csv",
        "light_load_penalty": 10,
        "light_load_threshold": 0.5,
        "compartment_setup_cost": 5,
        "format": "json",
    }


def test_from_yaml_with_unknown_field(tmp_path):
    data = _minimal_yaml_dict()
    data["unknown_field"] = 123
    yaml_path = _write_yaml(tmp_path, data)

    # Expect ValueError mentioning unknown configuration fields
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        Parameters.from_yaml(yaml_path)


def test_from_yaml_missing_required_field(tmp_path):
    data = _minimal_yaml_dict()
    del data["demand_file"]  # remove required field
    yaml_path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="Missing required fields"):
        Parameters.from_yaml(yaml_path) 