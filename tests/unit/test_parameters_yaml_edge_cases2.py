import textwrap
import yaml
from pathlib import Path
import pytest

from fleetmix.config.parameters import Parameters

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path, content: str) -> Path:
    """Helper that writes *content* to a new YAML file in *tmp_path* and returns the path."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(textwrap.dedent(content))
    return cfg_path


# ---------------------------------------------------------------------------
# Nominal – minimal valid configuration
# ---------------------------------------------------------------------------

MINIMAL_YAML = """
vehicles:
  van:
    capacity: 100
    fixed_cost: 0
    avg_speed: 30
    service_time: 5
    max_route_time: 480

depot:
  latitude: 0.0
  longitude: 0.0

goods: [dry]
demand_file: some.csv
format: csv
clustering:
  geo_weight: 0.7
  demand_weight: 0.3
variable_cost_per_hour: 10
light_load_penalty: 0.0
light_load_threshold: 0.2
compartment_setup_cost: 0.0
"""


def test_from_yaml_happy_path(tmp_path):
    cfg_path = _write_yaml(tmp_path, MINIMAL_YAML)
    params = Parameters.from_yaml(cfg_path)

    # basic field assertions
    assert params.vehicles["van"].capacity == 100
    assert params.depot.latitude == 0.0
    assert params.goods == ["dry"]

    # results_dir should be absolute and exist
    assert params.results_dir.is_absolute()
    assert params.results_dir.exists()


def test_to_yaml_roundtrip(tmp_path):
    cfg_path = _write_yaml(tmp_path, MINIMAL_YAML)
    params = Parameters.from_yaml(cfg_path)

    out_path = tmp_path / "exported.yaml"
    params.to_yaml(out_path)

    assert out_path.exists()
    loaded_again = Parameters.from_yaml(out_path)
    # vehicles capacities should match
    assert loaded_again.vehicles["van"].capacity == params.vehicles["van"].capacity


# ---------------------------------------------------------------------------
# Validation error branches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_field", ["vehicles", "goods", "demand_file"])
def test_missing_required_fields(tmp_path, missing_field):
    """Ensure that omitting any required top-level key raises a ValueError."""
    bad_yaml = MINIMAL_YAML.replace(f"{missing_field}:", f"# {missing_field}:")
    cfg_path = _write_yaml(tmp_path, bad_yaml)
    with pytest.raises(ValueError):
        Parameters.from_yaml(cfg_path)


def test_invalid_yaml_syntax(tmp_path):
    malformed = MINIMAL_YAML + "\nfoo: [unclosed\n"
    cfg_path = _write_yaml(tmp_path, malformed)
    with pytest.raises(ValueError):
        Parameters.from_yaml(cfg_path)


def test_weight_sum_validation(tmp_path):
    bad_yaml = MINIMAL_YAML.replace("demand_weight: 0.3", "demand_weight: 0.5")
    cfg_path = _write_yaml(tmp_path, bad_yaml)
    with pytest.raises(ValueError):
        Parameters.from_yaml(cfg_path)


@pytest.mark.parametrize(
    "yaml_mod, expected_msg",
    [
        ("variable_cost_per_hour:", "Missing required configuration"),
        ("unknown_key: 123", "Unknown configuration fields"),
    ],
)
def test_typeerror_branches(tmp_path, yaml_mod, expected_msg):
    """Trigger the two TypeError branches in Parameters.from_yaml."""
    if "unknown_key" in yaml_mod:
        modified = MINIMAL_YAML + "\n" + yaml_mod + "\n"
    else:
        # comment out an existing required field to trigger missing-field TypeError
        modified = MINIMAL_YAML.replace(yaml_mod, f"# {yaml_mod}")

    cfg_path = _write_yaml(tmp_path, modified)
    with pytest.raises(ValueError) as exc:
        Parameters.from_yaml(cfg_path)
    assert expected_msg in str(exc.value) 