import pytest
from argparse import Namespace

from fleetmix.config.parameters import Parameters
from fleetmix.utils.cli import load_parameters, get_parameter_overrides, parse_args


def test_default_yaml_weights_sum_to_one():
    # Load default config
    params = Parameters.from_yaml('src/fleetmix/config/default_config.yaml')
    geo = params.clustering['geo_weight']
    dem = params.clustering['demand_weight']
    assert pytest.approx(geo + dem, rel=1e-6) == 1.0


def test_invalid_weights_yaml(tmp_path):
    # Create invalid yaml file
    bad_yaml = tmp_path / 'bad.yaml'
    bad_yaml.write_text(
        "vehicles:\n  A:\n    capacity: 10\n    fixed_cost: 5\n    avg_speed: 30\n    service_time: 10\n    max_route_time: 5\nvariable_cost_per_hour: 1.0\ndepot:\n  latitude: 0.0\n  longitude: 0.0\ngoods:\n  - Dry\nclustering:\n  geo_weight: 0.8\n  demand_weight: 0.3\n  max_depth: 20\n  route_time_estimation: 'BHH'\ndemand_file: 'x.csv'\nlight_load_penalty: 0\nlight_load_threshold: 0.2\ncompartment_setup_cost: 50\nformat: 'excel'\n"
    )
    with pytest.raises(ValueError):
        _ = Parameters.from_yaml(str(bad_yaml))


def test_load_parameters_overrides():
    # Simulate CLI args
    args = Namespace(
        config=None,
        demand_file=None,
        light_load_penalty=500.0,
        light_load_threshold=None,
        compartment_setup_cost=None,
        verbose=False,
        route_time_estimation=None,
        clustering_method=None,
        clustering_distance=None,
        geo_weight=None,
        demand_weight=None,
        format=None,
        help_params=False
    )
    params = load_parameters(args)
    assert params.light_load_penalty == 500.0
    # Other values remain defaults
    default = Parameters.from_yaml('src/fleetmix/config/default_config.yaml')
    assert params.light_load_threshold == default.light_load_threshold


def test_get_parameter_overrides_filters_none():
    parser = parse_args()
    # Simulate args
    cli_args = parser.parse_args(['--light-load-penalty', '500', '--help-params'])
    overrides = get_parameter_overrides(cli_args)
    assert 'light_load_penalty' in overrides and overrides['light_load_penalty'] == 500
    assert 'help_params' not in overrides


def test_small_cluster_size_overrides(tmp_path):
    params = Parameters.from_yaml('src/fleetmix/config/default_config.yaml')
    assert params.small_cluster_size == 100
    assert params.nearest_merge_candidates == 100

    # Create a minimal YAML with overridden values
    yaml_content = (
        "vehicles:\n  A:\n    capacity: 10\n    fixed_cost: 5\n    avg_speed: 30\n    service_time: 10\n    max_route_time: 5\n"
        "variable_cost_per_hour: 1.0\n"
        "depot:\n  latitude: 0.0\n  longitude: 0.0\n"
        "goods:\n  - Dry\n"
        "clustering:\n  geo_weight: 0.5\n  demand_weight: 0.5\n"
        "demand_file: 'x.csv'\nlight_load_penalty: 0\nlight_load_threshold: 0.2\n"
        "compartment_setup_cost: 50\nformat: 'excel'\n"
        "small_cluster_size: 3\nnearest_merge_candidates: 5\n"
    )
    yaml_path = tmp_path / "test_override.yaml"
    yaml_path.write_text(yaml_content)

    params2 = Parameters.from_yaml(str(yaml_path))
    assert params2.small_cluster_size == 3
    assert params2.nearest_merge_candidates == 5 