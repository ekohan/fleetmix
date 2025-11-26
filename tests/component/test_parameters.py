import pytest

from fleetmix.config import load_fleetmix_params



def test_default_yaml_weights_sum_to_one():
    # Load default config
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")
    geo = params.algorithm.geo_weight
    dem = params.algorithm.demand_weight
    assert pytest.approx(geo + dem, rel=1e-6) == 1.0


def test_invalid_weights_yaml(tmp_path):
    # Create invalid yaml file
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "vehicles:\n  A:\n    capacity: 10\n    fixed_cost: 5\n    avg_speed: 30\n    service_time: 10\n    max_route_time: 5\nvariable_cost_per_hour: 1.0\ndepot:\n  latitude: 0.0\n  longitude: 0.0\ngoods:\n  - Dry\nclustering:\n  geo_weight: 0.8\n  demand_weight: 0.3\n  max_depth: 20\n  route_time_estimation: 'BHH'\ndemand_file: 'x.csv'\nlight_load_penalty: 0\nlight_load_threshold: 0.2\ncompartment_setup_cost: 50\nformat: 'excel'\n"
    )
    with pytest.raises(ValueError):
        _ = load_fleetmix_params(str(bad_yaml))




def test_small_cluster_size_overrides(tmp_path):
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")
    assert params.algorithm.small_cluster_size == 1000
    assert params.algorithm.nearest_merge_candidates == 1000

    # Create a minimal YAML with overridden values (using old flat format)
    yaml_content = (
        "vehicles:\n  A:\n    capacity: 10\n    fixed_cost: 5\n    avg_speed: 30\n    service_time: 10\n    max_route_time: 5\n"
        "variable_cost_per_hour: 1.0\n"
        "depot:\n  latitude: 0.0\n  longitude: 0.0\n"
        "goods:\n  - Dry\n"
        "clustering:\n  geo_weight: 0.5\n  demand_weight: 0.5\n"
        "demand_file: 'x.csv'\nlight_load_penalty: 0\nlight_load_threshold: 0.2\n"
        "compartment_setup_cost: 50\nformat: 'xlsx'\n"
        "small_cluster_size: 3\nnearest_merge_candidates: 5\n"
    )
    yaml_path = tmp_path / "test_override.yaml"
    yaml_path.write_text(yaml_content)

    params2 = load_fleetmix_params(str(yaml_path))
    assert params2.algorithm.small_cluster_size == 3
    assert params2.algorithm.nearest_merge_candidates == 5
