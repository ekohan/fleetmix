import pytest
import yaml
from argparse import Namespace

from fleetmix.utils.cli import parse_args, get_parameter_overrides, load_parameters
from fleetmix.config.parameters import Parameters


def test_get_parameter_overrides_filters_none_and_keys():
    args = Namespace(
        config=None,
        avg_speed=50.0,
        max_route_time=None,
        service_time=None,
        demand_file=None,
        light_load_penalty=None,
        light_load_threshold=None,
        compartment_setup_cost=None,
        verbose=True,
        route_time_estimation=None,
        clustering_method=None,
        clustering_distance=None,
        geo_weight=None,
        demand_weight=0.3,
        format=None,
        help_params=False
    )
    overrides = get_parameter_overrides(args)
    # Only include non-None and parameter keys - avg_speed is no longer a global parameter
    assert overrides == {'demand_weight': 0.3}


def test_parse_args_invalid_choice():
    parser = parse_args()
    with pytest.raises(SystemExit):
        parser.parse_args(['--route-time-estimation', 'INVALID'])


def write_minimal_yaml(path):
    cfg = {
        'vehicles': {
            'A': {
                'capacity': 10, 
                'fixed_cost': 5,
                'avg_speed': 20,
                'service_time': 10,
                'max_route_time': 5
            }
        },
        'variable_cost_per_hour': 1,
        'depot': {'latitude': 0.0, 'longitude': 0.0},
        'goods': ['Dry'],
        'clustering': {
            'method': 'minibatch_kmeans',
            'distance': 'euclidean',
            'geo_weight': 0.5,
            'demand_weight': 0.5,
            'route_time_estimation': 'Legacy',
            'max_depth': 1
        },
        'demand_file': 'file.csv',
        'light_load_penalty': 0,
        'light_load_threshold': 0,
        'compartment_setup_cost': 0,
        'format': 'excel'
    }
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)


def test_load_parameters_default(tmp_path):
    # Write minimal YAML and load parameters
    yaml_path = tmp_path / 'cfg.yaml'
    write_minimal_yaml(yaml_path)
    parser = parse_args()
    args = parser.parse_args(['--config', str(yaml_path)])
    params = load_parameters(args)
    assert isinstance(params, Parameters)
    # Values from YAML - timing parameters are now in vehicle specs
    assert params.demand_file == 'file.csv'
    assert params.clustering['method'] == 'minibatch_kmeans'
    assert params.variable_cost_per_hour == 1
    # Check that vehicle specs have timing parameters
    vehicle_a = params.vehicles['A']
    assert vehicle_a.avg_speed == 20
    assert vehicle_a.service_time == 10
    assert vehicle_a.max_route_time == 5


def test_load_parameters_with_clustering_overrides(tmp_path):
    # Write minimal YAML
    yaml_path = tmp_path / 'cfg.yaml'
    write_minimal_yaml(yaml_path)
    # Override clustering-related flags
    parser = parse_args()
    args = parser.parse_args([
        '--config', str(yaml_path),
        '--clustering-method', 'agglomerative',
        '--clustering-distance', 'composite',
        '--geo-weight', '0.8',
        '--demand-weight', '0.2',
        '--route-time-estimation', 'TSP'
    ])
    params = load_parameters(args)
    c = params.clustering
    assert c['method'] == 'agglomerative'
    assert c['distance'] == 'composite'
    assert c['geo_weight'] == 0.8
    assert c['demand_weight'] == 0.2
    assert c['route_time_estimation'] == 'TSP' 