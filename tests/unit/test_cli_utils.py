import pytest
import yaml
from argparse import Namespace
from io import StringIO
from unittest.mock import patch

from fleetmix.utils.cli import parse_args, get_parameter_overrides, load_parameters, print_parameter_help
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
    # Only include non-None and parameter keys
    assert overrides == {'avg_speed': 50.0, 'demand_weight': 0.3}


def test_parse_args_invalid_choice():
    parser = parse_args()
    with pytest.raises(SystemExit):
        parser.parse_args(['--route-time-estimation', 'INVALID'])


def write_minimal_yaml(path):
    cfg = {
        'vehicles': {'A': {'capacity': 10, 'fixed_cost': 5}},
        'variable_cost_per_hour': 1,
        'avg_speed': 20,
        'max_route_time': 5,
        'service_time': 10,
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
    # Values from YAML
    assert params.avg_speed == 20
    assert params.service_time == 10
    assert params.demand_file == 'file.csv'
    assert params.clustering['method'] == 'minibatch_kmeans'


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


class TestCliUtils:
    """Test CLI utility functions"""
    
    def test_print_parameter_help(self, capsys):
        """Test print_parameter_help displays help and exits"""
        with pytest.raises(SystemExit) as exc_info:
            print_parameter_help()
        
        # Check it exits with code 0
        assert exc_info.value.code == 0
        
        # Check help text was printed
        captured = capsys.readouterr()
        assert "Fleet Size and Mix Optimization Parameters" in captured.out
        assert "Core Parameters:" in captured.out
        assert "--avg-speed" in captured.out
        assert "--max-route-time" in captured.out
        assert "--service-time" in captured.out
        assert "--route-time-estimation" in captured.out
        assert "--light-load-penalty" in captured.out
        assert "--light-load-threshold" in captured.out
        assert "--compartment-setup-cost" in captured.out
        assert "Clustering Options:" in captured.out
        assert "--clustering-method" in captured.out
        assert "--clustering-distance" in captured.out
        assert "--geo-weight" in captured.out
        assert "--demand-weight" in captured.out
        assert "Input/Output:" in captured.out
        assert "--demand-file" in captured.out
        assert "--config" in captured.out
        assert "--format" in captured.out
        assert "Other Options:" in captured.out
        assert "--verbose" in captured.out
        assert "Examples:" in captured.out 