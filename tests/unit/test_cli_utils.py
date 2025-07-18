from argparse import Namespace

import pytest
import yaml

from fleetmix.utils.cli import get_parameter_overrides, load_parameters, parse_args


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
        help_params=False,
    )
    overrides = get_parameter_overrides(args)
    # Only include non-None and parameter keys - avg_speed is no longer a global parameter
    assert overrides == {"demand_weight": 0.3}


def test_parse_args_invalid_choice():
    parser = parse_args()
    with pytest.raises(SystemExit):
        parser.parse_args(["--route-time-estimation", "INVALID"])


def write_minimal_yaml(path):
    cfg = {
        "vehicles": {
            "A": {
                "capacity": 10,
                "fixed_cost": 5,
                "avg_speed": 20,
                "service_time": 10,
                "max_route_time": 5,
            }
        },
        "variable_cost_per_hour": 1,
        "depot": {"latitude": 0.0, "longitude": 0.0},
        "goods": ["Dry"],
        "clustering": {
            "method": "minibatch_kmeans",
            "distance": "euclidean",
            "geo_weight": 0.5,
            "demand_weight": 0.5,
            "route_time_estimation": "Legacy",
            "max_depth": 1,
        },
        "demand_file": "file.csv",
        "light_load_penalty": 0,
        "light_load_threshold": 0,
        "compartment_setup_cost": 0,
        "format": "xlsx",
    }
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)

def test_load_parameters_with_clustering_overrides(tmp_path):
    # Write minimal YAML
    yaml_path = tmp_path / "cfg.yaml"
    write_minimal_yaml(yaml_path)
    # Override clustering-related flags
    parser = parse_args()
    args = parser.parse_args(
        [
            "--config",
            str(yaml_path),
            "--clustering-method",
            "agglomerative",
            "--clustering-distance",
            "composite",
            "--geo-weight",
            "0.8",
            "--demand-weight",
            "0.2",
            "--route-time-estimation",
            "TSP",
        ]
    )
    params = load_parameters(args)
    assert params.algorithm.clustering_method == "agglomerative"
    assert params.algorithm.clustering_distance == "composite"
    assert params.algorithm.geo_weight == 0.8
    assert params.algorithm.demand_weight == 0.2
    assert params.algorithm.route_time_estimation == "TSP"
