import pytest
from argparse import Namespace

from fleetmix.config.parameters import Parameters
from fleetmix.internal_types import VehicleSpec, DepotLocation
from fleetmix.utils.cli import load_parameters, get_parameter_overrides, parse_args


def test_default_yaml_weights_sum_to_one():
    # Load default config
    params = Parameters.from_yaml('src/fleetmix/config/default_config.yaml')
    geo = params.clustering['geo_weight']
    dem = params.clustering['demand_weight']
    assert pytest.approx(geo + dem, rel=1e-6) == 1.0


def test_invalid_weights_yaml(tmp_path):
    # Create invalid yaml file with per-vehicle parameters
    bad_yaml = tmp_path / 'bad.yaml'
    bad_yaml.write_text(
        "vehicles:\n  A:\n    capacity: 10\n    fixed_cost: 5\n    avg_speed: 30.0\n    service_time: 25.0\n    max_route_time: 10.0\nvariable_cost_per_hour: 1.0\ndepot:\n  latitude: 0.0\n  longitude: 0.0\ngoods:\n  - Dry\nclustering:\n  geo_weight: 0.8\n  demand_weight: 0.3\ndemand_file: 'x.csv'\nlight_load_penalty: 0\nlight_load_threshold: 0.2\ncompartment_setup_cost: 50\nformat: 'excel'\n"
    )
    with pytest.raises(ValueError):
        _ = Parameters.from_yaml(str(bad_yaml))


def test_load_parameters_overrides():
    # Simulate CLI args
    args = Namespace(
        config=None,
        demand_file=None,
        light_load_penalty=None,
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
    # Check that vehicles have the new per-vehicle parameters
    for vehicle_name, vehicle_spec in params.vehicles.items():
        assert hasattr(vehicle_spec, 'avg_speed')
        assert hasattr(vehicle_spec, 'service_time')
        assert hasattr(vehicle_spec, 'max_route_time')


def test_per_vehicle_parameters():
    """Test that per-vehicle parameters are loaded correctly."""
    params = Parameters.from_yaml('src/fleetmix/config/default_config.yaml')
    
    # Check that each vehicle has operational parameters
    for vehicle_name, vehicle_spec in params.vehicles.items():
        assert vehicle_spec.avg_speed > 0
        assert vehicle_spec.service_time >= 0
        assert vehicle_spec.max_route_time > 0
        
    # Test creating Parameters with custom per-vehicle values
    params = Parameters(
        vehicles={
            "FastTruck": VehicleSpec(
                capacity=1000, 
                fixed_cost=100, 
                avg_speed=50.0,  # Faster than default
                service_time=15.0,  # Quicker service
                max_route_time=8.0  # Shorter routes
            ),
            "SlowTruck": VehicleSpec(
                capacity=2000,
                fixed_cost=150,
                avg_speed=20.0,  # Slower
                service_time=30.0,  # Longer service
                max_route_time=12.0  # Longer routes allowed
            )
        },
        variable_cost_per_hour=10,
        depot=DepotLocation(latitude=0, longitude=0),
        goods=["Dry"],
        clustering={"method": "kmeans", "geo_weight": 0.7, "demand_weight": 0.3},
        demand_file="test.csv",
        light_load_penalty=0,
        light_load_threshold=0,
        compartment_setup_cost=0,
        format="excel"
    )
    
    assert params.vehicles["FastTruck"].avg_speed == 50.0
    assert params.vehicles["SlowTruck"].avg_speed == 20.0


def test_small_cluster_size_overrides(tmp_path):
    params = Parameters.from_yaml('src/fleetmix/config/default_config.yaml')
    assert params.small_cluster_size == 100
    assert params.nearest_merge_candidates == 100

    # Create a minimal YAML with overridden values and per-vehicle parameters
    yaml_content = (
        "vehicles:\n  A:\n    capacity: 10\n    fixed_cost: 5\n"
        "    avg_speed: 30.0\n    service_time: 25.0\n    max_route_time: 10.0\n"
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