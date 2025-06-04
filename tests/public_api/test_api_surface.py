"""Test the public API surface of fleetmix."""
import pytest


def test_public_api_exports():
    """Test that the public API exports exactly what we expect."""
    import fleetmix
    
    expected_exports = {
        "optimize",
        "generate_vehicle_configurations",
        "generate_feasible_clusters",
        "optimize_fleet_selection",
        "improve_solution",
        "VehicleConfiguration",
        "ClusterAssignment",
        "FleetmixSolution",
        "Parameters",
        "__version__",
    }
    
    actual_exports = set(fleetmix.__all__)
    
    assert actual_exports == expected_exports, (
        f"Unexpected exports. "
        f"Missing: {expected_exports - actual_exports}, "
        f"Extra: {actual_exports - expected_exports}"
    )


def test_can_import_all_public_symbols():
    """Test that all public symbols can be imported."""
    from fleetmix import (
        optimize,
        generate_vehicle_configurations,
        generate_feasible_clusters,
        optimize_fleet_selection,
        improve_solution,
        VehicleConfiguration,
        ClusterAssignment,
        FleetmixSolution,
        Parameters,
        __version__,
    )
    
    # Verify they exist
    assert callable(optimize)
    assert callable(generate_vehicle_configurations)
    assert callable(generate_feasible_clusters)
    assert callable(optimize_fleet_selection)
    assert callable(improve_solution)
    
    # Verify types are classes
    assert isinstance(VehicleConfiguration, type)
    assert isinstance(ClusterAssignment, type)
    assert isinstance(FleetmixSolution, type)
    assert isinstance(Parameters, type)
    
    # Verify version is a string
    assert isinstance(__version__, str)


def test_public_types_structure():
    """Test that public types have the expected structure."""
    from fleetmix import VehicleConfiguration, ClusterAssignment, FleetmixSolution
    
    # Test VehicleConfiguration
    config = VehicleConfiguration(
        config_id=1,
        vehicle_type="truck",
        compartments={"dry": True, "frozen": False},
        capacity=100,
        fixed_cost=500.0,
        avg_speed=30.0,
        service_time=25.0,
        max_route_time=10.0
    )
    assert config.config_id == 1
    assert config.vehicle_type == "truck"
    assert config.compartments == {"dry": True, "frozen": False}
    assert config.capacity == 100
    assert config.fixed_cost == 500.0
    assert config.avg_speed == 30.0
    assert config.service_time == 25.0
    assert config.max_route_time == 10.0
    
    # Test ClusterAssignment
    cluster = ClusterAssignment(
        cluster_id=1,
        config_id=1,
        customer_ids=["C1", "C2"],
        route_time=2.5,
        total_demand={"dry": 50, "frozen": 0},
        centroid=(40.7128, -74.0060)
    )
    assert cluster.cluster_id == 1
    assert cluster.config_id == 1
    assert cluster.customer_ids == ["C1", "C2"]
    assert cluster.route_time == 2.5
    assert cluster.total_demand == {"dry": 50, "frozen": 0}
    assert cluster.centroid == (40.7128, -74.0060)
    
    # Test FleetmixSolution
    solution = FleetmixSolution(
        selected_clusters=[cluster],
        configurations_used=[config],
        total_cost=1000.0,
        total_vehicles=1,
        missing_customers=set(),
        solver_status="Optimal",
        solver_runtime_sec=0.5
    )
    assert len(solution.selected_clusters) == 1
    assert len(solution.configurations_used) == 1
    assert solution.total_cost == 1000.0
    assert solution.total_vehicles == 1
    assert solution.missing_customers == set()
    assert solution.solver_status == "Optimal"
    assert solution.solver_runtime_sec == 0.5 