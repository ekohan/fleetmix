import math

import pandas as pd

from fleetmix.core_types import (
    Cluster,
    DepotLocation,
    PseudoCustomer,
    VehicleConfiguration,
    VehicleSpec,
)
from fleetmix.config.params import FleetmixParams, ProblemParams, AlgorithmParams, IOParams, RuntimeParams
from fleetmix.optimization.core import optimize_fleet
from pathlib import Path


def _make_params(variable_cost_per_hour: float = 0.0) -> FleetmixParams:
    """Create a minimal FleetmixParams instance suitable for unit tests."""

    # Create minimal vehicle specs for the test
    minimal_vehicles = {
        "TestVehicle": VehicleSpec(
            capacity=1000,
            fixed_cost=100.0,
            compartments={"Dry": True, "Chilled": True, "Frozen": True},
        )
    }

    problem = ProblemParams(
        vehicles=minimal_vehicles,
        variable_cost_per_hour=variable_cost_per_hour,
        depot=DepotLocation(latitude=0.0, longitude=0.0),
        goods=["Dry", "Chilled", "Frozen"],
        light_load_penalty=0.0,
        light_load_threshold=0.0,
        compartment_setup_cost=0.0,  # Critical for this test
        allow_split_stops=True,  # Enable split-stop logic
    )
    
    algorithm = AlgorithmParams(
        clustering_max_depth=1,
        clustering_method="minibatch_kmeans",
        clustering_distance="euclidean",
        geo_weight=0.7,
        demand_weight=0.3,
        route_time_estimation="BHH",
        post_optimization=False,  # Keep solver deterministic for the test
    )
    
    io = IOParams(
        demand_file="dummy.csv",
        results_dir=Path("results"),
        format="json",
    )
    
    runtime = RuntimeParams(
        config=Path("test_config.yaml"),
        verbose=False,
        debug=False,
    )

    return FleetmixParams(
        problem=problem,
        algorithm=algorithm,
        io=io,
        runtime=runtime,
    )


def _make_test_data():
    """Prepare configurations, customers and clusters for the test."""

    # Vehicle configurations -------------------------------------------------
    single_dry = VehicleConfiguration(
        config_id=1,
        vehicle_type="SingleDry",
        capacity=100,
        fixed_cost=60.0,
        compartments={"Dry": True, "Chilled": False, "Frozen": False},
    )

    single_chilled = VehicleConfiguration(
        config_id=2,
        vehicle_type="SingleChilled",
        capacity=100,
        fixed_cost=60.0,
        compartments={"Dry": False, "Chilled": True, "Frozen": False},
    )

    multi_dry_chilled = VehicleConfiguration(
        config_id=3,
        vehicle_type="MultiDryChilled",
        capacity=100,
        fixed_cost=90.0,  # Cheaper than two single-compartment vehicles (60 + 60)
        compartments={"Dry": True, "Chilled": True, "Frozen": False},
    )

    configurations = [single_dry, single_chilled, multi_dry_chilled]

    # Pseudo-customers -------------------------------------------------------
    pc_dry = PseudoCustomer(
        customer_id="C1::Dry",
        origin_id="C1",
        subset=("Dry",),
        demands={"Dry": 10.0, "Chilled": 0.0, "Frozen": 0.0},
        location=(0.0, 0.0),
        service_time=25.0,
    )

    pc_chilled = PseudoCustomer(
        customer_id="C1::Chilled",
        origin_id="C1",
        subset=("Chilled",),
        demands={"Dry": 0.0, "Chilled": 5.0, "Frozen": 0.0},
        location=(0.0, 0.0),
        service_time=25.0,
    )

    pc_both = PseudoCustomer(
        customer_id="C1::Dry-Chilled",
        origin_id="C1",
        subset=("Dry", "Chilled"),
        demands={"Dry": 10.0, "Chilled": 5.0, "Frozen": 0.0},
        location=(0.0, 0.0),
        service_time=25.0,
    )

    customers = [pc_dry, pc_chilled, pc_both]

    # Clusters ---------------------------------------------------------------
    cluster_multi = Cluster(
        cluster_id=101,
        config_id=3,  # Uses the multi-compartment vehicle
        customers=[pc_both.customer_id],
        total_demand={"Dry": 10.0, "Chilled": 5.0, "Frozen": 0.0},
        centroid_latitude=0.0,
        centroid_longitude=0.0,
        goods_in_config=["Dry", "Chilled"],
        route_time=1.0,  # hours – irrelevant because variable_cost_per_hour=0
        method="unit_test",
    )

    cluster_dry = Cluster(
        cluster_id=102,
        config_id=1,
        customers=[pc_dry.customer_id],
        total_demand={"Dry": 10.0, "Chilled": 0.0, "Frozen": 0.0},
        centroid_latitude=0.0,
        centroid_longitude=0.0,
        goods_in_config=["Dry"],
        route_time=1.0,
        method="unit_test",
    )

    cluster_chilled = Cluster(
        cluster_id=103,
        config_id=2,
        customers=[pc_chilled.customer_id],
        total_demand={"Dry": 0.0, "Chilled": 5.0, "Frozen": 0.0},
        centroid_latitude=0.0,
        centroid_longitude=0.0,
        goods_in_config=["Chilled"],
        route_time=1.0,
        method="unit_test",
    )

    clusters = [cluster_multi, cluster_dry, cluster_chilled]
    return configurations, customers, clusters


def test_multi_compartment_selected_when_cheaper():
    """With compartment cost zero, solver should prefer one multi-compartment vehicle."""

    params = _make_params(variable_cost_per_hour=0.0)
    configurations, customers, clusters = _make_test_data()

    solution = optimize_fleet(clusters, configurations, customers, params)

    # Exactly one vehicle / cluster should be selected
    assert solution.total_vehicles == 1, "Expected a single vehicle to serve both goods"

    # The selected configuration must be the multi-compartment one (config_id=3)
    selected_configs = set(cluster.config_id for cluster in solution.selected_clusters)
    assert selected_configs == {3}, f"Solver chose configs {selected_configs}, expected {{3}}"

    # Sanity-check: total cost must equal the fixed cost of the multi-compartment vehicle
    assert math.isclose(solution.total_fixed_cost, 90.0, rel_tol=1e-6)
    assert math.isclose(solution.total_cost, 90.0, rel_tol=1e-6) 