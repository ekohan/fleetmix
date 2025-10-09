import pytest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fleetmix.clustering.generator import (
    _deduplicate_clusters,
    _generate_feasibility_mapping,
    _get_clustering_context_list,
    generate_feasible_clusters,
    process_configuration,
)
from fleetmix.config.params import (
    AlgorithmParams,
    FleetmixParams,
    IOParams,
    ProblemParams,
    RuntimeParams,
)
from fleetmix.core_types import (
    CapacitatedClusteringContext,
    Cluster,
    CustomerBase,
    DepotLocation,
    VehicleSpec,
    VehicleConfiguration,
)


class DummyCustomer(CustomerBase):
    def is_pseudo_customer(self) -> bool:
        return False

    def get_origin_id(self) -> str:
        return self.customer_id

    def get_goods_subset(self) -> tuple[str, ...]:
        return tuple(self.demands.keys())


@pytest.fixture
def sample_goods():
    return ["Dry", "Chilled"]


@pytest.fixture
def sample_customers():
    return [
        DummyCustomer("C1", {"Dry": 5, "Chilled": 0}, (0.0, 0.0), 10),
        DummyCustomer("C2", {"Dry": 3, "Chilled": 2}, (1.0, 1.0), 10),
        DummyCustomer("C3", {"Dry": 4, "Chilled": 0}, (2.0, 0.5), 10),
    ]


@pytest.fixture
def sample_configurations(sample_goods):
    return [
        VehicleConfiguration(
            config_id="1",
            vehicle_type="SmallVan",
            capacity=15,
            fixed_cost=100,
            compartments={good: True for good in sample_goods},
            avg_speed=40,
            service_time=15,
            max_route_time=8,
        ),
        VehicleConfiguration(
            config_id="2",
            vehicle_type="LargeVan",
            capacity=25,
            fixed_cost=150,
            compartments={good: True for good in sample_goods},
            avg_speed=45,
            service_time=10,
            max_route_time=8,
        ),
    ]


@pytest.fixture
def params(tmp_path, sample_goods, sample_configurations):
    depot = DepotLocation(latitude=0.0, longitude=0.0)
    vehicles = {
        cfg.config_id: VehicleSpec(
            capacity=cfg.capacity,
            fixed_cost=cfg.fixed_cost,
            compartments=dict(cfg.compartments),
            avg_speed=cfg.avg_speed,
            service_time=cfg.service_time,
            max_route_time=cfg.max_route_time,
        )
        for cfg in sample_configurations
    }
    problem = ProblemParams(
        vehicles=vehicles,
        depot=depot,
        goods=sample_goods,
        variable_cost_per_hour=10.0,
    )
    algorithm = AlgorithmParams(
        clustering_method="combine",
        geo_weight=0.5,
        demand_weight=0.5,
        route_time_estimation="BHH",
        clustering_max_depth=2,
        small_cluster_size=2,
        nearest_merge_candidates=1,
        pre_small_cluster_size=1,
        pre_nearest_merge_candidates=1,
    )
    io = IOParams(demand_file="dummy", results_dir=tmp_path, format="json")
    runtime = RuntimeParams(config=str(tmp_path / "cfg.yaml"))
    return FleetmixParams(problem=problem, algorithm=algorithm, io=io, runtime=runtime)


def make_cluster(config_id: str, vehicle_type: str, customer_ids: list[str]) -> Cluster:
    return Cluster(
        cluster_id=0,
        config_id=config_id,
        vehicle_type=vehicle_type,
        customers=customer_ids,
        total_demand={"Dry": 1.0, "Chilled": 0.0},
        centroid_latitude=0.0,
        centroid_longitude=0.0,
        goods_in_config=["Dry"],
        route_time=1.0,
        method="test",
    )


def test_generate_feasible_clusters_returns_empty_for_missing_inputs(params):
    assert generate_feasible_clusters([], [], params) == []


def test_generate_feasible_clusters_handles_no_feasible_customers(sample_customers, sample_configurations, params):
    with patch("fleetmix.clustering.generator._generate_feasibility_mapping", return_value={}) as mocked:
        clusters = generate_feasible_clusters(sample_customers, sample_configurations, params)
    mocked.assert_called_once()
    assert clusters == []


def test_generate_feasible_clusters_builds_tsp_matrices(sample_customers, sample_configurations, params):
    params_tsp = replace(
        params,
        algorithm=replace(params.algorithm, route_time_estimation="TSP"),
    )
    with patch("fleetmix.utils.route_time.build_distance_duration_matrices") as mock_builder:
        with patch("fleetmix.clustering.generator.create_initial_clusters", return_value=[[sample_customers[0]]]):
            with patch(
                "fleetmix.clustering.generator.process_clusters_recursively",
                return_value=[make_cluster("1", "SmallVan", ["C1"])],
            ):
                clusters = generate_feasible_clusters(sample_customers, sample_configurations[:1], params_tsp)
    assert clusters
    mock_builder.assert_called()


def test_process_configuration_filters_infeasible(sample_customers, sample_configurations, params):
    feasible = {"C1": ["cfg:1"], "C3": ["cfg:1"]}
    with patch(
        "fleetmix.clustering.generator.get_feasible_customers_subset",
        return_value=[sample_customers[0]],
    ) as subset_mock:
        with patch("fleetmix.clustering.generator.create_initial_clusters", return_value=[[sample_customers[0]]]):
            with patch(
                "fleetmix.clustering.generator.process_clusters_recursively",
                return_value=[make_cluster("1", "SmallVan", ["C1"])],
            ) as process_mock:
                clusters = process_configuration(
                    sample_configurations[0],
                    sample_customers,
                    feasible,
                    CapacitatedClusteringContext(
                        goods=["Dry"],
                        depot=DepotLocation(0, 0),
                        max_depth=2,
                        route_time_estimation="BHH",
                        geo_weight=0.5,
                        demand_weight=0.5,
                    ),
                    {},
                    {},
                    params,
                    "minibatch_kmeans",
                )
    subset_mock.assert_called_once()
    process_mock.assert_called_once()
    assert len(clusters) == 1


def test_deduplicate_clusters_is_vehicle_type_aware():
    clusters = [
        make_cluster("1", "Van", ["C1", "C2"]),
        make_cluster("1", "Van", ["C1", "C2"]),
        make_cluster("1", "Truck", ["C1", "C2"]),
    ]
    for idx, cluster in enumerate(clusters):
        cluster.cluster_id = idx
    unique = _deduplicate_clusters(clusters)
    assert len(unique) == 2
    vehicle_types = {cluster.vehicle_type for cluster in unique}
    assert vehicle_types == {"Van", "Truck"}


def test_generate_feasibility_mapping_filters_by_compartments(sample_customers, sample_configurations, sample_goods):
    configs = [
        VehicleConfiguration(
            config_id=cfg.config_id,
            vehicle_type=cfg.vehicle_type,
            capacity=cfg.capacity,
            fixed_cost=cfg.fixed_cost,
            compartments=dict(cfg.compartments),
            avg_speed=cfg.avg_speed,
            service_time=cfg.service_time,
            max_route_time=cfg.max_route_time,
        )
        for cfg in sample_configurations
    ]
    configs[0].compartments["Chilled"] = False
    mapping = _generate_feasibility_mapping(sample_customers, configs, sample_goods)
    assert "1" not in mapping.get("C2", [])


class MockParams:
    def __init__(self):
        depot = DepotLocation(0, 0)
        vehicles = {
            "1": VehicleSpec(
                capacity=10,
                fixed_cost=5,
                compartments={"Dry": True},
                avg_speed=30,
                service_time=10,
                max_route_time=8,
            )
        }
        algorithm = AlgorithmParams(
            clustering_method="combine",
            geo_weight=0.5,
            demand_weight=0.5,
            route_time_estimation="BHH",
            clustering_max_depth=3,
        )
        problem = ProblemParams(
            vehicles=vehicles,
            depot=depot,
            goods=["Dry"],
            variable_cost_per_hour=1.0,
        )
        io = IOParams(demand_file="d", results_dir=Path("."), format="json")
        runtime = RuntimeParams(config="./cfg")
        self.algorithm = algorithm
        self.problem = problem
        self.io = io
        self.runtime = runtime


def test_get_clustering_context_list_combine():
    params = MockParams()
    contexts = _get_clustering_context_list(params)
    names = [name for _, name in contexts]
    assert len(contexts) == 9  # 3 base + 6 weight combos
    assert {"minibatch_kmeans", "kmedoids", "gaussian_mixture", "agglomerative"}.issubset(set(names))


def test_process_configuration_requires_params(sample_customers, sample_configurations):
    with pytest.raises(ValueError):
        process_configuration(
            sample_configurations[0],
            sample_customers,
            {c.customer_id: ["cfg:1"] for c in sample_customers},
            CapacitatedClusteringContext(
                goods=["Dry"],
                depot=DepotLocation(0, 0),
                max_depth=1,
                route_time_estimation="BHH",
                geo_weight=0.5,
                demand_weight=0.5,
            ),
            {},
            {},
            main_params=None,
        )
