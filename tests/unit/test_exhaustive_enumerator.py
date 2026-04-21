"""Unit tests for exhaustive cluster enumeration."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fleetmix.clustering.exhaustive_enumerator import generate_exhaustive_clusters
from fleetmix.config.params import (
    AlgorithmParams,
    FleetmixParams,
    IOParams,
    ProblemParams,
    RuntimeParams,
)
from fleetmix.core_types import (
    Cluster,
    CustomerBase,
    DepotLocation,
    VehicleConfiguration,
    VehicleSpec,
)


class DummyCustomer(CustomerBase):
    def is_pseudo_customer(self) -> bool:
        return False

    def get_origin_id(self) -> str:
        return self.customer_id

    def get_goods_subset(self) -> tuple[str, ...]:
        return tuple(self.demands.keys())


@pytest.fixture
def goods() -> list[str]:
    return ["Dry", "Chilled"]


@pytest.fixture
def customers() -> list[CustomerBase]:
    return [
        DummyCustomer("C1", {"Dry": 5.0, "Chilled": 0.0}, (0.0, 0.0), 10),
        DummyCustomer("C2", {"Dry": 3.0, "Chilled": 2.0}, (0.01, 0.01), 10),
        DummyCustomer("C3", {"Dry": 4.0, "Chilled": 0.0}, (0.02, 0.0), 10),
    ]


@pytest.fixture
def configurations(goods: list[str]) -> list[VehicleConfiguration]:
    return [
        VehicleConfiguration(
            config_id="1",
            vehicle_type="Van",
            capacity=15,
            fixed_cost=100,
            compartments={g: True for g in goods},
            avg_speed=40,
            service_time=15,
            max_route_time=8,
        ),
        VehicleConfiguration(
            config_id="2",
            vehicle_type="Truck",
            capacity=25,
            fixed_cost=150,
            compartments={"Dry": True, "Chilled": False},
            avg_speed=40,
            service_time=10,
            max_route_time=8,
        ),
    ]


@pytest.fixture
def params(tmp_path, goods, configurations) -> FleetmixParams:
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
        for cfg in configurations
    }
    problem = ProblemParams(
        vehicles=vehicles,
        depot=depot,
        goods=goods,
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


def test_empty_customer_list(configurations, params):
    clusters = generate_exhaustive_clusters([], configurations, params)
    assert clusters == []


def test_enumerates_all_feasible_subsets(customers, configurations, params):
    clusters = generate_exhaustive_clusters(customers, configurations, params)
    # 3 customers → 2^3 - 1 = 7 non-empty subsets. All fit capacity.
    # Config 1 (dual compartment) accepts every subset (7).
    # Config 2 (dry only) rejects subsets touching C2 (needs Chilled): {C2},
    # {C1,C2}, {C2,C3}, {C1,C2,C3} — so 7 - 4 = 3 subsets accepted.
    config_ids = [c.config_id for c in clusters]
    assert config_ids.count("1") == 7
    assert config_ids.count("2") == 3
    assert all(isinstance(c, Cluster) for c in clusters)
    assert all(c.method == "exhaustive" for c in clusters)


def test_capacity_filter(customers, configurations, params):
    tiny = replace(configurations[0], capacity=4)
    clusters = generate_exhaustive_clusters(customers, [tiny], params)
    # Only single-customer subsets fit; C1 has 5 demand, over capacity.
    # C2 total=5 over cap, C3=4 ok. Singletons feasible: just {C3}.
    assert len(clusters) == 1
    assert clusters[0].customers == ["C3"]


def test_compartment_filter(customers, params):
    dry_only = VehicleConfiguration(
        config_id="D",
        vehicle_type="DryOnly",
        capacity=50,
        fixed_cost=100,
        compartments={"Dry": True, "Chilled": False},
        avg_speed=40,
        service_time=15,
        max_route_time=8,
    )
    clusters = generate_exhaustive_clusters(customers, [dry_only], params)
    # Subsets containing C2 (which needs Chilled) are rejected.
    for c in clusters:
        assert "C2" not in c.customers


def test_route_time_filter(customers, configurations, params):
    impossible = replace(configurations[0], max_route_time=0.01, service_time=60)
    clusters = generate_exhaustive_clusters(customers, [impossible], params)
    assert clusters == []


def test_unknown_estimator_raises(customers, configurations, params):
    bad = replace(
        params,
        algorithm=replace(params.algorithm, route_time_estimation="not-a-method"),
    )
    with pytest.raises(ValueError, match="Unknown route time estimation"):
        generate_exhaustive_clusters(customers, configurations, bad)
