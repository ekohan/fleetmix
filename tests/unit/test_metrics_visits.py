import math

from fleetmix.core_types import Cluster, FleetmixSolution
from fleetmix.experiments.fleet_composition.metrics import (
    _count_customer_visits,
    average_visits_per_customer,
    split_rate,
)


def make_cluster(cluster_id: int, customers: list[str]) -> Cluster:
    # Minimal viable Cluster for metrics
    return Cluster(
        cluster_id=cluster_id,
        config_id="cfg-1",
        vehicle_type="MCV",
        customers=customers,
        total_demand={"Dry": 0.0, "Chilled": 0.0, "Frozen": 0.0},
        centroid_latitude=0.0,
        centroid_longitude=0.0,
        goods_in_config=["Dry", "Chilled", "Frozen"],
        route_time=1.0,
    )


def test_visit_count_deduplicates_within_cluster():
    # Cluster 0 contains the same physical customer A multiple times via pseudo-IDs
    c0 = make_cluster(0, ["A::Dry", "A::Chilled", "B"])  # A must count once here
    # Cluster 1 visits A again and also C
    c1 = make_cluster(1, ["A::Frozen", "C"])  # A counts once more here

    sol = FleetmixSolution(selected_clusters=[c0, c1])

    counts = _count_customer_visits(sol)
    assert counts == {"A": 2, "B": 1, "C": 1}

    # Average visits per physical customer = (2 + 1 + 1) / 3 = 4/3
    avg_visits = average_visits_per_customer(sol)
    assert math.isclose(avg_visits, 4.0 / 3.0, rel_tol=1e-9)

    # Split rate: customers with >1 visit divided by total = 1/3
    sr = split_rate(sol)
    assert math.isclose(sr, 1.0 / 3.0, rel_tol=1e-9)


