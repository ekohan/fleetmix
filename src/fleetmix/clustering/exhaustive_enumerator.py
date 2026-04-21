"""
Exhaustive cluster enumeration for optimality gap analysis.

Instead of using heuristic clustering algorithms, this module enumerates
**all** feasible customer subsets for each vehicle configuration. When fed
to the MILP, the resulting solution is provably optimal for the
set-partitioning formulation — removing the heuristic gap introduced by
the clustering phase.

This is only tractable for small instances (n ≤ ~15 customers) since the
number of subsets grows as 2^n.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from fleetmix.config.params import FleetmixParams
from fleetmix.core_types import (
    Cluster,
    Customer,
    CustomerBase,
    VehicleConfiguration,
)
from fleetmix.registry import ROUTE_TIME_ESTIMATOR_REGISTRY
from fleetmix.utils.logging import FleetmixLogger
from fleetmix.utils.route_time import make_rt_context

logger = FleetmixLogger.get_logger(__name__)

__all__ = ["generate_exhaustive_clusters"]


def generate_exhaustive_clusters(
    customers: list[CustomerBase],
    configurations: list[VehicleConfiguration],
    params: FleetmixParams,
) -> list[Cluster]:
    """Enumerate all feasible (customer-subset, vehicle-config) pairs.

    For every non-empty subset S ⊆ customers and every configuration v,
    check capacity + compartment compatibility + route-time feasibility.
    If all pass, create a :class:`Cluster` and add it to the pool.

    Parameters
    ----------
    customers:
        All customers in the instance.
    configurations:
        All vehicle configurations to consider.
    params:
        FleetMix parameters (used for goods list, route-time method, etc.).

    Returns
    -------
    list[Cluster]
        Complete set of feasible clusters for the MILP.
    """
    n = len(customers)
    goods = params.problem.goods
    rt_method = params.algorithm.route_time_estimation

    FleetmixLogger.detail(
        f"Exhaustive enumeration: {n} customers, {len(configurations)} configs, "
        f"2^{n}-1 = {2**n - 1} subsets to check"
    )

    # Pre-build customer lookup and DataFrame rows for fast slicing
    customer_lookup: dict[str, CustomerBase] = {c.customer_id: c for c in customers}
    customers_df = Customer.to_dataframe(customers)
    customers_df = customers_df.set_index("Customer_ID", drop=False)

    # Pre-compute per-customer demands for fast subset aggregation
    customer_demands: dict[str, dict[str, float]] = {}
    for c in customers:
        customer_demands[c.customer_id] = {g: c.demands.get(g, 0.0) for g in goods}

    # Route-time estimator
    estimator_class = ROUTE_TIME_ESTIMATOR_REGISTRY.get(rt_method)
    if estimator_class is None:
        raise ValueError(f"Unknown route time estimation method: {rt_method}")
    estimator = estimator_class()

    cluster_id = 0
    all_clusters: list[Cluster] = []
    feasibility_checks = 0
    feasible_count = 0

    customer_ids = [c.customer_id for c in customers]

    for size in range(1, n + 1):
        for subset_ids in itertools.combinations(customer_ids, size):
            subset_set = set(subset_ids)

            # Aggregate demand for this subset
            subset_demand: dict[str, float] = {g: 0.0 for g in goods}
            for cid in subset_ids:
                for g in goods:
                    subset_demand[g] += customer_demands[cid][g]
            total_demand = sum(subset_demand.values())

            # Determine which goods this subset actually needs
            required_goods = {g for g in goods if subset_demand[g] > 0}

            # Compute centroid
            subset_customers = [customer_lookup[cid] for cid in subset_ids]
            centroid_lat = sum(c.location[0] for c in subset_customers) / size
            centroid_lon = sum(c.location[1] for c in subset_customers) / size

            # Prepare DataFrame slice for route-time estimation (computed once
            # per subset, reused across configs with matching speed/service_time)
            subset_df = customers_df.loc[list(subset_ids)]

            # Cache route time per (avg_speed, service_time, max_route_time)
            # since different configs may share these parameters
            rt_cache: dict[tuple[float, float, float], tuple[float, list[str]]] = {}

            for config in configurations:
                feasibility_checks += 1

                # 1. Compartment compatibility
                if not required_goods.issubset(
                    g for g in goods if config.compartments.get(g, False)
                ):
                    continue

                # 2. Capacity check
                if total_demand > config.capacity:
                    continue

                # 3. Route-time check
                rt_key = (config.avg_speed, config.service_time, config.max_route_time)
                if rt_key not in rt_cache:
                    rt_context = make_rt_context(
                        config,
                        params.problem.depot,
                        params.algorithm.prune_tsp,
                    )
                    route_time, tsp_seq = estimator.estimate_route_time(
                        subset_df, rt_context
                    )
                    rt_cache[rt_key] = (route_time, tsp_seq)

                route_time, tsp_seq = rt_cache[rt_key]

                if route_time > config.max_route_time:
                    continue

                # All checks passed — create cluster
                feasible_count += 1
                cluster_id += 1
                cluster = Cluster(
                    cluster_id=cluster_id,
                    config_id=config.config_id,
                    vehicle_type=config.vehicle_type,
                    customers=list(subset_ids),
                    total_demand=dict(subset_demand),
                    centroid_latitude=centroid_lat,
                    centroid_longitude=centroid_lon,
                    goods_in_config=[
                        g for g in goods if config.compartments.get(g, False)
                    ],
                    route_time=route_time,
                    method="exhaustive",
                    tsp_sequence=tsp_seq,
                )
                all_clusters.append(cluster)

    FleetmixLogger.detail(
        f"Exhaustive enumeration complete: {feasibility_checks} checks, "
        f"{feasible_count} feasible clusters generated"
    )
    return all_clusters
