"""
Customer clustering for the cluster-first heuristic (§4.2).
"""

from fleetmix.core_types import (
    Cluster,
    ClusteringContext,
    Customer,
    CustomerBase,
    PseudoCustomer,
)

from .generator import (
    _is_customer_feasible,
    generate_feasible_clusters,
)
from .heuristics import (
    compute_composite_distance,
    estimate_num_initial_clusters,
    get_cached_demand,
)

__all__ = [
    "Cluster",
    "ClusteringContext",
    "Customer",
    "CustomerBase",
    "PseudoCustomer",
    "_is_customer_feasible",
    "compute_composite_distance",
    "estimate_num_initial_clusters",
    "generate_feasible_clusters",
    "get_cached_demand",
]
