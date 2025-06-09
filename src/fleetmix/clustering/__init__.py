"""
Customer clustering for the cluster-first heuristic (§4.2).
"""

from fleetmix.core_types import (
    Cluster,
    ClusteringContext,
)

from .generator import (
    _is_customer_feasible,
    generate_clusters_for_configurations,
)
from .heuristics import (
    compute_composite_distance,
    estimate_num_initial_clusters,
    get_cached_demand,
)

__all__ = [
    "Cluster",
    "ClusteringContext",
    "_is_customer_feasible",
    "compute_composite_distance",
    "estimate_num_initial_clusters",
    "generate_clusters_for_configurations",
    "get_cached_demand",
]
