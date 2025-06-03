"""
Customer clustering for the cluster-first heuristic (§4.2).
"""

from fleetmix.core_types import (
    Cluster,
    ClusteringContext,
)

from .generator import (
    generate_feasible_clusters,
    _is_customer_feasible,
)

from .heuristics import (
    compute_composite_distance,
    estimate_num_initial_clusters,
    get_cached_demand,
)

__all__ = [
    'generate_feasible_clusters',
    'Cluster',
    'ClusteringContext',
    'compute_composite_distance',
    'estimate_num_initial_clusters',
    '_is_customer_feasible',
    'get_cached_demand',
] 