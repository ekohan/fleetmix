"""
Customer clustering for the cluster-first heuristic (§4.2).
"""

from .common import (
    Cluster,
    ClusteringSettings,
)

from .generator import (
    generate_clusters_for_configurations,
    _is_customer_feasible,
)

from .heuristics import (
    compute_composite_distance,
    estimate_num_initial_clusters,
    get_cached_demand,
)

__all__ = [
    'generate_clusters_for_configurations',
    'Cluster',
    'ClusteringSettings',
    'compute_composite_distance',
    'estimate_num_initial_clusters',
    '_is_customer_feasible',
    'get_cached_demand',
] 