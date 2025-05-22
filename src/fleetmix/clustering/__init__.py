"""
clustering module

This module provides functions and classes for clustering customers in the fleet optimization process.
"""

# Re-export public functions and classes from generator, heuristics, and common modules
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