"""
Customer clustering for the cluster-first heuristic (§4.2).
"""

# Import only what we need to re-export
from .generator import generate_feasible_clusters

__all__ = [
    "generate_feasible_clusters",
]
