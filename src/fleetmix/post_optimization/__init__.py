"""
Merge-phase improvement after initial MILP (paper §4.4).
"""

from .merge_phase import (
    # Other functions that might be used externally TODO: remove?
    generate_merge_phase_clusters,
    # Main public function
    improve_solution,
    validate_merged_cluster,
)

__all__ = [
    "generate_merge_phase_clusters",
    "improve_solution",
    "validate_merged_cluster",
]
