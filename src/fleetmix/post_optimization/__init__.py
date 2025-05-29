"""
Merge-phase improvement after initial MILP (paper §4.4).
"""

from .merge_phase import (
    # Main public function
    improve_solution,
    
    # Other functions that might be used externally TODO: remove?
    generate_merge_phase_clusters,
    validate_merged_cluster
)

__all__ = [
    'improve_solution',
    'generate_merge_phase_clusters',
    'validate_merged_cluster'
] 