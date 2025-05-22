"""
post_optimization module

This module provides functions for improving the solution quality through various post-optimization techniques
such as cluster merging after the initial optimization.
"""

# Re-export public functions from merge_phase
from .merge_phase import (
    # Main public function
    improve_solution,
    
    # Other functions that might be used externally
    generate_merge_phase_clusters,
    validate_merged_cluster
)

# Define what gets imported with "from fleetmix.post_optimization import *"
__all__ = [
    'improve_solution',
    'generate_merge_phase_clusters',
    'validate_merged_cluster'
] 