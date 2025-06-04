"""
MILP core for Fleet-Size-and-Mix optimisation (see §4.3 in the paper).
"""

# Re-export public functions from core
from .core import (
    optimize_fleet_selection,
    _create_model,
    _extract_solution,
    _validate_solution,
    _calculate_solution_statistics,
    _calculate_cluster_cost
)

__all__ = [
    'optimize_fleet_selection',
    '_create_model',
    '_extract_solution',
    '_validate_solution',
    '_calculate_solution_statistics',
    '_calculate_cluster_cost'
] 