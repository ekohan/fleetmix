"""
MILP core for Fleet-Size-and-Mix optimisation (see §4.3 in the paper).
"""

# Re-export public functions from core
from .core import (
    _calculate_cluster_cost,
    _calculate_solution_statistics,
    _create_model,
    _extract_solution,
    _solve_internal,
    _validate_solution,
    optimize_fleet,
)

__all__ = [
    "_calculate_cluster_cost",
    "_calculate_solution_statistics",
    "_create_model",
    "_extract_solution",
    "_solve_internal",
    "_validate_solution",
    "optimize_fleet",
]
