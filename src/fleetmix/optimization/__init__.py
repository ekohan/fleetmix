"""
optimization module

This module provides functions for solving the Fleet Size-and-Mix optimization problem.
"""

# Re-export public functions from core
from .core import (
    solve_fsm_problem,
    _create_model,
    _extract_solution,
    _validate_solution,
    _calculate_solution_statistics,
    _calculate_cluster_cost
)

__all__ = [
    'solve_fsm_problem',
    '_create_model',
    '_extract_solution',
    '_validate_solution',
    '_calculate_solution_statistics',
    '_calculate_cluster_cost'
] 