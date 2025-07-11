"""Protocol definitions for pluggable components in FleetMix."""

from typing import Protocol

import pandas as pd
import pulp

from fleetmix.config.params import RuntimeParams
from fleetmix.core_types import CapacitatedClusteringContext, RouteTimeContext

__all__ = [
    "Clusterer",
    "RouteTimeEstimator",
    "SolverAdapter",
]


class Clusterer(Protocol):
    """Protocol for clustering algorithms.

    Implementation note: fit() returns cluster labels (List[int]) to maintain
    compatibility with sklearn-style fit_predict() pattern used throughout the codebase.
    """

    def fit(
        self,
        customers: pd.DataFrame,
        *,
        context: CapacitatedClusteringContext,
        n_clusters: int,
    ) -> list[int]:
        """Cluster customers into n_clusters groups. Returns cluster labels."""
        ...


class RouteTimeEstimator(Protocol):
    """Protocol for route time estimation algorithms."""

    def estimate_route_time(
        self,
        cluster_customers: pd.DataFrame,
        context: RouteTimeContext,
    ) -> tuple[float, list[str]]:
        """Returns (route_time_hours, sequence)"""
        ...


class SolverAdapter(Protocol):
    """Thin wrapper around PuLP solvers to provide a consistent interface."""

    def get_pulp_solver(self, params: RuntimeParams) -> pulp.LpSolver:
        """Return the underlying PuLP solver instance configured and ready to use."""
        ...

    @property
    def name(self) -> str:
        """Solver name for logging."""
        ...

    @property
    def available(self) -> bool:
        """Check if this solver is available in the environment."""
        ...
