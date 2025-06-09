"""Registry for pluggable components in FleetMix."""

import pandas as pd

from fleetmix.utils.logging import FleetmixLogger

from .core_types import ClusteringContext
from .interfaces import Clusterer, RouteTimeEstimator, SolverAdapter

logger = FleetmixLogger.get_logger(__name__)

# Registries for each component type
CLUSTERER_REGISTRY: dict[str, type[Clusterer]] = {}
ROUTE_TIME_ESTIMATOR_REGISTRY: dict[str, type[RouteTimeEstimator]] = {}
SOLVER_ADAPTER_REGISTRY: dict[str, type[SolverAdapter]] = {}


def register_clusterer(name: str):
    """Decorator to register a clusterer implementation."""

    def decorator(cls: type[Clusterer]):
        CLUSTERER_REGISTRY[name] = cls
        return cls

    return decorator


def register_route_time_estimator(name: str):
    """Decorator to register a route time estimator implementation."""

    def decorator(cls: type[RouteTimeEstimator]):
        ROUTE_TIME_ESTIMATOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_solver_adapter(name: str):
    """Decorator to register a solver adapter implementation."""

    def decorator(cls: type[SolverAdapter]):
        SOLVER_ADAPTER_REGISTRY[name] = cls
        return cls

    return decorator


# Special implementation: CombinedClusterer
@register_clusterer("combine")
class CombinedClusterer:
    """Clusterer that combines results from multiple sub-methods."""

    def __init__(self, sub_methods: list[str] | None = None):
        # Default sub_methods if not provided
        self.sub_methods = sub_methods or [
            "minibatch_kmeans",
            "kmedoids",
            "gaussian_mixture",
        ]

    def fit(
        self, customers: pd.DataFrame, *, context: ClusteringContext, n_clusters: int
    ) -> list[int]:
        """
        Implementation that combines results from multiple clusterers.

        For the 'combine' method, we run multiple clustering algorithms and return
        all their results mapped to unique label ranges. This is handled specially
        in generator.py to create multiple context configurations.

        Note: This is a placeholder since the actual 'combine' logic is handled
        at a higher level in generator.py by creating multiple ClusteringContext objects.
        This implementation should not be called directly.
        """
        logger.warning(
            "CombinedClusterer.fit() called directly - this should be handled by generator.py"
        )
        # Return simple labels as fallback
        return list(range(min(n_clusters, len(customers))))
