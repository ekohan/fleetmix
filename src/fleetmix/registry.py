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

__all__ = [
    "register_clusterer",
    "register_route_time_estimator",
    "register_solver_adapter",
    # Expose registries for advanced users who need direct access
    "CLUSTERER_REGISTRY",
    "ROUTE_TIME_ESTIMATOR_REGISTRY",
    "SOLVER_ADAPTER_REGISTRY",
]


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
