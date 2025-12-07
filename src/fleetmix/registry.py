"""Registry for pluggable components in FleetMix."""

from typing import Callable, TypeVar

from fleetmix.utils.logging import FleetmixLogger

from .interfaces import Clusterer, RouteTimeEstimator, SolverAdapter

logger = FleetmixLogger.get_logger(__name__)

# Type variables for decorator types
ClustererType = TypeVar("ClustererType", bound=type[Clusterer])
RouteTimeEstimatorType = TypeVar(
    "RouteTimeEstimatorType", bound=type[RouteTimeEstimator]
)
SolverAdapterType = TypeVar("SolverAdapterType", bound=type[SolverAdapter])

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


def register_clusterer(name: str) -> Callable[[ClustererType], ClustererType]:
    """Decorator to register a clusterer implementation."""

    def decorator(cls: ClustererType) -> ClustererType:
        if name in CLUSTERER_REGISTRY:
            raise ValueError(f"Clusterer '{name}' is already registered")
        CLUSTERER_REGISTRY[name] = cls
        return cls

    return decorator


def register_route_time_estimator(
    name: str,
) -> Callable[[RouteTimeEstimatorType], RouteTimeEstimatorType]:
    """Decorator to register a route time estimator implementation."""

    def decorator(cls: RouteTimeEstimatorType) -> RouteTimeEstimatorType:
        if name in ROUTE_TIME_ESTIMATOR_REGISTRY:
            raise ValueError(f"Route time estimator '{name}' is already registered")
        ROUTE_TIME_ESTIMATOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_solver_adapter(
    name: str,
) -> Callable[[SolverAdapterType], SolverAdapterType]:
    """Decorator to register a solver adapter implementation."""

    def decorator(cls: SolverAdapterType) -> SolverAdapterType:
        if name in SOLVER_ADAPTER_REGISTRY:
            raise ValueError(f"Solver adapter '{name}' is already registered")
        SOLVER_ADAPTER_REGISTRY[name] = cls
        return cls

    return decorator
