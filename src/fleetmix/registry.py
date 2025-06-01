"""Registry for pluggable components in FleetMix."""
from typing import Dict, Type, List
import pandas as pd
import numpy as np

from .interfaces import Clusterer, RouteTimeEstimator, SolverAdapter
from .clustering.common import ClusteringSettings

from fleetmix.utils.logging import FleetmixLogger
logger = FleetmixLogger.get_logger(__name__)

# Registries for each component type
CLUSTERER_REGISTRY: Dict[str, Type[Clusterer]] = {}
ROUTE_TIME_ESTIMATOR_REGISTRY: Dict[str, Type[RouteTimeEstimator]] = {}
SOLVER_ADAPTER_REGISTRY: Dict[str, Type[SolverAdapter]] = {}


def register_clusterer(name: str):
    """Decorator to register a clusterer implementation."""
    def decorator(cls: Type[Clusterer]):
        CLUSTERER_REGISTRY[name] = cls
        return cls
    return decorator


def register_route_time_estimator(name: str):
    """Decorator to register a route time estimator implementation."""
    def decorator(cls: Type[RouteTimeEstimator]):
        ROUTE_TIME_ESTIMATOR_REGISTRY[name] = cls
        return cls
    return decorator


def register_solver_adapter(name: str):
    """Decorator to register a solver adapter implementation."""
    def decorator(cls: Type[SolverAdapter]):
        SOLVER_ADAPTER_REGISTRY[name] = cls
        return cls
    return decorator


# Special implementation: CombinedClusterer
@register_clusterer("combine")
class CombinedClusterer:
    """Clusterer that combines results from multiple sub-methods."""
    
    def __init__(self, sub_methods: List[str] = None):
        # Default sub_methods if not provided
        self.sub_methods = sub_methods or ['minibatch_kmeans', 'kmedoids', 'gaussian_mixture']
    
    def fit(self, customers: pd.DataFrame, *, settings: ClusteringSettings, n_clusters: int) -> List[int]:
        """
        Implementation that combines results from multiple clusterers.
        
        For the 'combine' method, we run multiple clustering algorithms and return
        all their results mapped to unique label ranges. This is handled specially
        in generator.py to create multiple settings configurations.
        
        Note: This is a placeholder since the actual 'combine' logic is handled
        at a higher level in generator.py by creating multiple ClusteringSettings.
        This implementation should not be called directly.
        """
        logger.warning("CombinedClusterer.fit() called directly - this should be handled by generator.py")
        # Return simple labels as fallback
        return list(range(min(n_clusters, len(customers)))) 