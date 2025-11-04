"""FleetMix: Fleet Size and Mix Optimizer."""

__version__ = "0.1.0b1"

# Main API
from .api import optimize
from .clustering.generator import generate_feasible_clusters

# Core types
from .config.params import FleetmixParams
from .core_types import (
    CapacitatedClusteringContext,
    Cluster,
    Customer,
    DepotLocation,
    FleetmixSolution,
    RouteTimeContext,
    VehicleConfiguration,
    VehicleSpec,
)
from .interfaces import Clusterer, RouteTimeEstimator, SolverAdapter

# Post-optimization
from .post_optimization.merge_phase import improve_solution

# Extension system
from .registry import (
    register_clusterer,
    register_route_time_estimator,
    register_solver_adapter,
)

# Stage functions (for advanced users)
from .utils.data_processing import load_customer_demand as load_demand
from .utils.vehicle_configurations import generate_vehicle_configurations

__all__ = [
    # === Version ===
    "__version__",
    # === Main API (primary entry point) ===
    "optimize",
    # === Pipeline stages (for advanced users) ===
    "load_demand",
    "generate_vehicle_configurations",
    "generate_feasible_clusters",
    "improve_solution",
    # === Core data types (input/output contracts) ===
    "FleetmixParams",
    "FleetmixSolution",
    "VehicleConfiguration",
    "VehicleSpec",
    "Cluster",
    "Customer",
    "DepotLocation",
    # === Extension system (plugin architecture) ===
    # Protocols
    "Clusterer",
    "RouteTimeEstimator",
    "SolverAdapter",
    # Registration decorators
    "register_clusterer",
    "register_route_time_estimator",
    "register_solver_adapter",
    # Context types
    "CapacitatedClusteringContext",
    "RouteTimeContext",
]
