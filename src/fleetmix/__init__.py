"""FleetMix: Fleet Size and Mix Optimizer."""

__version__ = "0.1.0b1"

# Main API
from .api import optimize
from .clustering.generator import generate_feasible_clusters

# Core types
from .config.params import FleetmixParams
from .core_types import (
    Cluster,
    Customer,
    DepotLocation,
    FleetmixSolution,
    VehicleConfiguration,
    VehicleSpec,
)
from .interfaces import Clusterer, RouteTimeEstimator, SolverAdapter
from .optimization.core import optimize_fleet

# VRP/Benchmarking
from .pipeline.vrp_interface import VRPType, convert_to_fsm, run_optimization
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
    # Version
    "__version__",
    # Main API
    "optimize",
    # Stage functions
    "load_demand",
    "generate_vehicle_configurations",
    "generate_feasible_clusters",
    "optimize_fleet",
    "improve_solution",
    # Types
    "FleetmixParams",
    "FleetmixSolution",
    "VehicleConfiguration",
    "VehicleSpec",
    "Cluster",
    "Customer",
    "DepotLocation",
    # VRP
    "VRPType",
    "convert_to_fsm",
    "run_optimization",
    # Extensions
    "register_clusterer",
    "register_route_time_estimator",
    "register_solver_adapter",
    "Clusterer",
    "RouteTimeEstimator",
    "SolverAdapter",
]
