# fleetmix package 

# Version
__version__ = "0.1.0b1"

# Main entry point
from .api import optimize

# Matheuristic stages
from .utils.vehicle_configurations import generate_vehicle_configurations
from .clustering import generate_feasible_clusters
from .optimization import optimize_fleet_selection
from .post_optimization import improve_solution

# Core public types
from .types import (
    VehicleConfiguration,
    ClusterAssignment,
    FleetmixSolution,
)
from .config.parameters import Parameters

# Public API
__all__ = [
    "optimize",
    "generate_vehicle_configurations",
    "generate_feasible_clusters",
    "optimize_fleet_selection",
    "improve_solution",
    "VehicleConfiguration",
    "ClusterAssignment",
    "FleetmixSolution",
    "Parameters",
    "__version__",
] 