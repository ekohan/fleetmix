# fleetmix package

# Export API function
from .api import optimize

# Export canonical clustering function
from .clustering import generate_clusters_for_configurations

# Export domain objects
from .core_types import Cluster, Customer

# Export canonical VRP interface functions
from .pipeline.vrp_interface import VRPType, convert_to_fsm, run_optimization

# Version
__version__ = "0.1.0b1"

# Export core modules
from . import clustering, config, optimization, post_optimization, utils

# Public API
__all__ = [
    "Cluster",
    "Customer",
    "VRPType",
    "__version__",
    "clustering",
    "config",
    "convert_to_fsm",
    "generate_clusters_for_configurations",
    "optimization",
    "optimize",
    "post_optimization",
    "run_optimization",
    "utils",
]
