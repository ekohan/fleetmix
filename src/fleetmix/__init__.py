# fleetmix package 

# Export API function
from .api import optimize

# Export canonical VRP interface functions
from .pipeline.vrp_interface import VRPType, convert_to_fsm, run_optimization

# Export canonical clustering function
from .clustering import generate_clusters_for_configurations

# Export domain objects
from .core_types import Customer, Cluster

# Version
__version__ = "0.1.0b1"

# Export core modules
from . import optimization
from . import clustering
from . import post_optimization
from . import config
from . import utils 

# Public API
__all__ = [
    "optimize",
    "VRPType", 
    "convert_to_fsm",
    "run_optimization", 
    "generate_clusters_for_configurations",
    "Customer",
    "Cluster",
    "optimization", 
    "clustering",
    "post_optimization",
    "config",
    "utils",
    "__version__"
] 