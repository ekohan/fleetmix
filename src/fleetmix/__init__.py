# fleetmix package 

# Export API function
from .api import optimize

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
    "optimization", 
    "clustering",
    "post_optimization",
    "config",
    "utils",
    "__version__"
] 