"""
Benchmark datasets, converters and wrappers used in §5–6 experiments.
"""

# Import model classes
from .models import MCVRPInstance, CVRPInstance, CVRPSolution

# Import parser functions and converter functions
from .parsers.mcvrp import parse_mcvrp
from .parsers.cvrp import CVRPParser
from .converters.mcvrp import convert_mcvrp_to_fsm
from .converters.cvrp import convert_cvrp_to_fsm, CVRPBenchmarkType
from .solvers import VRPSolver

__all__ = [
    # Models
    "MCVRPInstance",
    "CVRPInstance", 
    "CVRPSolution",
    # Parsers
    "parse_mcvrp",
    "CVRPParser",
    # Converters
    "convert_mcvrp_to_fsm",
    "convert_cvrp_to_fsm",
    "CVRPBenchmarkType",
    # Solvers
    "VRPSolver"
]
