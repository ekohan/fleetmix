"""benchmarking package

Provides scripts and parsers that convert *standard VRP benchmark instances* (e.g., CVRP, flexible
MCVRP) into the data structures required by the fleet-design heuristic so researchers can reproduce
Tables 2–3 in Sections 5–6 of the paper.

Key entry points
----------------
All entrypoints are centralized in the `fleetmix/cli` directory.

This package provides benchmark instances and converters that are used by the CLI tools:
• Parsers for benchmark instances
• Converters from standard formats to fleet sizing and mix problem format
• Thin wrapper around PyVRP to obtain lower/upper bounds

All helper functions are re-exported via `__all__` for convenience.
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
