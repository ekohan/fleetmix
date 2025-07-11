"""
Converters for VRP benchmark instances into FSM format.

This package provides converters to translate various Vehicle Routing Problem (VRP) benchmark
instances into the Fleet Size and Mix (FSM) format used by this library.
"""

from .cvrp import CVRPBenchmarkType, convert_cvrp_to_fsm
from .mcvrp import convert_mcvrp_to_fsm

__all__ = ["CVRPBenchmarkType", "convert_cvrp_to_fsm", "convert_mcvrp_to_fsm"]
