"""Models for benchmarking functionality."""

from .instance_spec import InstanceSpec
from .models import CVRPInstance, CVRPSolution, MCVRPInstance

__all__ = ["CVRPInstance", "CVRPSolution", "MCVRPInstance", "InstanceSpec"]
