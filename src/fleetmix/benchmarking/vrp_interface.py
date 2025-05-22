"""
Backwards compat shim: re-export VRP interface from the pipeline module.
"""
from fleetmix.pipeline.vrp_interface import VRPType, convert_to_fsm, run_optimization

__all__ = ["VRPType", "convert_to_fsm", "run_optimization"] 