"""
Unified converter for both CVRP and MCVRP instances to FSM format.
"""
from pathlib import Path
from typing import Union, List, Dict
import pandas as pd
from fleetmix.benchmarking.converters import cvrp as _cvrp
from fleetmix.benchmarking.converters import mcvrp as _mcvrp

from fleetmix.config.parameters import Parameters

__all__ = ["convert_vrp_to_fsm"]

def convert_vrp_to_fsm(
    vrp_type: Union[str, 'VRPType'],
    instance_names: List[str] = None,
    instance_path: Union[str, Path] = None,
    benchmark_type: Union[str, 'CVRPBenchmarkType'] = None,
    num_goods: int = 3,
    split_ratios: Dict[str, float] = None
) -> tuple[pd.DataFrame, Parameters]:
    """
    Unified converter that dispatches to the pipeline's converter functions,
    allowing tests to stub pipeline.convert_cvrp_to_fsm/convert_mcvrp_to_fsm.
    """
    # avoid circular import at module load
    from fleetmix.pipeline.vrp_interface import VRPType

    # Normalize vrp_type
    if not isinstance(vrp_type, VRPType):
        vrp_type = VRPType(vrp_type)

    if vrp_type == VRPType.MCVRP:
        return _mcvrp.convert_mcvrp_to_fsm(instance_path)
    else:
        return _cvrp.convert_cvrp_to_fsm(
            instance_names=instance_names,
            benchmark_type=benchmark_type,
            num_goods=num_goods,
            split_ratios=split_ratios
        ) 