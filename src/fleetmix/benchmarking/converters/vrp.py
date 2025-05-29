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
    instance_names: List[str] = None, # For CVRP, can be multiple for COMBINED type
    instance_path: Union[str, Path] = None, # For MCVRP (single file) or custom path for single CVRP
    benchmark_type: Union[str, 'CVRPBenchmarkType'] = None, # For CVRP
    num_goods: int = 3, # For CVRP
    split_ratios: Dict[str, float] = None, # For CVRP
    custom_instance_paths: Dict[str, Path] = None # New: For CVRP with multiple custom paths
) -> tuple[pd.DataFrame, Parameters]:
    """
    Dispatch CVRP/MCVRP conversion to the appropriate converter.
    """
    # avoid circular import at module load
    from fleetmix.pipeline.vrp_interface import VRPType

    # Normalize vrp_type
    if not isinstance(vrp_type, VRPType):
        vrp_type = VRPType(vrp_type.lower())

    if vrp_type == VRPType.MCVRP:
        if not instance_names and instance_path:
            # If instance_path is given, assume it's the name for MCVRP or a single file path
            # The convert_mcvrp_to_fsm expects instance_name and optional custom_instance_path
            mcvrp_name = Path(instance_path).stem
            mcvrp_custom_path = Path(instance_path) if Path(instance_path).is_file() else None
            if mcvrp_custom_path and not mcvrp_custom_path.exists(): # If it was meant to be a path, check it
                 raise FileNotFoundError(f"MCVRP instance file not found: {mcvrp_custom_path}")
            return _mcvrp.convert_mcvrp_to_fsm(instance_name=mcvrp_name, custom_instance_path=mcvrp_custom_path if mcvrp_custom_path else None)
        elif instance_names and isinstance(instance_names, list) and len(instance_names) == 1:
             # If instance_names has one entry, use it as the name for MCVRP
             # Custom path can be passed via instance_path if it's a single file
            mcvrp_custom_path = Path(instance_path) if instance_path and Path(instance_path).is_file() else None
            if mcvrp_custom_path and not mcvrp_custom_path.exists():
                 raise FileNotFoundError(f"MCVRP instance file not found: {mcvrp_custom_path}")
            return _mcvrp.convert_mcvrp_to_fsm(instance_name=instance_names[0], custom_instance_path=mcvrp_custom_path)
        else:
            raise ValueError("For MCVRP, provide a single instance_name via 'instance_names' list (e.g., ['pr01']) or a direct file path via 'instance_path'.")
    elif vrp_type == VRPType.CVRP:
        # CVRP-specific logic
        active_custom_paths = {}
        if custom_instance_paths:
            active_custom_paths.update(custom_instance_paths)
        # If instance_path is provided and it's a single CVRP instance, add it to custom_instance_paths
        if instance_path and instance_names and len(instance_names) == 1:
            if Path(instance_path).is_file(): # Make sure it is a file path
                 active_custom_paths[instance_names[0]] = Path(instance_path)
            # If instance_path is a directory, it's handled by the test providing full map via custom_instance_paths

        return _cvrp.convert_cvrp_to_fsm(
            instance_names=instance_names,
            benchmark_type=benchmark_type,
            num_goods=num_goods,
            split_ratios=split_ratios,
            custom_instance_paths=active_custom_paths if active_custom_paths else None
        )
    else:
        raise ValueError(f"Unsupported VRP type: {vrp_type}") 