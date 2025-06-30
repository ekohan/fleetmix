"""
Unified converter for both CVRP and MCVRP instances to FSM format.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Union

import pandas as pd

from fleetmix.benchmarking.converters import cvrp as _cvrp
from fleetmix.benchmarking.converters import mcvrp as _mcvrp
from fleetmix.benchmarking.converters.cvrp import CVRPBenchmarkType

if TYPE_CHECKING:
    from fleetmix.pipeline.vrp_interface import VRPType
# TODO fix nasty import.

from fleetmix.config.parameters import Parameters

__all__ = ["convert_vrp_to_fsm"]


def convert_vrp_to_fsm(
    vrp_type: Union[str, "VRPType"],
    instance_names: list[str]
    | None = None,  # For CVRP, can be multiple for COMBINED type
    instance_path: str
    | Path
    | None = None,  # For MCVRP (single file) or custom path for single CVRP
    benchmark_type: str | CVRPBenchmarkType | None = None,  # For CVRP
    num_goods: int = 3,  # For CVRP
    split_ratios: dict[str, float] | None = None,  # For CVRP
    custom_instance_paths: dict[str, Path]
    | None = None,  # New: For CVRP with multiple custom paths
) -> tuple[pd.DataFrame, Parameters]:
    """
    Dispatch CVRP/MCVRP conversion to the appropriate converter.
    """
    # Import VRPType here to avoid circular import
    from fleetmix.pipeline.vrp_interface import VRPType

    # Normalize vrp_type
    if not isinstance(vrp_type, VRPType):
        vrp_type = VRPType(vrp_type.lower())

    if vrp_type == VRPType.MCVRP:
        # MCVRP-specific logic
        if instance_path is None:
            raise ValueError("instance_path is required for MCVRP conversion")

        # Extract instance name from path and pass both parameters correctly
        instance_name = Path(instance_path).stem
        return _mcvrp.convert_mcvrp_to_fsm(
            instance_name=instance_name, custom_instance_path=Path(instance_path)
        )

    elif vrp_type == VRPType.CVRP:
        # CVRP-specific logic
        active_custom_paths = {}
        if custom_instance_paths:
            active_custom_paths.update(custom_instance_paths)
        # If instance_path is provided and it's a single CVRP instance, add it to custom_instance_paths
        if instance_path and instance_names and len(instance_names) == 1:
            if Path(instance_path).is_file():  # Make sure it is a file path
                active_custom_paths[instance_names[0]] = Path(instance_path)
            # If instance_path is a directory, it's handled by the test providing full map via custom_instance_paths

        # Ensure instance_names is not None for CVRP
        if instance_names is None:
            raise ValueError("instance_names is required for CVRP conversion")

        # Ensure benchmark_type is not None for CVRP and convert to proper type
        if benchmark_type is None:
            benchmark_type = CVRPBenchmarkType.NORMAL
        elif isinstance(benchmark_type, str):
            # Convert string to CVRPBenchmarkType enum
            benchmark_type = CVRPBenchmarkType(benchmark_type.lower())

        return _cvrp.convert_cvrp_to_fsm(
            instance_names=instance_names,
            benchmark_type=benchmark_type,
            num_goods=num_goods,
            split_ratios=split_ratios,
            custom_instance_paths=active_custom_paths if active_custom_paths else None,
        )
    else:
        raise ValueError(f"Unsupported VRP type: {vrp_type}")
