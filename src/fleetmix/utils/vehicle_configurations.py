import itertools
import pandas as pd
from typing import List, Dict, Any
from fleetmix.core_types import VehicleSpec, VehicleConfiguration

def generate_vehicle_configurations(vehicle_types: Dict[str, VehicleSpec], goods: List[str]) -> List[VehicleConfiguration]:
    """
    Enumerate every feasible vehicle–compartment combination (paper §4.4).
    """
    compartment_options = list(itertools.product([0, 1], repeat=len(goods)))
    configurations: List[VehicleConfiguration] = []
    config_id = 1
    
    for vt_name, vt_info in vehicle_types.items():
        for option in compartment_options:
            # Skip configuration if no compartments are selected
            if sum(option) == 0:
                continue
            
            # Create compartments dictionary
            compartments = {good: bool(option[i]) for i, good in enumerate(goods)}
            
            # Create VehicleConfiguration object
            config = VehicleConfiguration(
                config_id=config_id,
                vehicle_type=vt_name,
                capacity=vt_info.capacity,
                fixed_cost=vt_info.fixed_cost,
                compartments=compartments
            )
            configurations.append(config)
            config_id += 1

    return configurations