import itertools
import pandas as pd
from typing import List, Dict, Any
from fleetmix.core_types import VehicleSpec, VehicleConfiguration

def generate_vehicle_configurations(vehicle_types: Dict[str, VehicleSpec], goods: List[str]) -> List[VehicleConfiguration]:
    """
    Enumerate every feasible vehicle–compartment combination (paper §4.4).
    """
    # Hardcoded parameter for maximum number of compartments
    MAX_COMPARTMENTS = 1
    
    compartment_options = list(itertools.product([0, 1], repeat=len(goods)))
    configurations: List[VehicleConfiguration] = []
    config_id = 1
    
    for vt_name, vt_info in vehicle_types.items():
        for option in compartment_options:
            # Count the number of enabled compartments
            num_enabled_compartments = sum(option)
            
            # Skip configuration if no compartments are selected or exceeds MAX_COMPARTMENTS
            if num_enabled_compartments == 0 or num_enabled_compartments > MAX_COMPARTMENTS:
                continue
            
            # Create compartments dictionary
            compartments = {good: bool(option[i]) for i, good in enumerate(goods)}
            
            # Create VehicleConfiguration object with timing attributes from VehicleSpec
            config = VehicleConfiguration(
                config_id=config_id,
                vehicle_type=vt_name,
                capacity=vt_info.capacity,
                fixed_cost=vt_info.fixed_cost,
                compartments=compartments,
                avg_speed=vt_info.avg_speed,
                service_time=vt_info.service_time,
                max_route_time=vt_info.max_route_time
            )
            configurations.append(config)
            config_id += 1

    return configurations