import itertools
import pandas as pd
from typing import List, Dict, Any
from fleetmix.internal_types import VehicleSpec
from fleetmix.types import VehicleConfiguration


def generate_vehicle_configurations(vehicle_types: Dict[str, VehicleSpec], goods: List[str]) -> List[VehicleConfiguration]:
    """
    Enumerate every feasible vehicle–compartment combination (paper §4.4).
    
    Returns a list of VehicleConfiguration objects for the public API.
    """
    df = _generate_vehicle_configurations_df(vehicle_types, goods)
    
    # Convert DataFrame to list of VehicleConfiguration objects
    configurations = []
    for _, row in df.iterrows():
        compartments = {good: bool(row[good]) for good in goods}
        config = VehicleConfiguration(
            config_id=int(row['Config_ID']),
            vehicle_type=row['Vehicle_Type'],
            compartments=compartments,
            capacity=int(row['Capacity']),
            fixed_cost=float(row['Fixed_Cost']),
            avg_speed=float(row['avg_speed']),
            service_time=float(row['service_time']),
            max_route_time=float(row['max_route_time'])
        )
        configurations.append(config)
    
    return configurations


def _generate_vehicle_configurations_df(vehicle_types: Dict[str, VehicleSpec], goods: List[str]) -> pd.DataFrame:
    """
    Internal function that returns a DataFrame for backward compatibility.
    """
    compartment_options = list(itertools.product([0, 1], repeat=len(goods)))
    compartment_configs: List[Dict[str, Any]] = []
    config_id = 1
    
    for vt_name, vt_info in vehicle_types.items():
        for option in compartment_options:
            # Skip configuration if no compartments are selected
            if sum(option) == 0:
                continue
            compartment = dict(zip(goods, option))
            compartment['Vehicle_Type'] = vt_name
            compartment['Config_ID'] = config_id
            compartment['Capacity'] = vt_info.capacity
            compartment['Fixed_Cost'] = vt_info.fixed_cost
            # Operational parameters per vehicle
            compartment['avg_speed'] = vt_info.avg_speed
            compartment['service_time'] = vt_info.service_time
            compartment['max_route_time'] = vt_info.max_route_time
            compartment_configs.append(compartment)
            config_id += 1

    return pd.DataFrame(compartment_configs)