import itertools

from fleetmix.core_types import VehicleConfiguration, VehicleSpec
from fleetmix.utils.common import to_cfg_key


def generate_vehicle_configurations(
    vehicle_types: dict[str, VehicleSpec], goods: list[str]
) -> list[VehicleConfiguration]:
    """
    Enumerate every feasible vehicle–compartment combination (paper §4.4).
    
    If a vehicle has allowed_goods specified, only generate configurations
    for those goods. Otherwise, generate configurations for all global goods.
    """
    configurations: list[VehicleConfiguration] = []
    config_id = 1

    for vt_name, vt_info in vehicle_types.items():
        # Determine which goods this vehicle can carry
        vehicle_goods = vt_info.allowed_goods if vt_info.allowed_goods is not None else goods
        
        # Generate compartment options only for the vehicle's allowed goods
        compartment_options = list(itertools.product([0, 1], repeat=len(vehicle_goods)))
        
        for option in compartment_options:
            # Skip configuration if no compartments are selected
            if sum(option) == 0:
                continue

            # Create compartments dictionary - initialize all goods to False
            compartments = {good: False for good in goods}
            
            # Then set the vehicle's allowed goods based on the option
            for i, good in enumerate(vehicle_goods):
                compartments[good] = bool(option[i])

            # Create VehicleConfiguration object with timing attributes from VehicleSpec
            config = VehicleConfiguration(
                config_id=to_cfg_key(config_id),
                vehicle_type=vt_name,
                capacity=vt_info.capacity,
                fixed_cost=vt_info.fixed_cost,
                compartments=compartments,
                avg_speed=vt_info.avg_speed,
                service_time=vt_info.service_time,
                max_route_time=vt_info.max_route_time,
            )
            configurations.append(config)
            config_id += 1

    return configurations
