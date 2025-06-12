import itertools

from fleetmix.core_types import VehicleConfiguration, VehicleSpec


def generate_vehicle_configurations(
    vehicle_types: dict[str, VehicleSpec], goods: list[str]
) -> list[VehicleConfiguration]:
    """
    Enumerate every feasible vehicle–compartment combination (paper §4.4).
    """
    compartment_options = list(itertools.product([0, 1], repeat=len(goods)))
    configurations: list[VehicleConfiguration] = []
    config_id = 1

    for vt_name, vt_info in vehicle_types.items():
        for option in compartment_options:
            # Skip configuration if no compartments are selected
            if sum(option) == 0:
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
                max_route_time=vt_info.max_route_time,
            )
            configurations.append(config)
            config_id += 1

    return configurations


__all__ = ["generate_vehicle_configurations"]
