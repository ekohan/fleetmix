"""
Converter for MCVRP instances into FSM format.
"""

__all__ = ["convert_mcvrp_to_fsm"]

from pathlib import Path
from typing import Union

import pandas as pd
from fleetmix.benchmarking.parsers.mcvrp import parse_mcvrp
from fleetmix.config.parameters import Parameters
from fleetmix.utils.coordinate_converter import CoordinateConverter

def convert_mcvrp_to_fsm(path: Union[str, Path]) -> tuple:
    """
    Convert a MCVRP .dat instance into FSM customer DataFrame and Parameters.
    Args:
        path: Path to the .dat MCVRP instance file.
    Returns:
        A tuple of (customers_df, parameters).
    """
    # Parse the MCVRP instance
    instance = parse_mcvrp(path)

    # Convert coordinates to geospatial coordinates
    converter = CoordinateConverter(instance.coords)
    geo_coords = converter.convert_all_coordinates(instance.coords)

    # Build customer records
    customers = []
    for node_id, (lat, lon) in geo_coords.items():
        if node_id == instance.depot_id:
            continue
        dry, chilled, frozen = instance.demands[node_id]
        customers.append({
            'Customer_ID': str(node_id),
            'Latitude': lat,
            'Longitude': lon,
            'Dry_Demand': dry,
            'Chilled_Demand': chilled,
            'Frozen_Demand': frozen
        })
    customers_df = pd.DataFrame(customers)

    # Create parameters clone
    params = Parameters.from_yaml()
    # Set depot location
    depot_lat, depot_lon = geo_coords[instance.depot_id]
    params.depot = {'latitude': depot_lat, 'longitude': depot_lon}
    # No max route time by default
    params.max_route_time = float('inf')
    # Single multi-compartment vehicle
    params.vehicles = {
        'MCVRP': {
            'capacity': instance.capacity,
            'fixed_cost': 1000,
            'compartments': {'Dry': True, 'Chilled': True, 'Frozen': True}
        }
    }
    # Expected vehicles from instance
    params.expected_vehicles = instance.vehicles

    return customers_df, params 