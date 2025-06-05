"""
Converter for MCVRP instances into FSM format.
"""

__all__ = ["convert_mcvrp_to_fsm"]

from pathlib import Path
from typing import Union

import pandas as pd
from fleetmix.benchmarking.parsers.mcvrp import parse_mcvrp
from fleetmix.config.parameters import Parameters
from fleetmix.core_types import VehicleSpec, DepotLocation
from fleetmix.utils.coordinate_converter import CoordinateConverter

def convert_mcvrp_to_fsm(instance_name: str, custom_instance_path: Path = None) -> tuple:
    """Convert an MCVRP *.dat* file to Fleetmix inputs.

    Parameters
    ----------
    instance_name : str
        Name of the MCVRP instance (e.g., 'pr01').
    custom_instance_path : Path, optional
        Full path to the instance .dat file if not in default dataset location.

    Returns
    -------
    pd.DataFrame
        Customer demand table.
    Parameters
        Parameter set pre-filled with depot, capacity, and expected vehicles.

    Raises
    ------
    FileNotFoundError
        If the instance file does not exist.
    ValueError
        If mandatory headers are missing.
    """
    if custom_instance_path:
        file_path = custom_instance_path
    else:
        file_path = Path(__file__).parent.parent / 'datasets' / 'mcvrp' / f'{instance_name}.dat'

    if not file_path.exists():
        raise FileNotFoundError(f"MCVRP instance file not found: {file_path} for instance '{instance_name}'. Provide custom_instance_path if using non-standard location.")

    # Parse the MCVRP instance
    instance = parse_mcvrp(file_path)

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
    params.depot = DepotLocation(latitude=depot_lat, longitude=depot_lon)
    # Single multi-compartment vehicle
    params.vehicles = {
        'MCVRP': VehicleSpec(
            capacity=instance.capacity,
            fixed_cost=1000,
            compartments={'Dry': True, 'Chilled': True, 'Frozen': True},
            extra={},
            avg_speed=30.0,  # Default speed for benchmarking
            service_time=25.0,  # Default service time
            max_route_time=float('inf')  # No time limit for benchmarking
        )
    }
    # Expected vehicles from instance
    params.expected_vehicles = instance.vehicles

    return customers_df, params 