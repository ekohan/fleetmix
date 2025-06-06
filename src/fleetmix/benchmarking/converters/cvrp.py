"""
Converter for CVRP instances into FSM format.
"""

__all__ = ["convert_cvrp_to_fsm", "CVRPBenchmarkType"]

from enum import Enum
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
import numpy as np

from fleetmix.config.parameters import Parameters
from fleetmix.core_types import VehicleSpec, DepotLocation
from fleetmix.utils.coordinate_converter import CoordinateConverter
import fleetmix.benchmarking.parsers.cvrp as cvrp_parser
from fleetmix.utils.logging import log_detail, log_debug, log_progress

class CVRPBenchmarkType(Enum):
    """Types of CVRP benchmarks supported."""
    NORMAL = "normal"      # 1 product type
    SPLIT = "split"        # Split original demand across compartments
    SCALED = "scaled"      # Multiply demand by number of goods
    COMBINED = "combined"  # Combine multiple instances to create a multi-good instance


def convert_cvrp_to_fsm(
    instance_names: Union[str, List[str]],
    benchmark_type: CVRPBenchmarkType,
    num_goods: int = 3,
    split_ratios: Dict[str, float] = None,
    custom_instance_paths: Dict[str, Path] = None
) -> tuple:
    """
    Convert CVRP instance(s) to FSM format based on benchmark type.
    """
    if isinstance(instance_names, str):
        instance_names = [instance_names]
        
    if benchmark_type == CVRPBenchmarkType.COMBINED and len(instance_names) < 2:
        raise ValueError("Combined benchmark type requires at least 2 instances")

    # Default split ratios if not provided
    if split_ratios is None:
        if num_goods == 2:
            split_ratios = {'dry': 0.6, 'chilled': 0.4}
        else:
            split_ratios = {'dry': 0.5, 'chilled': 0.3, 'frozen': 0.2}
            
    # Parse instances
    instances = []
    for name in instance_names:
        if custom_instance_paths and name in custom_instance_paths:
            instance_path = custom_instance_paths[name]
        else:
            instance_path = Path(__file__).parent.parent / 'datasets' / 'cvrp' / f'{name}.vrp'
        
        if not instance_path.exists():
            raise FileNotFoundError(f"CVRP instance file not found: {instance_path} for instance '{name}'. Provide custom_instance_paths if using non-standard location.")
            
        parser = cvrp_parser.CVRPParser(str(instance_path))
        instances.append(parser.parse())
        
    # Convert based on benchmark type
    if benchmark_type == CVRPBenchmarkType.NORMAL:
        return _convert_normal(instances[0])
    elif benchmark_type == CVRPBenchmarkType.SPLIT:
        return _convert_split(instances[0], split_ratios)
    elif benchmark_type == CVRPBenchmarkType.SCALED:
        return _convert_scaled(instances[0], num_goods)
    else:
        return _convert_combined(instances)


def _convert_normal(instance) -> tuple:
    """Type 1: Normal conversion - single good (dry)"""
    # Print total demand for debugging
    total_demand = sum(instance.demands.values())
    log_detail(f"Total CVRP demand: {total_demand}")
    log_detail(f"CVRP capacity per vehicle: {instance.capacity}")
    log_detail(f"Minimum theoretical vehicles needed: {total_demand / instance.capacity:.2f}")
    
    customers_data = _create_customer_data(
        instance,
        lambda demand: {'Dry_Demand': demand, 'Chilled_Demand': 0, 'Frozen_Demand': 0}
    )
    
    # Verify converted demand
    df = pd.DataFrame(customers_data)
    total_converted = df['Dry_Demand'].sum()
    log_detail(f"Total converted demand: {total_converted}")
    
    params = _create_base_params(instance)
    params.expected_vehicles = instance.num_vehicles
    
    # Override the default vehicles with just our CVRP vehicle
    params.vehicles = {
        'CVRP': VehicleSpec(
            capacity=instance.capacity,
            fixed_cost=1000,
            compartments={'Dry': True, 'Chilled': False, 'Frozen': False},
            extra={},
            avg_speed=30.0,
            service_time=25.0,
            max_route_time=float('inf')  # No time limit
        )
    }
    
    log_detail(f"\nVehicle Configuration:")
    log_detail(f"Capacity: {instance.capacity}")
    log_detail(f"Fixed Cost: {params.vehicles['CVRP']['fixed_cost']}")
    log_detail(f"Compartments: {params.vehicles['CVRP']['compartments']}")
    
    return pd.DataFrame(customers_data), params


def _convert_split(instance, split_ratios: Dict[str, float]) -> tuple:
    """Type 2: Split demand across goods"""
    # Convert split_ratios keys to match DataFrame column names
    df_split_ratios = {
        f'{good.capitalize()}_Demand': ratio 
        for good, ratio in split_ratios.items()
    }
    
    customers_data = _create_customer_data(
        instance,
        lambda demand: {
            column: demand * ratio 
            for column, ratio in df_split_ratios.items()
        }
    )
    
    params = _create_base_params(instance)
    params.expected_vehicles = instance.num_vehicles
    
    # Override vehicles with just the multi-compartment CVRP vehicle
    params.vehicles = {
        'CVRP_Multi': VehicleSpec(
            capacity=instance.capacity,
            fixed_cost=1000,
            compartments={good: True for good in split_ratios},
            extra={},
            avg_speed=30.0,
            service_time=25.0,
            max_route_time=float('inf')  # No time limit for benchmarking
        )
    }
    
    return pd.DataFrame(customers_data), params


def _convert_scaled(instance, num_goods: int) -> tuple:
    """Type 3: Scale instance for multiple goods - only scale dry goods"""
    customers_data = _create_customer_data(
        instance,
        lambda demand: {
            'Dry_Demand': demand * num_goods,
            'Chilled_Demand': 0,
            'Frozen_Demand': 0
        }
    )
    
    params = _create_base_params(instance)
    # Handle cases where instance.num_vehicles might be None
    base_num_vehicles = instance.num_vehicles if instance.num_vehicles is not None else 1
    params.expected_vehicles = base_num_vehicles * num_goods
    
    # Override vehicles with scaled CVRP vehicle
    params.vehicles = {
        'CVRP_Scaled': VehicleSpec(
            capacity=instance.capacity * num_goods,
            fixed_cost=1000,
            compartments={'Dry': True, 'Chilled': False, 'Frozen': False},
            extra={},
            avg_speed=30.0,
            service_time=25.0,
            max_route_time=float('inf')  # No time limit for benchmarking
        )
    }
    
    return pd.DataFrame(customers_data), params


def _convert_combined(instances: List) -> tuple:
    """Type 4: Combine multiple instances"""
    # Only use as many goods as we have instances
    goods = ['Dry', 'Chilled', 'Frozen'][:len(instances)]
    goods_columns = [f'{good}_Demand' for good in goods]
    
    customers_data = []
    for idx, (instance, good, column) in enumerate(zip(instances, goods, goods_columns)):
        instance_data = _create_customer_data(
            instance,
            lambda demand: {col: demand if col == column else 0 for col in goods_columns}
        )
        for customer in instance_data:
            customer['Customer_ID'] = f"{idx+1}_{customer['Customer_ID']}"
        customers_data.extend(instance_data)
    
    params = _create_base_params(instances[0])  # Use first instance for depot
    params.expected_vehicles = sum(inst.num_vehicles for inst in instances if inst.num_vehicles is not None)
    
    # Create a vehicle type for each instance with its specific capacity and good
    params.vehicles = {
        f'CVRP_{idx+1}': VehicleSpec(
            capacity=instance.capacity,
            fixed_cost=1000,
            compartments={g: (g == good_outer) for g in goods},
            extra={},
            avg_speed=30.0,
            service_time=25.0,
            max_route_time=float('inf')  # No time limit for benchmarking
        )
        for idx, (instance, good_outer) in enumerate(zip(instances, goods))
    }
    
    return pd.DataFrame(customers_data), params


def _create_customer_data(instance, demand_func) -> List[Dict]:
    """Helper to create customer data with given demand function"""
    converter = CoordinateConverter(instance.coordinates)
    geo_coords = converter.convert_all_coordinates(instance.coordinates)
    
    customers_data = []
    for cust_id, coords in geo_coords.items():
        if cust_id != instance.depot_id:
            customer = {
                'Customer_ID': str(cust_id),
                'Latitude': coords[0],
                'Longitude': coords[1],
                'Dry_Demand': 0,
                'Chilled_Demand': 0,
                'Frozen_Demand': 0  # Initialize all demands to 0
            }
            # Update with any non-zero demands from the demand_func
            customer.update(demand_func(instance.demands.get(cust_id, 0)))
            customers_data.append(customer)
            
    return customers_data


def _create_base_params(instance) -> Parameters:
    """Helper to create base parameters"""
    params = Parameters.from_yaml()
    converter = CoordinateConverter(instance.coordinates)
    geo_coords = converter.convert_all_coordinates(instance.coordinates)
    depot_coords = geo_coords[instance.depot_id]
    
    params.depot = DepotLocation(
        latitude=depot_coords[0],
        longitude=depot_coords[1]
    )
    
    return params

