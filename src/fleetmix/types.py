"""
Public API types for Fleetmix.

This module contains the core data structures exposed in the public API.
These are simplified versions of the internal types, designed for ease of use.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional


@dataclass
class VehicleConfiguration:
    """A specific vehicle type with compartment assignment."""
    config_id: int
    vehicle_type: str
    compartments: Dict[str, bool]
    capacity: int
    fixed_cost: float
    # Operational parameters (km/h, minutes per stop, max route time in hours)
    avg_speed: float
    service_time: float
    max_route_time: float


@dataclass
class ClusterAssignment:
    """Assignment of customers to a cluster."""
    cluster_id: int
    config_id: int
    customer_ids: List[str]
    route_time: float
    total_demand: Dict[str, float]
    centroid: Tuple[float, float]


@dataclass
class FleetmixSolution:
    """Solution from fleet optimization."""
    selected_clusters: List[ClusterAssignment]
    configurations_used: List[VehicleConfiguration]
    total_cost: float
    total_vehicles: int
    missing_customers: Set[str]
    solver_status: str
    solver_runtime_sec: float
    
    # Additional cost breakdown attributes
    total_fixed_cost: float = 0.0
    total_variable_cost: float = 0.0
    total_penalties: float = 0.0
    
    # Vehicle usage summary
    vehicles_used: Optional[Dict[str, int]] = None