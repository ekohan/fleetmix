from enum import Enum
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class VRPSolution:
    """Results from VRP solver."""
    total_cost: float
    fixed_cost: float
    variable_cost: float
    total_distance: float
    num_vehicles: int
    routes: List[List[int]]
    vehicle_loads: List[float]
    execution_time: float
    solver_status: str
    route_sequences: List[List[str]]  # List of customer sequences per route
    vehicle_utilization: List[float]  # Capacity utilization per route
    vehicle_types: List[int]  # Vehicle type index per route
    route_times: List[float]
    route_distances: List[float]
    route_feasibility: List[bool]  # New field to track which routes exceed constraints

class BenchmarkType(Enum):
    """Types of VRP benchmarks."""
    SINGLE_COMPARTMENT = "single_compartment"  # Upper bound - Separate VRPs per product
    MULTI_COMPARTMENT = "multi_compartment"    # Lower bound - Aggregate demand, post-process for compartments 