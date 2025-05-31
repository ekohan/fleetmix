from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple

import pandas as pd

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

# Example of a core type, expand as needed
@dataclass
class Customer:
    """Represents a single customer with their demands."""
    customer_id: str
    demands: Dict[str, float]  # e.g., {'dry': 10, 'chilled': 5}
    location: Optional[Tuple[float, float]] = None  # (latitude, longitude)


def empty_dataframe_factory():
    """Ensures a new empty DataFrame is created for default."""
    return pd.DataFrame()

def empty_dict_factory():
    """Ensures a new empty dict is created for default."""
    return {}

def empty_set_factory():
    """Ensures a new empty set is created for default."""
    return set()


@dataclass
class FleetmixSolution:
    """
    Represents the solution of a fleet optimization problem.
    """
    selected_clusters: pd.DataFrame = field(default_factory=empty_dataframe_factory)
    total_fixed_cost: float = 0.0
    total_variable_cost: float = 0.0
    total_penalties: float = 0.0
    total_light_load_penalties: float = 0.0
    total_compartment_penalties: float = 0.0
    total_cost: float = 0.0
    vehicles_used: Dict[str, int] = field(default_factory=empty_dict_factory)
    total_vehicles: int = 0 # Added field based on _calculate_solution_statistics
    missing_customers: Set[str] = field(default_factory=empty_set_factory)
    solver_status: str = "Unknown"
    solver_name: str = "Unknown"
    solver_runtime_sec: float = 0.0
    post_optimization_runtime_sec: Optional[float] = None # Changed to Optional
    time_measurements: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """
        Post-initialization to calculate derived fields or perform validation.
        For example, ensuring total_cost is consistent.
        """
        # Ensuring total_cost is correctly calculated if its components are provided
        # and total_cost itself isn't directly set to a different value.
        # If total_cost is explicitly passed, this won't override it unless
        # it's the default 0.0 and other costs are non-zero.
        if (self.total_fixed_cost != 0.0 or \
            self.total_variable_cost != 0.0 or \
            self.total_penalties != 0.0) and \
           self.total_cost == 0.0: # only recalculate if total_cost appears to be default
            self.total_cost = self.total_fixed_cost + self.total_variable_cost + self.total_penalties 