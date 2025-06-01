from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional, Tuple, Any

import pandas as pd
from fleetmix.utils.time_measurement import TimeMeasurement

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
    total_vehicles: int = 0
    missing_customers: Set[str] = field(default_factory=empty_set_factory)
    solver_status: str = "Unknown"
    solver_name: str = "Unknown"
    solver_runtime_sec: float = 0.0
    time_measurements: Optional[List[TimeMeasurement]] = None

    def __post_init__(self):
        """
        Post-initialization to calculate derived fields or perform validation.
        For example, ensuring total_cost is consistent.
        """
        if (self.total_fixed_cost != 0.0 or \
            self.total_variable_cost != 0.0 or \
            self.total_penalties != 0.0) and \
           self.total_cost == 0.0:
            self.total_cost = self.total_fixed_cost + self.total_variable_cost + self.total_penalties

@dataclass
class VehicleSpec:
    capacity: int
    fixed_cost: float
    compartments: Dict[str, bool] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        if item == 'compartments':
            return self.compartments
        if hasattr(self, item):
            return getattr(self, item)
        if item in self.extra:
            return self.extra[item]
        raise KeyError(f"'{item}' not found in VehicleSpec or its extra fields")

    def to_dict(self) -> Dict[str, Any]:
        data = {"capacity": self.capacity, "fixed_cost": self.fixed_cost, "compartments": self.compartments}
        data.update(self.extra)
        return data

@dataclass
class DepotLocation:
    latitude: float
    longitude: float

    def __getitem__(self, key: str) -> float:
        if key == 'latitude':
            return self.latitude
        elif key == 'longitude':
            return self.longitude
        else:
            raise KeyError(f"Invalid key for DepotLocation: {key}")

    def as_tuple(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)