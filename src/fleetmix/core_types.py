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

@dataclass
class Customer:
    """Represents a single customer with their demands."""
    customer_id: str
    demands: Dict[str, float]  # e.g., {'dry': 10, 'chilled': 5}
    location: Tuple[float, float]  # (latitude, longitude)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> List['Customer']:
        """Convert DataFrame to list of Customer objects."""
        customers = []
        for _, row in df.iterrows():
            # Extract demands from columns ending with '_Demand'
            demand_cols = [col for col in df.columns if col.endswith('_Demand')]
            demands = {}
            for col in demand_cols:
                good_name = col.replace('_Demand', '').lower()
                demands[good_name] = row[col]
            
            customer = Customer(
                customer_id=row['Customer_ID'],
                demands=demands,
                location=(row.get('Latitude', 0.0), row.get('Longitude', 0.0))  # Handle missing location
            )
            customers.append(customer)
        return customers

    @staticmethod
    def to_dataframe(customers: List['Customer']) -> pd.DataFrame:
        """Convert list of Customer objects to DataFrame."""
        if len(customers) == 0:
            # Return DataFrame with proper schema
            return pd.DataFrame(columns=['Customer_ID', 'Latitude', 'Longitude'])
        
        # Determine all goods from all customers
        all_goods = set()
        for customer in customers:
            all_goods.update(customer.demands.keys())
        all_goods = sorted(all_goods)
        
        data = []
        for customer in customers:
            row = {
                'Customer_ID': customer.customer_id,
                'Latitude': customer.location[0],
                'Longitude': customer.location[1]
            }
            # Add demand columns
            for good in all_goods:
                col_name = f'{good.title()}_Demand'
                row[col_name] = customer.demands.get(good, 0.0)
            data.append(row)
        
        return pd.DataFrame(data)


@dataclass
class PseudoCustomer:
    """Represents a pseudo-customer for split-stop capability.
    
    A pseudo-customer represents a subset of goods that a physical customer needs,
    allowing the physical customer to be served by multiple vehicles.
    """
    customer_id: str  # Format: "original_id::good1-good2-..."
    origin_id: str    # Original physical customer ID
    subset: Tuple[str, ...]  # Tuple of goods this pseudo-customer represents
    demands: Dict[str, float]  # Demand vector (only for goods in subset)
    location: Tuple[float, float]  # Same location as original customer
    service_time: float = 25.0  # Service time in minutes


def empty_dataframe_factory():
    """Ensures a new empty DataFrame is created for default."""
    return pd.DataFrame()

def empty_dict_factory():
    """Ensures a new empty dict is created for default."""
    return {}

def empty_set_factory():
    """Ensures a new empty set is created for default."""
    return set()

def empty_list_factory():
    """Ensures a new empty list is created for default."""
    return []


@dataclass
class VehicleOperationContext:
    """Base context for vehicle operations - shared operational parameters."""
    depot: 'DepotLocation'


@dataclass
class ClusteringContext(VehicleOperationContext):
    """Context for customer clustering algorithms."""
    goods: List[str]
    max_depth: int
    route_time_estimation: str
    geo_weight: float
    demand_weight: float


@dataclass
class RouteTimeContext(VehicleOperationContext):
    """Context for route time estimation algorithms."""
    avg_speed: float  # km/h
    service_time: float  # minutes per customer
    max_route_time: float  # hours
    prune_tsp: bool = False
    
    def __post_init__(self):
        """Allow max_route_time to be optional for route time estimation."""
        # For route time estimation, max_route_time might be None during estimation
        pass


@dataclass
class Cluster:
    """Represents a cluster of customers that can be served by a vehicle configuration."""
    cluster_id: int
    config_id: int
    customers: List[str]
    total_demand: Dict[str, float]
    centroid_latitude: float
    centroid_longitude: float
    goods_in_config: List[str]
    route_time: float
    method: str = ''
    tsp_sequence: List[str] = field(default_factory=empty_list_factory)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> List['Cluster']:
        """Convert DataFrame to list of Cluster objects."""
        clusters = []
        for _, row in df.iterrows():
            # Handle TSP_Sequence column if it exists
            tsp_sequence = []
            if 'TSP_Sequence' in df.columns and row['TSP_Sequence'] is not None:
                tsp_sequence = row['TSP_Sequence'] if isinstance(row['TSP_Sequence'], list) else []
            
            # Extract goods_in_config if available, otherwise derive from Total_Demand
            goods_in_config = []
            if 'Goods_In_Config' in df.columns and row['Goods_In_Config'] is not None:
                goods_in_config = row['Goods_In_Config'] if isinstance(row['Goods_In_Config'], list) else []
            elif 'Total_Demand' in df.columns and isinstance(row['Total_Demand'], dict):
                goods_in_config = [good for good, demand in row['Total_Demand'].items() if demand > 0]
            
            cluster = Cluster(
                cluster_id=row['Cluster_ID'],
                config_id=row.get('Config_ID', 'unassigned'),  # Handle missing Config_ID
                customers=row['Customers'] if isinstance(row['Customers'], list) else [],
                total_demand=row['Total_Demand'] if isinstance(row['Total_Demand'], dict) else {},
                centroid_latitude=row.get('Centroid_Latitude', 0.0),  # Handle missing centroids
                centroid_longitude=row.get('Centroid_Longitude', 0.0),  # Handle missing centroids
                goods_in_config=goods_in_config,
                route_time=row['Route_Time'],
                method=row.get('Method', ''),
                tsp_sequence=tsp_sequence
            )
            clusters.append(cluster)
        return clusters

    @staticmethod
    def to_dataframe(clusters: List['Cluster']) -> pd.DataFrame:
        """Convert list of Cluster objects to DataFrame."""
        if len(clusters) == 0:
            return pd.DataFrame()
        
        data = []
        for cluster in clusters:
            row = {
                'Cluster_ID': cluster.cluster_id,
                'Config_ID': cluster.config_id,
                'Customers': cluster.customers,
                'Total_Demand': cluster.total_demand,
                'Centroid_Latitude': cluster.centroid_latitude,
                'Centroid_Longitude': cluster.centroid_longitude,
                'Goods_In_Config': cluster.goods_in_config,
                'Route_Time': cluster.route_time,
                'Method': cluster.method
            }
            # Only add TSP_Sequence if it exists and is not empty
            if cluster.tsp_sequence:
                row['TSP_Sequence'] = cluster.tsp_sequence
            data.append(row)
        
        return pd.DataFrame(data)

    def to_dict(self) -> Dict:
        """Convert cluster to dictionary format."""
        data = {
            'Cluster_ID': self.cluster_id,
            'Config_ID': self.config_id,
            'Customers': self.customers,
            'Total_Demand': self.total_demand,
            'Centroid_Latitude': self.centroid_latitude,
            'Centroid_Longitude': self.centroid_longitude,
            'Goods_In_Config': self.goods_in_config,
            'Route_Time': self.route_time,
            'Method': self.method
        }
        # Only add sequence if it exists
        if self.tsp_sequence:
            data['TSP_Sequence'] = self.tsp_sequence
        return data

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
    avg_speed: float = 30.0  # km/h - default value
    service_time: float = 25.0  # minutes per customer - default value
    max_route_time: float = 10.0  # hours - default value
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
        data = {
            "capacity": self.capacity, 
            "fixed_cost": self.fixed_cost, 
            "compartments": self.compartments,
            "avg_speed": self.avg_speed,
            "service_time": self.service_time,
            "max_route_time": self.max_route_time
        }
        data.update(self.extra)
        return data

@dataclass
class VehicleConfiguration:
    """Represents a specific vehicle configuration with compartment assignments."""
    config_id: int
    vehicle_type: str
    capacity: int
    fixed_cost: float
    compartments: Dict[str, bool]
    avg_speed: float = 30.0  # km/h - default value
    service_time: float = 25.0  # minutes per customer - default value
    max_route_time: float = 10.0  # hours - default value

    def __getitem__(self, key: str) -> Any:
        """Support bracket notation access for backward compatibility."""
        if key == 'Config_ID':
            return self.config_id
        elif key == 'Vehicle_Type':
            return self.vehicle_type
        elif key == 'Capacity':
            return self.capacity
        elif key == 'Fixed_Cost':
            return self.fixed_cost
        elif key in self.compartments:
            return 1 if self.compartments[key] else 0
        else:
            raise KeyError(f"'{key}' not found in VehicleConfiguration")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = {
            'Config_ID': self.config_id,
            'Vehicle_Type': self.vehicle_type,
            'Capacity': self.capacity,
            'Fixed_Cost': self.fixed_cost,
            'avg_speed': self.avg_speed,
            'service_time': self.service_time,
            'max_route_time': self.max_route_time
        }
        # Add compartment flags
        for good, has_compartment in self.compartments.items():
            data[good] = 1 if has_compartment else 0
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