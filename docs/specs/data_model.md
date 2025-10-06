# Data Model and Core Types

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

Defines the core data structures used throughout FleetMix. These types ensure type safety, clear contracts between modules, and facilitate testing and documentation.

## Paper Connection

- **Primary Reference**: Paper §3 "Problem Definition" (notation)
- **Implementation**: `src/fleetmix/core_types.py`

## Core Types

### Customer Types

```python
@dataclass
class CustomerBase(ABC):
    """Base class for all customer types."""
    customer_id: str
    demands: dict[str, float]  # product_type -> quantity
    location: tuple[float, float]  # (latitude, longitude)
    service_time: float  # minutes

@dataclass
class Customer(CustomerBase):
    """Regular customer (not split)."""
    # Inherits all fields from CustomerBase
    # Returns self.customer_id for get_origin_id()

@dataclass
class PseudoCustomer(CustomerBase):
    """Pseudo-customer for split-stop capability."""
    origin_id: str  # Original physical customer ID
    subset: tuple[str, ...]  # Goods this pseudo-customer represents
```

### Vehicle Types

```python
@dataclass
class VehicleSpec:
    """Specification for a vehicle type (from config)."""
    capacity: int
    fixed_cost: float
    compartments: dict[str, bool] = field(default_factory=dict)
    avg_speed: float = 30.0  # km/h
    service_time: float = 25.0  # minutes per customer
    max_route_time: float = 10.0  # hours
    allowed_goods: list[str] | None = None  # Optional restriction
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class VehicleConfiguration:
    """A vehicle type with specific compartment setup (generated)."""
    config_id: str  # Unique identifier
    vehicle_type: str  # e.g., "A", "B", "C"
    capacity: int  # Total capacity (kg)
    fixed_cost: float
    compartments: dict[str, bool]  # product_type -> is_active
    avg_speed: float = 30.0  # km/h
    service_time: float = 25.0  # minutes per customer
    max_route_time: float = 10.0  # hours
```

### Cluster

```python
@dataclass
class Cluster:
    """Feasible customer cluster."""
    cluster_id: int
    config_id: str | int  # Which vehicle config this is for
    vehicle_type: str  # Vehicle type that serves this cluster
    customers: list[str]  # Customer IDs in this cluster
    total_demand: dict[str, float]  # Per product type
    centroid_latitude: float
    centroid_longitude: float
    goods_in_config: list[str]  # Goods this configuration can carry
    route_time: float  # Estimated hours
    method: str = ""  # Clustering method used
    tsp_sequence: list[str] = field(default_factory=list)  # Optional TSP sequence
```

### Solution

```python
@dataclass
class FleetmixSolution:
    """Optimization solution."""
    configurations: list[VehicleConfiguration] = field(default_factory=list)
    selected_clusters: list[Cluster] = field(default_factory=list)
    total_fixed_cost: float = 0.0
    total_variable_cost: float = 0.0
    total_penalties: float = 0.0
    total_light_load_penalties: float = 0.0
    total_compartment_penalties: float = 0.0
    total_cost: float = 0.0
    vehicles_used: dict[str, int] = field(default_factory=dict)  # config_id -> count
    total_vehicles: int = 0
    missing_customers: set[str] = field(default_factory=set)
    solver_status: str = "Unknown"
    solver_name: str = "Unknown"
    solver_runtime_sec: float = 0.0
    time_measurements: list[TimeMeasurement] | None = None
    optimality_gap: float | None = None  # Relative gap (%) or None
```

### Location

```python
@dataclass
class DepotLocation:
    """Depot location."""
    latitude: float
    longitude: float
```

### Context Objects

```python
@dataclass
class VehicleOperationContext:
    """Base context for vehicle operations."""
    depot: DepotLocation

@dataclass
class CapacitatedClusteringContext(VehicleOperationContext):
    """Context for clustering algorithms."""
    # Inherits: depot: DepotLocation
    goods: list[str]
    max_depth: int
    route_time_estimation: str  # "BHH" or "TSP"
    geo_weight: float  # λ in distance metric
    demand_weight: float  # 1-λ

@dataclass
class RouteTimeContext(VehicleOperationContext):
    """Context for route time estimation."""
    # Inherits: depot: DepotLocation
    avg_speed: float  # km/h
    service_time: float  # minutes per customer
    max_route_time: float  # hours
    prune_tsp: bool = False
```

## Type System Philosophy

FleetMix uses:
1. **Dataclasses**: For data containers with clear structure
2. **Protocols**: For behavior contracts (see [protocols.md](protocols.md))
3. **Type hints**: Throughout for clarity and static type checking
4. **Immutability**: Configuration parameters use `frozen=True` (ProblemParams, AlgorithmParams, etc.)
5. **ABC**: Abstract base classes where inheritance is needed (CustomerBase)

## See Also

- [protocols.md](protocols.md): Behavioral interfaces
- [All module specs](../README.md#-module-specifications): Usage of these types

---

**Navigation**: [← Protocols](protocols.md) | [↑ Specs Index](../README.md#-module-specifications)

