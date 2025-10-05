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
class CustomerBase:
    """Individual customer with location and demand."""
    customer_id: str
    latitude: float
    longitude: float
    demands: dict[str, float]  # product_type -> quantity

@dataclass
class Customer(CustomerBase):
    """Extended customer with origin tracking."""
    origin_id: str  # For multi-stop tracking
```

### Vehicle Configuration

```python
@dataclass
class VehicleConfiguration:
    """A vehicle type with specific compartment setup."""
    config_id: str
    vehicle_type: str
    capacity: float  # Total capacity (kg)
    fixed_cost: float
    compartments: dict[str, bool]  # product_type -> active
    avg_speed: float  # km/h
    service_time: float  # minutes per stop
    max_route_time: float  # hours
```

### Cluster

```python
@dataclass
class Cluster:
    """Feasible customer cluster."""
    cluster_id: str
    config_id: str  # Which vehicle config this is for
    method: str  # Clustering method used
    customer_ids: list[str]
    total_demand: dict[str, float]  # Per product type
    route_time: float  # Estimated hours
    total_cost: float  # Fixed + variable
    centroid_lat: float
    centroid_lon: float
```

### Solution

```python
@dataclass
class FleetmixSolution:
    """Optimization solution."""
    total_cost: float
    total_fixed_cost: float
    total_variable_cost: float
    selected_clusters: pd.DataFrame
    vehicles_used: dict[str, int]  # config_id -> count
    solver_status: str
    solve_time_seconds: float
    optimality_gap: float
```

### Context Objects

```python
@dataclass
class CapacitatedClusteringContext:
    """Context for clustering algorithms."""
    vehicle_config: VehicleConfiguration
    depot: DepotLocation
    goods: list[str]
    geo_weight: float  # λ in distance metric
    demand_weight: float  # 1-λ
    route_time_estimation: str  # "bhh" or "tsp"

@dataclass
class RouteTimeContext:
    """Context for route time estimation."""
    depot: DepotLocation
    avg_speed: float
    service_time: float
    max_route_time: float
    prune_tsp: bool
```

## Type System Philosophy

FleetMix uses:
1. **Dataclasses**: For data containers with clear structure
2. **Protocols**: For behavior contracts (see [protocols.md](protocols.md))
3. **Type hints**: Throughout for clarity and MyPy checking
4. **Immutability**: Where practical (using `frozen=True`)

## See Also

- [protocols.md](protocols.md): Behavioral interfaces
- [All module specs](../README.md#-module-specifications): Usage of these types

---

**Navigation**: [← Protocols](protocols.md) | [↑ Specs Index](../README.md#-module-specifications)

