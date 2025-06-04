# Fleetmix Public API

## Overview

The Fleetmix package now provides a clean, matheuristic-oriented public API that exposes:

1. **Main optimization function** (`optimize`)
2. **Four matheuristic stages** as individual functions
3. **Core data types** for working with solutions

## Public API Reference

### Main Function

```python
from fleetmix import optimize

solution = optimize(
    demand,      # CSV path, Path object, or pandas DataFrame
    config=None, # YAML path, Path object, or Parameters object
    output_dir="results",
    format="excel",
    verbose=False
)
```

### Matheuristic Stages

The four stages of the matheuristic can be called individually:

```python
from fleetmix import (
    generate_vehicle_configurations,
    generate_feasible_clusters,
    optimize_fleet_selection,
    improve_solution,
)

# Stage 1: Generate all vehicle-compartment combinations
configs_df = generate_vehicle_configurations(vehicles, goods)

# Stage 2: Generate feasible customer clusters
clusters_df = generate_feasible_clusters(customers, configs_df, params)

# Stage 3: Solve the fleet selection MILP
solution = optimize_fleet_selection(clusters_df, configs_df, customers_df, params)

# Stage 4: Post-optimization improvement
improved = improve_solution(solution, configs_df, customers_df, params)
```

### Core Types

```python
from fleetmix import VehicleConfiguration, ClusterAssignment, FleetmixSolution

# Vehicle configuration with compartments
config = VehicleConfiguration(
    config_id=1,
    vehicle_type="truck",
    compartments={"dry": True, "frozen": False},
    capacity=100,
    fixed_cost=500.0
)

# Customer cluster assignment
cluster = ClusterAssignment(
    cluster_id=1,
    config_id=1,
    customer_ids=["C1", "C2"],
    route_time=2.5,
    total_demand={"dry": 50, "frozen": 0},
    centroid=(40.7128, -74.0060)
)

# Solution object
solution = FleetmixSolution(
    selected_clusters=[cluster, ...],
    configurations_used=[config, ...],
    total_cost=1000.0,
    total_vehicles=5,
    missing_customers=set(),
    solver_status="Optimal",
    solver_runtime_sec=0.5
)
```

### Parameters

```python
from fleetmix import Parameters

# Load from YAML
params = Parameters.from_yaml("config.yaml")

# Or use default
params = Parameters.from_yaml()
```

## What Changed

### Before (Old API)
- Many internal modules exposed in `__all__`
- Complex internal types with pandas DataFrames
- Mixed concerns (GUI, benchmarking, etc.)

### After (New API)
- Only 10 symbols in `__all__`
- Clean dataclass types for external use
- Focused on the matheuristic workflow
- Internal types moved to `internal_types.py`

## Migration Guide

For existing code:

1. **Imports**: Update any imports of internal types
   ```python
   # Old
   from fleetmix.core_types import FleetmixSolution
   
   # New (for internal use)
   from fleetmix.internal_types import FleetmixSolution
   
   # New (for public use)
   from fleetmix import FleetmixSolution
   ```

2. **Solution Access**: The public `FleetmixSolution` uses lists instead of DataFrames
   ```python
   # Old
   for _, row in solution.selected_clusters.iterrows():
       print(row['Cluster_ID'])
   
   # New
   for cluster in solution.selected_clusters:
       print(cluster.cluster_id)
   ```

3. **Direct Stage Access**: Import stages from top-level
   ```python
   # Old
   from fleetmix.clustering import generate_feasible_clusters
   
   # New
   from fleetmix import generate_feasible_clusters
   ```

## Benefits

1. **Cleaner API**: Only essential functions and types are exposed
2. **Type Safety**: Proper dataclasses instead of dictionaries
3. **Matheuristic-Oriented**: API mirrors the four-stage algorithm
4. **Backward Compatible**: Internal modules still accessible via full paths
5. **Easier Testing**: Clean types are easier to mock and test 