# Configuration System

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

FleetMix uses YAML configuration files to define fleet characteristics, optimization parameters, and algorithmic choices. This enables parameter sweeps, sensitivity analysis, and reproducibility without code changes.

## Configuration Schema

### Complete Example

```yaml
# ---------------------------------------------------------------------------
# PROBLEM params
# ---------------------------------------------------------------------------

# Vehicle Types and Capacities
vehicles:
  A:
    capacity: 2700  # kg
    fixed_cost: 100  # daily cost
    avg_speed: 30  # km/h
    service_time: 25  # minutes per customer
    max_route_time: 10  # hours
  B:
    capacity: 3300
    fixed_cost: 175
    avg_speed: 30
    service_time: 25
    max_route_time: 10
    allowed_goods:  # Optional constraint
      - Chilled
      - Frozen
  C:
    capacity: 4500
    fixed_cost: 225
    avg_speed: 30
    service_time: 25
    max_route_time: 10
    allowed_goods:
      - Frozen

# Cost Parameters
variable_cost_per_hour: 10.00

# Location
depot:
  latitude: 4.7
  longitude: -74.1

# Product Types
goods:
  - Dry
  - Chilled
  - Frozen

# Math model Parameters
light_load_penalty: 0  # Penalty value -- 0 == don't penalize
light_load_threshold: 0  # Threshold for light load (e.g., 0.2 for 20%)
compartment_setup_cost: 0  # Cost per additional compartment beyond the first

# Split-stop capability
allow_split_stops: true  # Allow customers to be served by multiple vehicles

# ---------------------------------------------------------------------------
# ALGORITHM params
# ---------------------------------------------------------------------------

# Clustering Parameters
clustering:
  max_depth: 20
  method: combine  # Options: minibatch_kmeans, kmedoids, agglomerative, gaussian_mixture, combine
  distance: euclidean  # Options: euclidean, composite (only for agglomerative)
  geo_weight: 0.7  # Weight for geographical distance (composite-only)
  demand_weight: 0.3  # Weight for demand distance (composite-only)
  route_time_estimation: 'BHH'  # Options: TSP, BHH

prune_tsp: false  # Skip TSP based on BHH estimate

# Post-optimization
post_optimization: true  # Enable iterative improvement phase
small_cluster_size: 1000  # ≤ customers per "small" cluster
nearest_merge_candidates: 1000  # max neighbour clusters to probe
max_improvement_iterations: 20  # max iterations for iterative post-optimization

# Merge phase pre-MILP
pre_small_cluster_size: 5
pre_nearest_merge_candidates: 50

# ---------------------------------------------------------------------------
# IO params
# ---------------------------------------------------------------------------

# Data Files
demand_file: "sales_2024-07-02_demand.csv"

# Output format
format: 'json'  # Options: json, xlsx, csv

# ---------------------------------------------------------------------------
# RUNTIME params
# ---------------------------------------------------------------------------

verbose: true  # Enable verbose output
debug: false  # Enable debug mode for solver and model
solver: gurobi  # auto | gurobi | cbc
gap_rel: 0.005  # relative MIP gap
time_limit: 60  # seconds; None = no limit
```

### Parameter Reference

| Section | Parameter | Type | Default | Description |
|---------|-----------|------|---------|-------------|
| `vehicles` | `capacity` | int | Required | Max load (kg) |
| | `fixed_cost` | float | Required | Daily vehicle cost |
| | `avg_speed` | float | Required | km/h |
| | `service_time` | float | Required | minutes per customer |
| | `max_route_time` | float | Required | hours |
| | `allowed_goods` | list[str] | null | Restrict compartments |
| Problem | `variable_cost_per_hour` | float | Required | Operating cost |
| | `compartment_setup_cost` | float | 0.0 | Cost per compartment |
| | `allow_split_stops` | bool | false | Multi-stop delivery |
| `clustering` | `method` | str | "combine" | Clustering algorithm |
| | `max_depth` | int | 20 | Recursive split depth |
| | `geo_weight` | float | 0.7 | Geographic weight (λ) |
| | `route_time_estimation` | str | "BHH" | BHH or TSP |
| Algorithm | `post_optimization` | bool | true | Improvement phase |
| | `max_improvement_iterations` | int | 20 | Max iterations |
| Runtime | `solver` | str | "auto" | MILP solver |
| | `time_limit` | int | null | Max seconds |
| | `gap_rel` | float | 0.0 | Relative MIP gap |

## Loading Configuration

```python
from fleetmix.config.loader import load_yaml

params = load_yaml("config.yaml")
```

## Validation

Configuration is validated on load:
- Required fields present (`vehicles`, `depot`, `goods`, `variable_cost_per_hour`, `demand_file`)
- Types correct
- Values in valid ranges
- Cross-field consistency:
  - `geo_weight + demand_weight` must equal 1.0
  - `allowed_goods` must reference goods in global `goods` list
  - No duplicate goods or allowed_goods entries




---

**Navigation**: [← Data Model](data_model.md) | [↑ Specs Index](../README.md#-module-specifications)

