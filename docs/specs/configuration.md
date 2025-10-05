# Configuration System

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

FleetMix uses YAML configuration files to define fleet characteristics, optimization parameters, and algorithmic choices. This enables parameter sweeps, sensitivity analysis, and reproducibility without code changes.

## Configuration Schema

### Complete Example

```yaml
# Fleet configuration
vehicles:
  TruckA:
    capacity: 2700  # kg
    fixed_cost: 100  # daily cost
    avg_speed: 30  # km/h
    service_time: 25  # minutes per stop
    max_route_time: 10  # hours
    allowed_goods: ["Dry", "Chilled"]  # Optional constraint
  
  TruckB:
    capacity: 4500
    fixed_cost: 225
    avg_speed: 30
    service_time: 25
    max_route_time: 12
    # No allowed_goods = can carry all

# Product types
goods:
  - Dry
  - Chilled
  - Frozen

# Depot location
depot:
  latitude: 4.6097
  longitude: -74.0817

# Optimization settings
optimization:
  solver: gurobi  # or 'cbc', 'cplex'
  time_limit: 300  # seconds
  mip_gap: 0.01  # 1% optimality gap acceptable
  improvement_enabled: true
  max_improvement_iterations: 5

# Clustering settings
clustering:
  methods:
    - minibatch_kmeans
    - kmedoids
    - gaussian_mixture
    - agglomerative
  
  geo_weight: [1.0, 0.8]  # λ values to try
  demand_weight: [0.0, 0.2]  # 1-λ
  
  n_clusters_range:
    - auto  # Automatic based on capacity
    - 10
    - 15
  
  recursive_split:
    max_depth: 5
  
  merge:
    min_cluster_size: 3
    max_distance: 10.0  # km

# Route time estimation
route_time:
  method: bhh  # or 'tsp'
  
  bhh:
    beta: 0.75  # BHH constant
  
  tsp:
    max_iterations: 10000
    population_size: 50

# Costs
costs:
  variable_cost_per_hour: 10  # Operating cost per hour
  compartment_setup_cost: 50  # Per active compartment
```

### Parameter Reference

| Section | Parameter | Type | Default | Description |
|---------|-----------|------|---------|-------------|
| `vehicles` | `capacity` | float | Required | Max load (kg) |
| | `fixed_cost` | float | Required | Daily vehicle cost |
| | `allowed_goods` | list[str] | null | Restrict compartments |
| `optimization` | `solver` | str | "cbc" | MILP solver |
| | `time_limit` | int | 300 | Max seconds |
| | `improvement_enabled` | bool | true | Run Phase 4 |
| `clustering` | `methods` | list[str] | all | Which algorithms |
| | `geo_weight` | list[float] | [1.0] | λ in distance metric |
| `route_time` | `method` | str | "bhh" | Estimation approach |

## Loading Configuration

```python
from fleetmix.config.loader import load_config

params = load_config("config.yaml")
```

## Validation

Configuration is validated on load:
- Required fields present
- Types correct
- Values in valid ranges
- Cross-field consistency (e.g., `geo_weight + demand_weight` combinations)

## Paper Connection

**Baseline parameters** (Paper §6, Table):
```yaml
# Bogotá case study baseline
vehicles:
  TypeA:
    capacity: 2700
    fixed_cost: 100
    # ... (from paper Table)
```

## See Also

- [Paper §6](../../paper/main.tex): Baseline parameter values
- [REPRODUCIBILITY.md](../REPRODUCIBILITY.md): Reproducing paper experiments

---

**Navigation**: [← Data Model](data_model.md) | [↑ Specs Index](../README.md#-module-specifications)

