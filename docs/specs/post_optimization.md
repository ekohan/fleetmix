# Post-Optimization Improvement Phase

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

The improvement phase iteratively merges small selected clusters and re-optimizes until no cost improvement is achieved. This is Phase 4 of the matheuristic.

## Paper Connection

- **Primary Reference**: Paper §4.4 "Improvement Phase"
- **Key insight**: Merges only selected clusters from optimal solution (not all clusters)

## Algorithm

For each iteration until convergence:

1. Extract small clusters from current solution ($\leq$ `small_cluster_size` customers)
2. Find nearest feasible merge pairs (capacity + route time constraints)
3. Add merged clusters to pool and re-optimize
4. If cost improves, continue; else terminate

Terminates when no improvement or `max_improvement_iterations` reached.

## Function

```python
def improve_solution(
    initial_solution: FleetmixSolution,
    configurations: list[VehicleConfiguration],
    customers: list[CustomerBase],
    params: FleetmixParams,
) -> FleetmixSolution:
```

**Key parameters**:
- `params.algorithm.post_optimization`: Enable/disable (default: True)
- `params.algorithm.max_improvement_iterations`: Max iterations (default: 20)
- `params.algorithm.small_cluster_size`: Small cluster threshold (default: 1000)

## Implementation

- **Module**: `src/fleetmix/post_optimization/merge_phase.py`
- **Main function**: `improve_solution()`
- **Dependencies**: `fleetmix.optimization`, `fleetmix.merging.core`

## Usage

```python
from fleetmix.post_optimization import improve_solution

# After initial optimization
improved = improve_solution(initial_sol, configs, customers, params)
```

## See Also

- [optimization.md](optimization.md) - Provides initial solution
- [clustering.md](clustering.md) - Cluster structures
- [pipeline.md](pipeline.md) - Full flow orchestration

---

**Navigation**: [← Optimization](optimization.md) | [↑ Specs Index](../README.md#-module-specifications) | [Pipeline →](pipeline.md)

