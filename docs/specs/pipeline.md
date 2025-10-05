# Pipeline Orchestration

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

The pipeline module orchestrates the complete matheuristic flow from customer demand to optimal fleet design. It connects all four phases (configuration generation, clustering, optimization, improvement) into a single, cohesive execution.

## Paper Connection

- **Primary Reference**: Paper §4 "Matheuristic Approach" (overall flow)
- **Figure**: Figure 1 (methodological overview diagram)
- **Sections**: Combines §4.1, §4.2, §4.3, §4.4

## Dataflow

```
Input: Demand Data + Configuration
    ↓
Phase 1: Generate Vehicle Configurations (§4.1)
    → Set V of configurations
    ↓
Phase 2: Generate Feasible Clusters (§4.2)
    → Set K of clusters (for each v ∈ V)
    ↓
Phase 3: Optimize Fleet Size and Mix (§4.3)
    → Initial solution (selected clusters, costs)
    ↓
Phase 4: Improvement Phase (§4.4)
    → Enhanced solution
    ↓
Output: Fleet Design Solution
```

## Key Functions

### Main Entry Point

```python
def optimize_fleet_pipeline(
    customers: list[CustomerBase],
    params: FleetmixParams,
) -> FleetmixSolution:
    """
    Complete fleet design pipeline.
    
    Returns:
        FleetmixSolution with optimal fleet composition
    """
```

### Phase Orchestration

Each phase is called sequentially with appropriate data transformations between phases.

## Implementation Notes

- **Primary Module**: `src/fleetmix/pipeline/vrp_interface.py`
- **Entry Points**: 
  - Python API: `fleetmix.api.optimize()`
  - CLI: `fleetmix optimize`
  - GUI: `fleetmix gui`

## Error Handling

- Validates input data
- Checks intermediate results (e.g., empty cluster set)
- Logs warnings and errors
- Returns informative error messages

## Performance Monitoring

- Times each phase
- Logs progress at key milestones
- Reports bottlenecks

## References

### Related Modules

- [vehicle_configurations.md](vehicle_configurations.md)
- [clustering.md](clustering.md)
- [optimization.md](optimization.md)
- [post_optimization.md](post_optimization.md)

---

**Navigation**: [← Post-Optimization](post_optimization.md) | [↑ Specs Index](../README.md#-module-specifications) | [Architecture](../ARCHITECTURE.md)

