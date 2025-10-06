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

**Public API**: `fleetmix.api.optimize()` - High-level interface for users

**Internal Pipeline**: `fleetmix.pipeline.vrp_interface.run_optimization()` - Core orchestration

### Phase Execution

1. Generate vehicle configurations
2. Apply split-stop preprocessing (if enabled)
3. Generate feasible clusters
4. Solve optimization (MILP)
5. Apply post-optimization improvement (if enabled)

## Implementation Notes

- **Primary Module**: `src/fleetmix/pipeline/vrp_interface.py`
- **API Entry**: `src/fleetmix/api.py` 
- **CLI Entry**: `src/fleetmix/app.py`

### Entry Points

- **Python API**: `import fleetmix; fleetmix.optimize()`
- **CLI**: `fleetmix optimize` 
- **GUI**: `fleetmix gui`

### Features

- Validates input data at each stage
- Times each phase (via `TimeRecorder`)
- Logs progress and warnings
- Returns self-contained `FleetmixSolution` with configurations and time measurements

## References

### Related Modules

- [vehicle_configurations.md](vehicle_configurations.md)
- [clustering.md](clustering.md)
- [optimization.md](optimization.md)
- [post_optimization.md](post_optimization.md)

---

**Navigation**: [← Post-Optimization](post_optimization.md) | [↑ Specs Index](../README.md#-module-specifications) | [Architecture](../ARCHITECTURE.md)

