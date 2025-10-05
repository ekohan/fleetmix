# Post-Optimization Improvement Phase

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

The improvement phase iteratively enhances the initial fleet design by identifying pairs of selected clusters that can be merged and re-solving the optimization. This is Phase 4 of the matheuristic and typically reduces costs by 5-15%.

## Paper Connection

- **Primary Reference**: Paper §4.4 "Improvement Phase"
- **Algorithm**: Iterative cluster merging until convergence
- **Key Difference**: Unlike clustering-phase merging (which merges small clusters generally), this merges only **selected** clusters from optimal solution

## Mathematical Formulation

### Concept

Given an initial solution with selected clusters $K^* \subseteq K$, attempt to create new clusters:

$$k_{\text{new}} = k_i \cup k_j, \quad k_i, k_j \in K^*$$

that are feasible and, when added to $K$, enable a better solution when re-solving Problem (P).

### Feasibility

The merged cluster $k_{\text{new}}$ must satisfy:

1. **Capacity**: $\sum_{i \in k_{\text{new}}} \sum_{p \in P} d_{ip} \leq Q_v$ for some $v \in V$
2. **Route time**: $t_{v, k_{\text{new}}} \leq T_v$ for that configuration $v$

### Improvement Criterion

After adding feasible merges to cluster set and re-solving:

$$Z_{\text{new}} < Z_{\text{current}}$$

where $Z$ is the objective value (total cost).

### Termination

Stop when:
- No feasible merges found, OR
- No cost improvement achieved, OR
- Maximum iterations reached

## Design Decisions

### Why Iterative Improvement?

**Alternative**: One-shot merging after optimization

**Our choice**: Iterative until convergence

**Rationale**:
- New merges may enable further merges in next iteration
- Cost reduction compounds
- Computational cost acceptable (typically 2-3 iterations)

### Which Clusters to Merge?

**Strategy**: Merge pairs of clusters from **current optimal solution**

**Rationale**:
- High likelihood both clusters are "good" (selected by optimizer)
- Merging reduces fixed costs (fewer vehicles)
- Variable costs may increase (longer routes) but net effect often positive

**Rejected alternative**: Merge all pairs of small clusters
- Exponential number of candidates
- Most won't be selected anyway
- Wastes computation

### Distance Criterion

Only consider merging $(k_i, k_j)$ if:
- Both selected in current solution
- Centroid-to-centroid distance ≤ $\Delta$ (default: no limit)

**Configurable**: Can restrict to geographically close pairs for speed

## Interfaces

### Input

```python
initial_solution: FleetmixSolution  # From Phase 3 (optimization)
configurations: list[VehicleConfiguration]
customers: list[CustomerBase]
params: FleetmixParams  # Contains max_improvement_iterations, etc.
```

### Output

```python
FleetmixSolution  # Same structure, potentially lower total_cost
```

### Key Parameters

```python
params.optimization.max_improvement_iterations: int = 5
params.optimization.improvement_enabled: bool = True
```

## Key Algorithms

### Main Improvement Loop

**Purpose**: Iteratively merge and re-optimize until no improvement

**Pseudocode**:

```python
def improve_solution(initial_solution, configs, customers, params):
    best_solution = initial_solution
    best_cost = initial_solution.total_cost
    
    for iteration in range(max_iterations):
        # 1. Extract selected clusters
        selected = extract_selected_clusters(best_solution)
        
        # 2. Generate merge candidates
        merged_clusters = []
        for (ki, kj) in all_pairs(selected):
            if is_close_enough(ki, kj):
                k_new = merge(ki, kj)
                if is_feasible(k_new, configs):
                    merged_clusters.append(k_new)
        
        if not merged_clusters:
            break  # No feasible merges
        
        # 3. Re-optimize with expanded cluster set
        all_clusters = original_clusters + merged_clusters
        new_solution = optimize_fleet(all_clusters, configs, customers, params)
        
        # 4. Check improvement
        if new_solution.total_cost < best_cost:
            best_solution = new_solution
            best_cost = new_solution.total_cost
            logger.info(f"Improvement: {best_cost}")
        else:
            break  # No improvement, done
    
    return best_solution
```

**Complexity**: 
- Merge candidates: $O(n^2)$ where $n$ = # selected clusters (typically 20-50)
- Per iteration: 1 MILP solve (1-5s)
- Total: 2-3 iterations × 5s = 10-15s

### Merge Feasibility Check

**Purpose**: Verify merged cluster satisfies constraints

**Steps**:

1. **Combine customer sets**:
```python
k_new.customer_ids = ki.customer_ids ∪ kj.customer_ids
```

2. **Sum demands**:
```python
k_new.demand[p] = ki.demand[p] + kj.demand[p] for all p
```

3. **Find feasible configuration**:
```python
for v in configurations:
    if sum(k_new.demand) <= v.capacity:
        # Check route time
        t_new = estimate_route_time(k_new, v)
        if t_new <= v.max_route_time:
            return True, v
return False, None
```

4. **Compute cost** (if feasible):
```python
k_new.total_cost = v.fixed_cost + variable_rate * t_new
```

### Caching

**Route time cache**: Memoize route time for customer sets

```python
cache_key = frozenset(k_new.customer_ids)
if cache_key in route_time_cache:
    return route_time_cache[cache_key]
else:
    t = compute_route_time(k_new)
    route_time_cache[cache_key] = t
    return t
```

**Benefits**: Avoid recomputing route time for repeated customer sets

## Implementation Notes

### Code Organization

- **Primary Module**: `src/fleetmix/post_optimization/merge_phase.py` (204 lines)
- **Key Functions**:
  - `improve_solution()`: Main entry point
  - `_find_merge_candidates()`: Generate merge pairs
  - `_is_merge_feasible()`: Check feasibility
- **Supporting**: `src/fleetmix/merging/core.py` (merge utilities)

### Dependencies

- **Internal**: 
  - `fleetmix.optimization.core` (re-solve)
  - `fleetmix.utils.route_time` (feasibility check)
  - `fleetmix.merging.core` (merge logic)
- **External**: `pandas`

### Performance Considerations

**Typical runtime**:
- Merge candidate generation: < 1s
- Feasibility checking: 1-2s
- Re-solve MILP: 2-5s
- **Per iteration**: 3-7s
- **Total** (2-3 iterations): 6-20s

**Bottleneck**: MILP re-solve (can use warm start to speed up)

**Optimization**: 
- Limit merge candidates (geographic proximity filter)
- Skip if initial solution has very few vehicles

### Edge Cases

1. **No improvement**: Returns original solution unchanged
2. **All merges infeasible**: Loop terminates immediately
3. **Infinite loop**: Protected by max_iterations parameter
4. **Numerical issues**: Cost comparisons use tolerance (e.g., 0.01)

## Usage Examples

### Basic Usage

```python
from fleetmix.post_optimization.merge_phase import improve_solution

# After initial optimization
initial_sol = optimize_fleet(clusters, configs, customers, params)

# Improve
improved_sol = improve_solution(initial_sol, configs, customers, params)

print(f"Initial cost: ${initial_sol.total_cost:,.2f}")
print(f"Improved cost: ${improved_sol.total_cost:,.2f}")
print(f"Savings: ${initial_sol.total_cost - improved_sol.total_cost:,.2f}")
```

### Controlling Iterations

```yaml
# config.yaml
optimization:
  improvement_enabled: true
  max_improvement_iterations: 5  # More iterations for better solution
```

### Analyzing Improvement

```python
def analyze_improvement(initial, improved):
    """Analyze what changed."""
    print(f"Vehicles before: {sum(initial.vehicles_used.values())}")
    print(f"Vehicles after: {sum(improved.vehicles_used.values())}")
    
    # Which configs changed?
    for cfg in set(initial.vehicles_used.keys()) | set(improved.vehicles_used.keys()):
        before = initial.vehicles_used.get(cfg, 0)
        after = improved.vehicles_used.get(cfg, 0)
        if before != after:
            print(f"{cfg}: {before} → {after}")
```

## Extension Points

### Custom Merge Criteria

To add domain-specific merge rules:

```python
def should_merge(ki, kj, params):
    """Custom logic for merge candidacy."""
    # Example: Only merge if same-day delivery
    if ki.delivery_date != kj.delivery_date:
        return False
    
    # Example: Only merge if geographic proximity
    if distance(ki.centroid, kj.centroid) > max_distance:
        return False
    
    return True
```

### Multi-Level Merging

To merge 3+ clusters:

```python
# After pairwise merges
for (ki, kj, kl) in all_triples(selected):
    k_new = merge(ki, kj, kl)
    if is_feasible(k_new):
        candidates.append(k_new)
```

**Trade-off**: Exponentially more candidates, diminishing returns

### Penalty-Based Improvement

To optimize secondary objectives:

```python
# After cost improvement stagnates
if iterations_without_improvement >= 2:
    # Switch to minimizing vehicle count
    objective = num_vehicles + alpha * cost
```

## Testing

### Unit Tests

- **Location**: `tests/unit/test_post_optimization.py`
- **Coverage**:
  - Merge feasibility logic
  - Cost improvement detection
  - Iteration termination

### Integration Tests

- **Location**: `tests/integration/test_improvement_phase.py`
- **Scenarios**:
  - End-to-end improvement
  - No improvement case
  - Multiple iterations

## Comparison with Literature

### Ostermeier & Hübner (2018)

**Their approach**: Integrated into ALNS (destroy-repair with merge operators)

**FleetMix**: Post-optimization phase

**Similarity**: Both use iterative improvement

**Difference**: They integrate routing, we work with clusters

### Local Search Metaheuristics

Classic **Local Search** framework:
1. Start with solution
2. Generate neighborhood (nearby solutions)
3. Move to better neighbor
4. Repeat until local optimum

**FleetMix improvement phase** follows this pattern:
- **Current**: Selected clusters
- **Neighborhood**: All feasible merges
- **Move**: Re-solve MILP with expanded cluster set

## References

### Related Modules

- **[optimization.md](optimization.md)**: Provides initial solution
- **[clustering.md](clustering.md)**: Defines cluster structures
- **[route_time_estimation.md](route_time_estimation.md)**: Checks feasibility
- **[pipeline.md](pipeline.md)**: Orchestrates full flow

### Literature

1. **Ostermeier & Hübner (2018)** - *Vehicle selection*: ALNS with merge operators
2. **Vidal et al. (2013)** - *Hybrid genetic search for VRPTW*: Improvement heuristics
3. **Pisinger & Ropke (2007)** - *Large neighborhood search*: LNS framework

## See Also

- [← Optimization](optimization.md)
- [Next: Pipeline →](pipeline.md)
- [↑ Architecture](../ARCHITECTURE.md)
- [Docs Home](../README.md)

---

**Navigation**: [← Optimization](optimization.md) | [↑ Specs Index](../README.md#-module-specifications) | [Pipeline →](pipeline.md)

