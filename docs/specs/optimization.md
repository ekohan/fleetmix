# Fleet Size and Mix Optimization

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

The optimization module solves the core fleet size and mix problem: selecting which vehicle configurations to use and which customer clusters each should serve. This is the heart of the matheuristic (Phase 3) and determines the optimal fleet composition that minimizes total costs while satisfying all customer demand.

## Paper Connection

- **Primary Reference**: Paper §4.3 "Optimize Fleet Size and Mix with Heterogeneous MCVs"
- **Mathematical Model**: Problem (P), Equations (1)-(4)
- **Decision Variables**: $x_{vk}$ binary variables
- **Extensions**: Multi-stop delivery policy (alternative constraints)

## Mathematical Formulation

### Notation

| Symbol | Description |
|--------|-------------|
| $V$ | Set of vehicle configurations |
| $K$ | Set of feasible clusters |
| $K_v \subseteq K$ | Clusters feasible for configuration $v$ |
| $V_k \subseteq V$ | Configurations that can serve cluster $k$ |
| $K_i \subseteq K$ | Clusters containing customer $i$ |
| $N$ | Set of customers |
| $c_{vk}$ | Total cost of serving cluster $k$ with configuration $v$ |
| $x_{vk}$ | Binary variable: 1 if config $v$ serves cluster $k$, 0 otherwise |

### Problem (P): Set Covering Formulation

**Objective Function** (Equation 1):
$$\text{minimize} \quad Z = \sum_{v \in V}\sum_{k \in K_v} c_{vk} \cdot x_{vk}$$

**Subject to**:

**Customer Coverage** (Equation 2):
$$\sum_{k \in K_i} \sum_{v \in V_k} x_{vk} = 1, \quad \forall i \in N$$

Ensures each customer $i$ is served by exactly one cluster-configuration pair.

**Cluster Usage** (Equation 3):
$$\sum_{v \in V_k} x_{vk} \leq 1, \quad \forall k \in K$$

Ensures each cluster $k$ is used by at most one configuration.

**Variable Domain** (Equation 4):
$$x_{vk} \in \{0,1\}, \quad \forall v \in V, k \in K_v$$

### Cost Structure

**Total cost** $c_{vk}$ includes:

1. **Fixed cost**: Vehicle acquisition + compartment setup
   $$c_{vk}^{\text{fixed}} = \text{vehicle\_fixed\_cost} + \sum_{m \in M_v} \text{compartment\_setup\_cost}_m$$

2. **Variable cost**: Distance or time-based
   $$c_{vk}^{\text{variable}} = \text{variable\_cost\_rate} \times t_{vk}$$

   where $t_{vk}$ is route time (see [route_time_estimation.md](route_time_estimation.md))

**Total**: $c_{vk} = c_{vk}^{\text{fixed}} + c_{vk}^{\text{variable}}$

### Multi-Stop Delivery Policy (Extension)

**Alternative to Equation 2**: Allow customers to be served by multiple vehicles, but each product type must be delivered in full by one vehicle.

Let:
- $P_i$ = product types demanded by customer $i$
- $S_i^p$ = all subsets of $P_i$ containing product $p$
- $K_i^s$ = clusters containing customer $i$ that can serve products in $s$

**Modified constraint**:
$$\sum_{s \in S_i^p} \sum_{k \in K_i^s} \sum_{v \in V_k} x_{vk} = 1, \quad \forall i \in N, p \in P_i$$

**Note**: Default is single-stop policy (Equation 2), multi-stop is optional.

## Design Decisions

### Why Set Covering Formulation?

**Alternative approaches**:
1. **Arc-flow formulation**: Model explicit routes
2. **Vehicle-routing formulation**: Integrated routing-assignment
3. **Column generation**: Dynamically generate clusters

**Our choice**: Set covering over pre-generated clusters

**Rationale**:
- **Simplicity**: Standard MILP, solvable by commercial solvers
- **Scalability**: 1000+ customers in seconds with good solver (Gurobi)
- **Flexibility**: Easy to add side constraints (e.g., fleet size limits)
- **Separation of concerns**: Clustering algorithms can be swapped without changing MILP

**Trade-off**: Relies on cluster quality; poor clustering → suboptimal solution

### Solver Choice

**Supported**: PuLP interface supports multiple backends

| Solver | Speed | License | Use Case |
|--------|-------|---------|----------|
| **CBC** | Good | Open-source | Default, free |
| **Gurobi** | Excellent | Commercial | Large problems, best performance |
| **CPLEX** | Excellent | Commercial | Alternative to Gurobi |

**Default**: CBC (freely available)

**Recommendation**: Gurobi for problems with >500 customers

### Warm Starting

To enable iterative improvement (Phase 4):
- Support warm start from previous solution
- Set variable hints: $x_{vk} = 1$ if selected in previous iteration
- Speeds up re-solve when adding merged clusters

## Interfaces

### Input

```python
clusters: list[Cluster]  # From clustering phase
configurations: list[VehicleConfiguration]  # From Phase 1
customers: list[CustomerBase]  # For validation
parameters: FleetmixParams  # Optimization settings
```

### Output

```python
FleetmixSolution(
    total_cost=10543.75,
    total_fixed_cost=8500.0,
    total_variable_cost=2043.75,
    selected_clusters=pd.DataFrame([...]),  # Clusters used
    vehicles_used={'cfg_001': 3, 'cfg_005': 2},  # Count per config
    solver_status="Optimal",
    solve_time_seconds=2.5,
    optimality_gap=0.0,
)
```

### Data Types

```python
@dataclass
class FleetmixSolution:
    total_cost: float
    total_fixed_cost: float
    total_variable_cost: float
    selected_clusters: pd.DataFrame  # x_vk = 1 entries
    vehicles_used: dict[str, int]  # config_id -> count
    solver_status: str  # "Optimal", "Feasible", "Infeasible"
    solve_time_seconds: float
    optimality_gap: float
    # ... other fields
```

## Key Algorithms

### MILP Construction

**Purpose**: Build PuLP model from clusters and configs

**Steps**:

1. **Create decision variables**:
```python
x = {}
for v in configurations:
    for k in clusters if k feasible for v:
        x[v.id, k.id] = pulp.LpVariable(
            f"x_{v.id}_{k.id}",
            cat=pulp.LpBinary
        )
```

2. **Set objective**:
```python
model += pulp.lpSum(
    cluster.total_cost * x[v.id, k.id]
    for v, k in valid_pairs
)
```

3. **Add customer coverage constraints**:
```python
for customer_id in customers:
    # Find all (v,k) pairs that serve this customer
    covering_vars = [x[v,k] for (v,k) if customer_id in k]
    model += pulp.lpSum(covering_vars) == 1
```

4. **Add cluster usage constraints**:
```python
for k in clusters:
    # Each cluster used by at most one config
    usage_vars = [x[v,k.id] for v if k feasible for v]
    model += pulp.lpSum(usage_vars) <= 1
```

5. **Solve**:
```python
status = model.solve(solver)
```

**Complexity**: 
- Variables: $O(|V| \times |K|)$, typically 1000-5000
- Constraints: $O(|N| + |K|)$, typically 500-2000
- Solve time: 1-5s (Gurobi), 5-20s (CBC)

### Solution Extraction

**Purpose**: Convert PuLP solution to `FleetmixSolution`

**Steps**:

1. **Extract selected variables**:
```python
selected = [
    (v, k) for (v, k), var in x.items()
    if pulp.value(var) > 0.5  # Binary variables
]
```

2. **Count vehicles per configuration**:
```python
vehicles_used = defaultdict(int)
for (v, k) in selected:
    vehicles_used[v] += 1
```

3. **Compute costs**:
```python
total_cost = pulp.value(model.objective)
total_fixed = sum(config.fixed_cost for (v,k) in selected)
total_variable = total_cost - total_fixed
```

4. **Validate coverage**:
```python
served_customers = set()
for (v, k) in selected:
    served_customers.update(k.customer_ids)

assert served_customers == set(customers), "Not all customers served!"
```

## Implementation Notes

### Code Organization

- **Primary Module**: `src/fleetmix/optimization/core.py` (729 lines)
- **Key Functions**:
  - `optimize_fleet()`: Main entry point
  - `_solve_internal()`: Internal implementation
  - `_build_milp_model()`: Construct PuLP model
  - `_extract_solution()`: Parse solver output

### Dependencies

- **Internal**: 
  - `fleetmix.core_types`
  - `fleetmix.utils.solver` (solver selection)
  - `fleetmix.utils.cluster_conversion`
- **External**:
  - `pulp`: MILP modeling interface
  - `gurobipy` (optional): Gurobi solver
  - `pandas`: Solution representation

### Edge Cases

1. **No feasible solution**: Should not happen if clustering correct, but logged as error
2. **Infeasible due to gaps**: Some customers not in any cluster → validation fails
3. **Multiple optimal solutions**: Solver returns one arbitrarily
4. **Degenerate solutions**: Very small cost differences → multiple near-optimal

## Usage Examples

### Basic Usage

```python
from fleetmix.optimization.core import optimize_fleet

solution = optimize_fleet(
    clusters=clusters,
    configurations=vehicle_configs,
    customers=customers,
    parameters=params,
)

print(f"Total cost: ${solution.total_cost:,.2f}")
print(f"Vehicles used: {sum(solution.vehicles_used.values())}")
print(f"Fleet composition:")
for cfg_id, count in solution.vehicles_used.items():
    print(f"  {cfg_id}: {count}")
```

### With Custom Solver

```python
import pulp

# Use Gurobi with custom settings
gurobi_solver = pulp.GUROBI(
    timeLimit=300,  # 5 minutes max
    mip_gap=0.01,  # 1% optimality gap acceptable
    msg=True,  # Show solver output
)

solution = optimize_fleet(
    clusters, configs, customers, params,
    solver=gurobi_solver,
)
```

### Analyzing Solution

```python
# Selected clusters
selected_df = solution.selected_clusters

# Average cluster size
avg_size = selected_df['customer_count'].mean()
print(f"Average cluster size: {avg_size:.1f}")

# Capacity utilization
for idx, row in selected_df.iterrows():
    util = row['total_demand'] / row['capacity'] * 100
    print(f"Cluster {row['cluster_id']}: {util:.1f}% utilization")

# Cost breakdown
print(f"Fixed costs: ${solution.total_fixed_cost:,.2f} "
      f"({solution.total_fixed_cost/solution.total_cost*100:.1f}%)")
print(f"Variable costs: ${solution.total_variable_cost:,.2f} "
      f"({solution.total_variable_cost/solution.total_cost*100:.1f}%)")
```

## Extension Points

### Adding Fleet Size Limits

To limit max vehicles of each type:

```python
# Add to MILP model
for v in vehicle_types:
    model += pulp.lpSum(x[v,k] for k if v,k valid) <= max_vehicles[v]
```

### Adding Fixed Compartment Capacities

Modify cluster feasibility to check per-compartment capacity:

```python
# In clustering phase
def is_feasible_fixed_compartments(cluster, config):
    for product in products:
        if cluster.demand[product] > config.compartment_capacity[product]:
            return False
    return True
```

### Multi-Depot Variant

1. **Partition clusters by depot**:
```python
K_d = clusters assigned to depot d
```

2. **Add depot assignment constraints**:
```python
for d in depots:
    model += pulp.lpSum(x[v,k] for k in K_d) <= vehicles_at_depot[d]
```

See paper §7 (TODO) for discussion of variants.

### Adding Penalties

For soft constraints (e.g., prefer balanced fleet):

```python
# Add slack variables
s_v = pulp.LpVariable("slack_v", lowBound=0)

# Add to objective
model += original_cost + penalty_weight * s_v

# Add constraint with slack
model += pulp.lpSum(x[v,k] for k) <= target[v] + s_v
```

## Testing

### Unit Tests

- **Location**: `tests/unit/test_optimization.py`
- **Coverage**:
  - Model builds correctly
  - All customers covered in solution
  - Cost computation accurate
  - Infeasible cases handled

### Integration Tests

- **Location**: `tests/integration/test_optimization_pipeline.py`
- **Scenarios**:
  - Full pipeline from clusters to solution
  - Multiple solver backends
  - Warm start functionality

## Comparison with Literature

### Ostermeier & Hübner (2018)

**Their approach**: ALNS metaheuristic with destroy-repair

**FleetMix**: MILP over pre-generated clusters

**Trade-off**:
- They: Potentially better solution quality (integrated routing)
- We: Faster, more scalable, easier to extend

### Henke et al. (2015)

**Their approach**: Branch-and-cut for MILP (integrated routing)

**FleetMix**: Cluster-first separation

**Scalability**: 
- They: Up to 50 customers
- We: 1000+ customers

## References

### Related Modules

- **[clustering.md](clustering.md)**: Generates input clusters
- **[post_optimization.md](post_optimization.md)**: Improvement phase
- **[vehicle_configurations.md](vehicle_configurations.md)**: Defines $V$
- **[data_model.md](data_model.md)**: `FleetmixSolution` structure

### Literature

1. **Ostermeier & Hübner (2018)** - *Vehicle selection in a city distribution center*
2. **Henke et al. (2015)** - *Multi-compartment vehicle routing problem*
3. **Balinski & Quandt (1964)** - *On an integer program for delivery problem*: Set covering origins

### External Documentation

- [PuLP Documentation](https://coin-or.github.io/pulp/)
- [Gurobi Python API](https://www.gurobi.com/documentation/)

## See Also

- [← Route Time Estimation](route_time_estimation.md)
- [Next: Post-Optimization →](post_optimization.md)
- [↑ Architecture](../ARCHITECTURE.md)
- [Docs Home](../README.md)

---

**Navigation**: [← Route Time](route_time_estimation.md) | [↑ Specs Index](../README.md#-module-specifications) | [Post-Opt →](post_optimization.md)

