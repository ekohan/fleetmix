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

1. **Fixed cost**: Vehicle fixed cost
2. **Variable cost**: Time-based
   $$c_{vk}^{\text{variable}} = \text{variable cost per hour} \times t_{vk}$$
   where $t_{vk}$ is route time (see [route_time_estimation.md](route_time_estimation.md))
3. **Compartment setup cost**: $(|M_v| - 1) \times \text{compartment setup cost}$
   where $|M_v|$ is number of active compartments
4. **Light load penalty**: Applied if load % < threshold

**Total**: $c_{vk} = \text{fixed} + \text{variable} + \text{compartment} + \text{penalty}$

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
    selected_clusters=[...],  # list[Cluster]
    vehicles_used={'cfg_001': 3, 'cfg_005': 2},  # Count per config
    solver_status="Optimal",
    solver_runtime_sec=2.5,
    optimality_gap=0.0,
)
```

### Data Types

See [data_model.md](data_model.md) for complete `FleetmixSolution` structure.

## Key Algorithms

### Model Creation

**Implementation**: `_create_model()` in `optimization/core.py`

**Key steps**:
1. Create decision variables $x_{vk}$ and $y_k$ 
2. Compute cost coefficients $c_{vk}$ (fixed + variable + compartment + penalties)
3. Set objective: minimize $\sum c_{vk} \cdot x_{vk}$
4. Add customer coverage constraints (Eq. 2)
5. Add cluster usage constraints (Eq. 3)
6. Link variables: $x_{vk} \leq y_k$

### Solution Extraction

**Implementation**: `_extract_solution()` in `optimization/core.py`

**Key steps**:
1. Extract selected variables ($x_{vk} = 1$)
2. Count vehicles per configuration
3. Compute cost breakdown (fixed, variable, penalties)
4. Validate all customers served
5. Return `FleetmixSolution` object

## Implementation Notes

### Code Organization

- **Primary Module**: `src/fleetmix/optimization/core.py`
- **Key Functions**:
  - `optimize_fleet()`: Main entry point
  - `_solve_internal()`: Internal implementation
  - `_create_model()`: Construct PuLP model
  - `_extract_solution()`: Parse solver output
  - `_calculate_cluster_cost()`: Compute cost coefficients

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

1. **No feasible solution**: Logged as error; may indicate invalid cluster generation
2. **Cluster without compatible vehicle**: Handled with "NoVehicle" placeholder (forced to 0)
3. **Multiple optimal solutions**: Solver returns one arbitrarily

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
# Selected clusters (list[Cluster])
print(f"Total clusters selected: {len(solution.selected_clusters)}")

# Cost breakdown
print(f"Fixed costs: ${solution.total_fixed_cost:,.2f} "
      f"({solution.total_fixed_cost/solution.total_cost*100:.1f}%)")
print(f"Variable costs: ${solution.total_variable_cost:,.2f} "
      f"({solution.total_variable_cost/solution.total_cost*100:.1f}%)")
print(f"Penalties: ${solution.total_penalties:,.2f}")
```

## See Also

- [← Route Time Estimation](route_time_estimation.md)
- [Next: Post-Optimization →](post_optimization.md)
- [↑ Architecture](../ARCHITECTURE.md)
- [Docs Home](../README.md)

---

**Navigation**: [← Route Time](route_time_estimation.md) | [↑ Specs Index](../README.md#-module-specifications) | [Post-Opt →](post_optimization.md)

