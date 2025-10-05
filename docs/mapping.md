# Code ↔ Paper Mapping

> **Comprehensive cross-reference between the paper and FleetMix implementation**

This document enables readers of *Designing Multi-Compartment Last-Mile Vehicle Fleets: An Open-Source Matheuristic* to navigate seamlessly between the research methodology and its implementation.

---

## Quick Reference Table

| Paper Element | Code Location | Specification |
|---------------|---------------|---------------|
| §4.1 Generate Vehicle Configurations | `src/fleetmix/utils/vehicle_configurations.py` | [specs/vehicle_configurations.md](specs/vehicle_configurations.md) |
| §4.2 Generate Feasible Clusters | `src/fleetmix/clustering/` | [specs/clustering.md](specs/clustering.md) |
| §4.2 Route Time Computation | `src/fleetmix/utils/route_time.py` | [specs/route_time_estimation.md](specs/route_time_estimation.md) |
| §4.3 Fleet Size & Mix MILP | `src/fleetmix/optimization/core.py` | [specs/optimization.md](specs/optimization.md) |
| §4.4 Improvement Phase | `src/fleetmix/post_optimization/merge_phase.py` | [specs/post_optimization.md](specs/post_optimization.md) |
| §5 Benchmark Experiments | `src/fleetmix/benchmarking/` | [specs/benchmarking.md](specs/benchmarking.md) |
| §6 Case Study | `src/fleetmix/benchmarking/datasets/case/` | [specs/benchmarking.md](specs/benchmarking.md) |

---

## Paper Section → Code

### §3 Problem Definition

| Concept | Notation | Implementation |
|---------|----------|----------------|
| Set of customers | $N = \\{1, ..., n\\}$ | `customers: pd.DataFrame` |
| Set of product types | $P$ | `config['goods']` |
| Customer demand | $d_{ip}$ | `customers[f'demand_{product}']` |
| Vehicle configurations | $V$ | `List[VehicleConfig]` from `vehicle_configurations.py` |
| Clusters | $K$ | `List[ClusterInfo]` from `clustering/generator.py` |
| Max capacity | $Q_v$ | `vehicle_config.capacity` |
| Max route time | $T_v$ | `vehicle_config.max_route_time` |

**Code entry point**: `src/fleetmix/core_types.py`  
**Spec**: [specs/data_model.md](specs/data_model.md)

---

### §4.1 Generate Vehicle Configurations

**Paper Description**: For each vehicle type, generate $2^{|M|}-1$ configurations by enumerating all non-empty subsets of compartments $M$.

**Implementation**:
```python
# src/fleetmix/utils/vehicle_configurations.py
def generate_vehicle_configurations(
    vehicles: dict,
    goods: list[str],
) -> list[VehicleConfig]:
    """
    For each vehicle type, generates all 2^|M| - 1 feasible 
    compartment combinations.
    """
```

**Key Functions**:
- `generate_vehicle_configurations()`: Main entry point
- `_get_feasible_compartment_subsets()`: Respects `allowed_goods` constraints
- `_generate_single_vehicle_configs()`: Per-vehicle generation

**Spec**: [specs/vehicle_configurations.md](specs/vehicle_configurations.md)

---

### §4.2 Generate Feasible Customer Clusters

#### Clustering Algorithms

**Paper Description**: Apply k-means, k-medoids, Gaussian mixture, and agglomerative clustering with composite distance metric.

**Composite Distance Metric** (Equation in §4.2):

$$D_{ij} = \lambda \cdot D^{\text{geo}}_{ij} + (1-\lambda) \cdot D^{\text{prod}}_{ij}$$

**Implementation**:
```python
# src/fleetmix/clustering/heuristics.py
class MiniBatchKMeansClusterer:
    def fit(self, customers: pd.DataFrame, *, 
            context: CapacitatedClusteringContext,
            n_clusters: int) -> list[int]:
        # Computes composite distance matrix
        # Returns cluster labels
```

**Available Clusterers**:
- `MiniBatchKMeansClusterer` 
- `KMedoidsClusterer`
- `GaussianMixtureClusterer`
- `AgglomerativeClusterer`

**Spec**: [specs/clustering.md](specs/clustering.md)

#### Split-and-Merge Procedure

**Paper Algorithm 1**: Recursive cluster splitting

**Implementation**:
```python
# src/fleetmix/clustering/heuristics.py
def process_clusters_recursively(
    cluster: ClusterInfo,
    depth: int,
    max_depth: int,
    context: CapacitatedClusteringContext,
) -> list[ClusterInfo]:
    """Implements Algorithm 1 from paper"""
```

**Merging**:
```python
# src/fleetmix/merging/core.py
def merge_small_clusters(
    clusters: list[ClusterInfo],
    min_size: int,
    max_distance: float,
) -> list[ClusterInfo]:
```

**Spec**: [specs/clustering.md](specs/clustering.md)

#### Route Time Computation

**Paper Equation** (§4.2, BHH approximation):

$$t_{vk} \approx \alpha_{vk} + 2 \cdot \delta_{vk} + \beta \cdot \sqrt{n \cdot A} + \gamma \cdot n$$

where:
- $\alpha_{vk}$: Setup time
- $\delta_{vk}$: Depot-to-cluster line-haul time  
- $\beta$: BHH constant
- $A$: Service area
- $\gamma$: Service time per customer

**Implementation**:
```python
# src/fleetmix/utils/route_time.py
class BHHRouteTimeEstimator:
    def estimate_route_time(
        self,
        cluster_customers: pd.DataFrame,
        context: RouteTimeContext,
    ) -> tuple[float, list[str]]:
        """Implements BHH continuous approximation formula"""
```

**Alternative** (TSP-based):
```python
class TSPRouteTimeEstimator:
    def estimate_route_time(self, ...) -> tuple[float, list[str]]:
        """Uses PyVRP for exact TSP solution"""
```

**Spec**: [specs/route_time_estimation.md](specs/route_time_estimation.md)

---

### §4.3 Optimize Fleet Size and Mix

**Paper Problem (P)** (Equations 1-4):

**Objective**:
$$\text{minimize} \quad \sum_{v \in V}\sum_{k \in K_v} c_{vk} \cdot x_{vk}$$

**Constraints**:
$$\sum_{k \in K_i} \sum_{v \in V_k} x_{vk} = 1, \quad \forall i \in N$$
$$\sum_{v \in V_k} x_{vk} \leq 1, \quad \forall k \in K$$
$$x_{vk} \in \{0,1\}, \quad \forall v \in V, k \in K_v$$

**Implementation**:
```python
# src/fleetmix/optimization/core.py
def optimize_fleet(
    clusters: list[Cluster],
    configurations: list[VehicleConfiguration],
    customers: list[CustomerBase],
    parameters: FleetmixParams,
) -> FleetmixSolution:
    """
    Main entry point for fleet size and mix optimization.
    Builds and solves Problem (P) as PuLP model.
    """
```

**Key Internal Functions**:
- `_solve_internal()`: Internal DataFrame-based implementation
- `_create_model()`: Builds PuLP model with variables and constraints

**Multi-Stop Policy** (alternative constraints):
```python
def _add_product_coverage_constraints_multistop(model, x_vars, ...):
    """Implements multi-stop variant from §4.3"""
```

**Spec**: [specs/optimization.md](specs/optimization.md)

---

### §4.4 Improvement Phase

**Paper Description**: Iteratively merge pairs of clusters from optimal solution and re-solve MILP until no improvement.

**Implementation**:
```python
# src/fleetmix/post_optimization/merge_phase.py
def improve_solution(
    initial_solution: FleetmixSolution,
    configurations: list[VehicleConfiguration],
    customers: list[CustomerBase],
    params: FleetmixParams,
) -> FleetmixSolution:
    """
    Implements improvement phase from §4.4.
    Iterates until convergence.
    """
```

**Process**:
1. Extract selected clusters from solution
2. Try merging all pairs $(k_i, k_j)$
3. Check feasibility of $k_i \cup k_j$
4. Add feasible merges to cluster set
5. Re-solve MILP
6. Repeat if improved

**Spec**: [specs/post_optimization.md](specs/post_optimization.md)

---

### §5 Effectiveness of the Matheuristic Approach

#### Benchmark Instances

**Paper**: Tests on Henke et al. (2015, 2019) MCVRP instances

**Implementation**:
```python
# src/fleetmix/benchmarking/parsers/mcvrp.py
def parse_mcvrp_instance(filepath: Path) -> MCVRPInstance:
    """Parses Henke's .dat format"""

# src/fleetmix/benchmarking/converters/mcvrp.py
def convert_mcvrp_to_fsm(
    instance: MCVRPInstance,
    config_template: dict,
) -> tuple[pd.DataFrame, dict]:
    """Converts to FleetMix format"""
```

**Benchmark Runner**:
```bash
# Reproduces Table 1 in paper
fleetmix benchmark mcvrp
```

**Spec**: [specs/benchmarking.md](specs/benchmarking.md)

#### CVRP Adaptations

**Paper**: Four transformation strategies (split, scaled, combined, spatial)

**Implementation**:
```python
# src/fleetmix/benchmarking/converters/cvrp.py
def convert_cvrp_to_fsm(
    instance: CVRPInstance,
    benchmark_type: Literal["split", "scaled", "combined", "spatial"],
    config_template: dict,
) -> tuple[pd.DataFrame, dict]:
```

**Spec**: [specs/benchmarking.md](specs/benchmarking.md)

---

### §6 Case Study

**Paper**: Real-world data from Bogotá, Colombia food distributor

**Implementation**:
- **Data**: `src/fleetmix/benchmarking/datasets/case/*.csv`
- **Runner**: 
```bash
fleetmix benchmark case
```

**Sensitivity Analysis**:
```python
# Implemented via parametric config sweeps
# See docs/experimental_design.md
```

**Spec**: [specs/benchmarking.md](specs/benchmarking.md)

---

## Code → Paper

### Implementation to Paper Section

| Code Module | Paper Reference |
|-------------|----------------|
| `api.py` | User-facing API (not in paper) |
| `app.py` | CLI (not in paper) |
| `gui.py` | Web interface (mentioned §5) |
| `clustering/generator.py` | §4.2 |
| `clustering/heuristics.py` | §4.2, Appendix A |
| `config/loader.py` | Configuration (§5, §6 parameters) |
| `core_types.py` | §3 notation |
| `interfaces.py` | Protocol architecture (§5) |
| `merging/core.py` | §4.2 (split-and-merge) |
| `optimization/core.py` | §4.3 (Problem P) |
| `pipeline/vrp_interface.py` | §4 (overall flow) |
| `post_optimization/merge_phase.py` | §4.4 |
| `preprocess/demand.py` | Data preparation (§6) |
| `registry.py` | Plugin system (§5) |
| `utils/vehicle_configurations.py` | §4.1 |
| `utils/route_time.py` | §4.2 (route time computation) |
| `utils/cluster_conversion.py` | Internal utilities |
| `utils/coordinate_converter.py` | Geographic distance (§4.2) |
| `benchmarking/parsers/` | §5 benchmark data |
| `benchmarking/converters/` | §5 instance adaptation |
| `benchmarking/solvers/vrp_solver.py` | §5 bounds computation |

---

## Figures → Code

| Figure | Description | Code Implementation |
|--------|-------------|---------------------|
| Figure 1 | Matheuristic pipeline | `pipeline/vrp_interface.py` |
| Algorithm 1 | Recursive cluster splitting | `clustering/heuristics.py:process_clusters_recursively()` |
| Table 1 | Comparison with Henke et al. | `benchmarking/` results |
| Table in §6 | Baseline parameters | Configuration files |

---

## Equations → Code

| Equation | Description | Implementation |
|----------|-------------|----------------|
| (1) | Objective function | `optimization/core.py:_create_model()` |
| (2) | Customer coverage | `optimization/core.py:_create_model()` |
| (3) | Cluster usage | `optimization/core.py:_create_model()` |
| (4) | Variable domain | `optimization/core.py:_create_model()` |
| §4.2 (BHH) | Route time estimation | `utils/route_time.py:BHHRouteTimeEstimator` |
| §4.2 (distance) | Composite distance | `clustering/heuristics.py:compute_composite_distance()` |

---

## How to Use This Document

### For Paper Readers

1. **Find a concept in the paper** (e.g., "Algorithm 1")
2. **Look up in this document** → Find code location
3. **Read the spec** → Understand implementation details
4. **Inspect the code** → See actual implementation

### For Code Readers

1. **Find a module** (e.g., `clustering/heuristics.py`)
2. **Look up in this document** → Find paper reference
3. **Read paper section** → Understand mathematical foundation
4. **Read the spec** → Bridge theory and implementation

### For Reproducibility

1. **Find experiment in paper** (e.g., "Table 1")
2. **Look up command** → `fleetmix benchmark mcvrp`
3. **Check expected results** → Compare with paper
4. **Debug if needed** → Use specs to understand algorithms

---

## Notation Concordance

| Paper Notation | Code Identifier | Type |
|----------------|-----------------|------|
| $N$ | `customers` | `pd.DataFrame` |
| $P$ | `goods` | `list[str]` |
| $d_{ip}$ | `customers[f'demand_{p}']` | `float` |
| $V$ | `vehicle_configs` | `list[VehicleConfig]` |
| $K$ | `clusters` | `list[ClusterInfo]` |
| $Q_v$ | `vehicle_config.capacity` | `float` |
| $T_v$ | `vehicle_config.max_route_time` | `float` |
| $c_{vk}$ | `cluster.total_cost` | `float` |
| $x_{vk}$ | `x_vars[v, k]` | `pulp.LpVariable` |
| $t_{vk}$ | `cluster.route_time` | `float` |

---

## Additional Resources

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: High-level system design
- **[specs/](specs/)**: Detailed module specifications  
- **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**: Reproduce paper experiments
- **EXTENDING.md**: (TODO) Guide for implementing problem variants

---

**Last Updated**: 2025-10-05  
**Paper Version**: Submitted to *Computers and Industrial Engineering*

---

**Navigation**: [← Back to Docs Home](README.md) | [Architecture →](ARCHITECTURE.md)
