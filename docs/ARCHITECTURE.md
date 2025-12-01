# FleetMix Architecture

> **High-level system design and module interactions**

---

## Overview

FleetMix implements a **cluster-first, fleet design-second matheuristic** for heterogeneous multi-compartment vehicle fleet optimization. This document provides the architectural overview connecting the paper's methodology (§4) to the codebase structure.

---

## Matheuristic Pipeline

The system implements the four-phase pipeline described in the paper:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Input: Demand Data                          │
│            (customers × product types × quantities)                 │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: Generate Vehicle Configurations                           │
│  ────────────────────────────────────────────────────────────────── │
│  • For each vehicle type                                            │
│  • Generate 2^|M| - 1 compartment combinations                      │
│  • Apply allowed_goods constraints                                  │
│  • Result: Set V of feasible configurations                         │
│  ────────────────────────────────────────────────────────────────── │
│  📄 Paper §4.1 | 📦 Module: utils/vehicle_configurations.py         │
│  📐 Spec: specs/vehicle_configurations.md                           │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 2: Generate Feasible Customer Clusters                       │
│  ────────────────────────────────────────────────────────────────── │
│  • For each configuration v ∈ V                                     │
│  • Apply clustering algorithms (k-means, k-medoids, GMM, agglo)     │
│  • Check capacity: Σ demand ≤ Q_v                                   │
│  • Check route time: t_vk ≤ T_v                                     │
│  • Split-and-merge procedure                                        │
│  • Result: Set K of feasible clusters (potentially overlapping)     │
│  ────────────────────────────────────────────────────────────────── │
│  📄 Paper §4.2, Algorithm 1 | 📦 Module: clustering/                │
│  📐 Specs: specs/clustering.md, specs/route_time_estimation.md      │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 3: Optimize Fleet Size and Mix (MILP)                        │
│  ────────────────────────────────────────────────────────────────── │
│  • Decision variables: x_vk ∈ {0,1}                                 │
│  • Minimize: Σ c_vk · x_vk                                          │
│  • Subject to:                                                      │
│    - Each customer served: Σ x_vk = 1 ∀i                            │
│    - Each cluster used ≤ once: Σ x_vk ≤ 1 ∀k                        │
│  • Result: Initial fleet design                                     │
│  ────────────────────────────────────────────────────────────────── │
│  📄 Paper §4.3, Problem (P) | 📦 Module: optimization/core.py       │
│  📐 Spec: specs/optimization.md                                     │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 4: Improvement (Iterative Merge)                             │
│  ────────────────────────────────────────────────────────────────── │
│  • For each pair of selected clusters (k_i, k_j)                    │
│  • Check if k_i ∪ k_j is feasible                                   │
│  • Add merged cluster to K                                          │
│  • Re-solve MILP                                                    │
│  • Repeat until no improvement                                      │
│  • Result: Enhanced fleet design                                    │
│  ────────────────────────────────────────────────────────────────── │
│  📄 Paper §4.4 | 📦 Module: post_optimization/merge_phase.py        │
│  📐 Spec: specs/post_optimization.md                                │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Output: Fleet Design Solution                    │
│  • Number of vehicles per configuration                             │
│  • Customer-to-vehicle assignments                                  │
│  • Compartment capacities per vehicle                               │
│  • Total costs (fixed + variable)                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Organization

### Core Components

Located in `src/fleetmix/`, these implement the matheuristic pipeline:

| Module | Purpose | Paper Ref | Spec |
|--------|---------|-----------|------|
| `utils/vehicle_configurations.py` | Generate vehicle configurations | §4.1 | [specs/vehicle_configurations.md](specs/vehicle_configurations.md) |
| `clustering/` | Feasible cluster generation | §4.2 | [specs/clustering.md](specs/clustering.md) |
| `utils/route_time.py` | Route duration estimation | §4.2 | [specs/route_time_estimation.md](specs/route_time_estimation.md) |
| `optimization/core.py` | Fleet size & mix MILP | §4.3 | [specs/optimization.md](specs/optimization.md) |
| `post_optimization/merge_phase.py` | Improvement phase | §4.4 | [specs/post_optimization.md](specs/post_optimization.md) |
| `api.py` | Pipeline orchestration & two-phase split-stop | §4 | [specs/pipeline.md](specs/pipeline.md) |

### Supporting Infrastructure

| Module | Purpose | Spec |
|--------|---------|------|
| `interfaces.py` | Protocol definitions | [specs/protocols.md](specs/protocols.md) |
| `core_types.py` | Data structures | [specs/data_model.md](specs/data_model.md) |
| `config/` | Configuration system | [specs/configuration.md](specs/configuration.md) |
| `registry.py` | Plugin registration | [specs/protocols.md](specs/protocols.md) |
| `preprocess/demand.py` | Demand preprocessing | [specs/data_model.md](specs/data_model.md) |
| `merging/core.py` | Cluster merging utilities | [specs/clustering.md](specs/clustering.md) |

### User Interfaces

| Module | Purpose |
|--------|---------|
| `api.py` | Python API facade (`fleetmix.optimize()`) |
| `app.py` | CLI (Typer-based, `fleetmix` command) |
| `gui.py` | Web interface (Streamlit-based) |

### Benchmarking & Validation

Located in `src/fleetmix/benchmarking/`:

| Component | Purpose | Paper Ref |
|-----------|---------|-----------|
| `parsers/` | CVRP/MCVRP instance parsers | §5 |
| `converters/` | VRP → FSM format conversion | §5 |
| `solvers/vrp_solver.py` | PyVRP integration for bounds | §5 |
| `datasets/` | Benchmark instances (Henke, Uchoa, Case study) | §5, §6 |

See [specs/benchmarking.md](specs/benchmarking.md) for details.

---

## Data Flow

### Input Pipeline

```
User Input (CSV/YAML)
    │
    ▼
config.loader.parse_config()
    │
    ▼
preprocess.demand.process_demand_data()
    │
    ▼
core_types.CapacitatedClusteringContext
```

### Clustering Pipeline

```
CapacitatedClusteringContext
    │
    ▼
clustering.generator.generate_all_clusters()
    │
    ├─→ heuristics.KMeansClusterer.fit()
    ├─→ heuristics.KMedoidsClusterer.fit()
    ├─→ heuristics.GaussianMixtureClusterer.fit()
    └─→ heuristics.AgglomerativeClusterer.fit()
    │
    ▼
Split-and-merge (Algorithm 1)
    │
    ▼
List[ClusterInfo]
```

### Optimization Pipeline

```
List[ClusterInfo]
    │
    ▼
optimization.core.build_milp_model()
    │
    ▼
PuLP MILP Model
    │
    ▼
solver.solve() [Gurobi/CBC/etc.]
    │
    ▼
Solution (x_vk values)
    │
    ▼
post_optimization.merge_phase.improve_solution()
    │
    ▼
Final Solution
```

### Output Pipeline

```
Final Solution
    │
    ▼
utils.save_results.format_solution()
    │
    ├─→ JSON
    ├─→ Excel
    └─→ HTML Report
```

---

## Key Design Principles

### 1. Protocol-Based Plugin Architecture

FleetMix uses Python Protocols (PEP 544) instead of abstract base classes:

```python
# interfaces.py
class Clusterer(Protocol):
    def fit(self, customers: pd.DataFrame, *, 
            context: CapacitatedClusteringContext, 
            n_clusters: int) -> list[int]:
        ...
```

**Benefits**:
- **Structural subtyping**: Any class with matching signature works
- **No inheritance required**: Easy to integrate external libraries
- **Type-safe**: MyPy validates protocol compliance

See [specs/protocols.md](specs/protocols.md) for plugin development guide.

### 2. Separation of Concerns

- **Algorithms** (clustering, route time) are pure functions
- **Orchestration** (pipeline) coordinates but doesn't contain logic
- **Configuration** is centralized and validated
- **I/O** is isolated in utils/

### 3. Composability

Each component can be:
- **Used standalone**: Import and call directly
- **Replaced via registry**: `@register_clusterer("my_method")`
- **Configured via YAML**: No code changes needed for parameter tuning

---

## Extension Points

FleetMix is designed for research extensibility:

| Extension | How to Implement | Relevant Spec |
|-----------|------------------|---------------|
| Custom clustering | Implement `Clusterer` protocol, register | [specs/clustering.md](specs/clustering.md) |
| Custom route time | Implement `RouteTimeEstimator` protocol | [specs/route_time_estimation.md](specs/route_time_estimation.md) |
| Custom solver | Implement `SolverAdapter` protocol | [specs/optimization.md](specs/optimization.md) |
| Fixed compartments | Modify configuration generation | [specs/vehicle_configurations.md](specs/vehicle_configurations.md) |
| Problem variants | See paper §8 for extensions | Paper §8 (future work) |

---

## Dependencies

### External Libraries

- **Optimization**: PuLP (interface), Gurobi/CBC (solvers)
- **Clustering**: scikit-learn
- **Routing**: PyVRP (for benchmarking bounds)
- **Numerics**: NumPy, pandas
- **Config**: PyYAML, pydantic
- **CLI**: Typer, rich
- **GUI**: Streamlit

---

## Configuration System

FleetMix uses hierarchical YAML configuration:

```yaml
vehicles:          # Vehicle types and costs
  TypeA: {...}
goods:             # Product types
  - Dry
  - Chilled
optimization:      # MILP parameters
  solver: gurobi
  time_limit: 300
clustering:        # Clustering parameters
  methods: [kmeans, kmedoids, gmm, agglomerative]
  n_clusters_range: [auto]
route_time:        # Route estimation method
  method: bhh      # or 'tsp'
```

See [specs/configuration.md](specs/configuration.md) for complete schema.

---

## See Also

- [mapping.md](mapping.md) - Detailed paper ↔ code cross-reference
- [specs/](specs/) - Individual module specifications
- Paper §8 - Discussion of future extensions
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) - Reproduce paper experiments

---

**Navigation**: [← Back to Docs Home](README.md) | [Module Specs →](specs/)

