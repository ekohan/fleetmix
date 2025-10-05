# Feasible Customer Clustering

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

The clustering module generates capacity- and time-feasible customer clusters for each vehicle configuration. This is the second phase of the matheuristic pipeline and creates the input set for the fleet size and mix optimization.

Clusters are groups of customers that can be feasibly served by a specific vehicle configuration, satisfying both:
1. **Capacity constraints**: Total demand ≤ vehicle capacity
2. **Route duration constraints**: Estimated route time ≤ maximum allowed time

Clusters may overlap (a customer can appear in multiple clusters), allowing the optimization phase to choose the best allocation.

## Paper Connection

- **Primary Reference**: Paper §4.2 "Generate Feasible Customer Clusters"
- **Related Sections**: 
  - Appendix A: Clustering techniques comparison
  - §4.2: Split-and-merge procedure (Algorithm 1)
  - §4.2: Composite distance metric
- **Key Equations**: Distance metric, BHH formula
- **Algorithms**: Algorithm 1 (Recursive Cluster Splitting)

## Mathematical Formulation

### Notation

| Symbol | Description |
|--------|-------------|
| $K$ | Set of all feasible clusters |
| $K_v \subseteq K$ | Clusters feasible for vehicle configuration $v$ |
| $K_i \subseteq K$ | Clusters containing customer $i$ |
| $Q_v$ | Capacity of vehicle configuration $v$ |
| $T_v$ | Maximum route duration for configuration $v$ |
| $t_{vk}$ | Route time for cluster $k$ with configuration $v$ |
| $N$ | Set of customers |
| $P$ | Set of product types |
| $d_{ip}$ | Demand of customer $i$ for product $p$ |

### Feasibility Conditions

A cluster $k$ is feasible for configuration $v$ if:

1. **Capacity feasibility**:
$$\sum_{i \in k} \sum_{p \in P} d_{ip} \leq Q_v$$

2. **Route duration feasibility**:
$$t_{vk} \leq T_v$$

where $t_{vk}$ is estimated using continuous approximation or TSP solver (see [route_time_estimation.md](route_time_estimation.md)).

### Composite Distance Metric

To account for both geographical and demand similarity:

$$D_{ij} = \lambda \cdot D^{\text{geo}}_{ij} + (1-\lambda) \cdot D^{\text{prod}}_{ij}$$

where:
- $\lambda \in [0,1]$ is the geo-weight parameter
- $D^{\text{geo}}_{ij}$ is the geographical distance (Haversine formula)
- $D^{\text{prod}}_{ij}$ is the cosine distance between demand composition profiles

**Demand Composition Profile** for customer $i$:

$$\text{profile}_i = \frac{1}{\sum_p d_{ip}} \cdot [w_p \cdot d_{ip}]_{p \in P}$$

where $w_p$ are product-specific weights (capturing temperature sensitivity).

## Design Decisions

### Why Multiple Clustering Algorithms?

**Algorithms used**:
1. K-means (MiniBatch variant for speed)
2. K-medoids
3. Gaussian Mixture Model
4. Agglomerative clustering

**Rationale**:
- Different algorithms find different cluster shapes
- No single algorithm dominates across all demand patterns
- "Ensemble" approach: Generate diverse candidates, let optimizer choose
- Computational cost is acceptable when parallelized

**Evidence from experiments** (Paper Appendix A):
- K-means: 38% selection frequency
- K-medoids: 21%
- GMM: 18%
- Agglomerative: 8%
- Merged: 15%

No clear winner → justify using all methods

### Split-and-Merge Strategy

**Problem**: Initial clustering often produces infeasible clusters (too large)

**Solution**: Two-phase approach

**Phase 1: Recursive Splitting** (Algorithm 1)
- When cluster exceeds capacity or time, split in two
- Recurse until feasible or max depth reached
- Produces many small clusters

**Phase 2: Merging**
- Merge small clusters ($|k| < \eta$) with nearby neighbors
- Check feasibility of merged cluster
- Enriches cluster set with consolidated options

**Alternative rejected**: Capacitated clustering during initial phase
- Would require custom implementations
- Less flexible for experimentation
- Post-hoc splitting is simpler and more modular

### Overlapping vs. Partitioning

**Our choice**: Clusters can overlap (customer in multiple clusters)

**Benefits**:
- Optimization model chooses best allocation
- More flexibility, better solutions
- Natural with multiple clustering methods

**Implementation**: Set covering formulation in optimization phase (§4.3)

## Interfaces

### Input

```python
customers: list[CustomerBase]  # Customer data with location & demand
configurations: list[VehicleConfiguration]  # From Phase 1
params: FleetmixParams  # Clustering parameters
```

### Output

```python
list[Cluster]
# Where each Cluster has:
Cluster(
    cluster_id="cfg_001_kmeans_k5_0",
    config_id="cfg_001",
    method="kmeans",
    customer_ids=["c1", "c2", "c3"],
    total_demand={"Dry": 100, "Chilled": 50, "Frozen": 30},
    route_time=4.5,  # hours
    total_cost=350.0,  # fixed + variable
)
```

### Protocol Definition

```python
class Clusterer(Protocol):
    """Protocol for clustering algorithms."""
    
    def fit(
        self,
        customers: pd.DataFrame,
        *,
        context: CapacitatedClusteringContext,
        n_clusters: int,
    ) -> list[int]:
        """
        Cluster customers into n_clusters groups.
        
        Args:
            customers: DataFrame with Latitude, Longitude, demand columns
            context: Contains vehicle config, goods, weights
            n_clusters: Target number of clusters
            
        Returns:
            List of cluster labels (integers) for each customer
        """
        ...
```

## Key Algorithms

### Algorithm 1: Recursive Cluster Splitting

**Purpose**: Ensure all clusters satisfy capacity and route duration constraints

**Input**: Initial cluster (potentially infeasible), max recursion depth

**Output**: List of feasible sub-clusters

**Pseudocode**:

```python
def split_cluster_recursively(cluster, depth, max_depth, context):
    if depth >= max_depth:
        return [cluster]  # Stop recursion
    
    # Check feasibility
    if is_feasible(cluster, context):
        return [cluster]  # Feasible, done
    
    # Infeasible, split into two
    sub_clusters = split_into_two(cluster)
    
    # Recurse on each sub-cluster
    result = []
    for sub in sub_clusters:
        result.extend(
            split_cluster_recursively(sub, depth + 1, max_depth, context)
        )
    
    return result
```

**Complexity**: $O(\log n)$ depth typically, worst case $O(\text{max\_depth})$

**Implementation note**: Uses K-means with k=2 for splitting

### Merge Small Clusters

**Purpose**: Consolidate clusters smaller than threshold $\eta$

**Steps**:
1. Identify small clusters: $|k| < \eta$
2. For each small cluster $k_i$:
   - Find neighbors within distance $\Delta$
   - For each neighbor $k_j$:
     - Check if $k_i \cup k_j$ is feasible
     - If yes, add to cluster set
3. Keep both original and merged clusters (optimizer decides)

**Complexity**: $O(m^2)$ where $m$ = number of small clusters

**Parameters**:
- $\eta$: Minimum cluster size (default: 3)
- $\Delta$: Maximum merge distance (default: 10 km)

### Composite Distance Computation

**Purpose**: Combine geographic and demand similarity into single metric

**Steps**:

1. **Compute geographic distance matrix**:
```python
D_geo[i,j] = haversine(lat_i, lon_i, lat_j, lon_j)
```

2. **Compute demand composition profiles**:
```python
profile_i = normalize([w_p * demand_ip for p in products])
```

3. **Compute cosine distance**:
```python
D_prod[i,j] = 1 - cosine_similarity(profile_i, profile_j)
```

4. **Combine with weight**:
```python
D[i,j] = λ * D_geo[i,j] + (1-λ) * D_prod[i,j]
```

**Complexity**: $O(n^2)$ for $n$ customers

## Implementation Notes

### Code Organization

- **Primary Modules**:
  - `src/fleetmix/clustering/generator.py`: Main orchestration (445 lines)
  - `src/fleetmix/clustering/heuristics.py`: Algorithm implementations (767 lines)
  - `src/fleetmix/merging/core.py`: Merge operations (178 lines)

- **Key Functions**:
  - `generate_feasible_clusters()`: Entry point
  - `create_initial_clusters()`: Apply clustering algorithms
  - `process_clusters_recursively()`: Splitting (Algorithm 1)
  - `generate_merge_phase_clusters()`: Post-clustering merging

### Dependencies

- **Internal**: 
  - `fleetmix.core_types`
  - `fleetmix.utils.route_time` (see [route_time_estimation.md](route_time_estimation.md))
  - `fleetmix.interfaces.Clusterer`
  
- **External**:
  - `scikit-learn`: K-means, GMM, Agglomerative
  - `kmedoids`: K-medoids (FasterPAM algorithm)
  - `numpy`, `pandas`: Data structures
  - `joblib`: Parallelization

### Performance Considerations

**Typical runtime** (1000 customers, 21 configurations):
- Initial clustering: 5-10s
- Feasibility checking: 1-2s (BHH), 30-60s (TSP)
- Splitting & merging: 2-5s
- **Total**: 10-20s (BHH), 40-80s (TSP)

**Parallelization**:
- Cluster generation per configuration is independent
- Uses `joblib.Parallel` with `n_jobs=-1` (all cores)
- Speedup: ~4x on 8-core machine

**Memory**:
- Distance matrices: $O(n^2)$ per configuration
- Cached when multiple methods share same distance metric
- TSP requires distance matrix precomputation

### Edge Cases

1. **No feasible clusters**: Returns empty list, logged as warning
2. **Single-customer clusters**: Always feasible (if customer demand ≤ capacity)
3. **Very large clusters**: Recursion depth limit prevents infinite splitting
4. **Identical customer locations**: Geographic distance = 0, uses demand distance

## Usage Examples

### Basic Usage

```python
from fleetmix.clustering.generator import generate_feasible_clusters
from fleetmix.config.params import FleetmixParams

# Assuming customers and configs already created
clusters = generate_feasible_clusters(
    customers=customers,
    configurations=vehicle_configs,
    params=params,
)

print(f"Generated {len(clusters)} feasible clusters")

# Analyze cluster sizes
sizes = [len(c.customer_ids) for c in clusters]
print(f"Average cluster size: {sum(sizes) / len(sizes):.1f}")
print(f"Min/Max: {min(sizes)} / {max(sizes)}")
```

### Per-Configuration Analysis

```python
# Group clusters by configuration
from collections import defaultdict

by_config = defaultdict(list)
for cluster in clusters:
    by_config[cluster.config_id].append(cluster)

for cfg_id, cfg_clusters in by_config.items():
    print(f"{cfg_id}: {len(cfg_clusters)} clusters")
```

### Custom Clustering Parameters

```yaml
# config.yaml
clustering:
  methods:
    - minibatch_kmeans
    - kmedoids
    - gaussian_mixture
    - agglomerative
  
  geo_weight: [1.0, 0.8, 0.6]  # Geographic vs demand trade-off
  demand_weight: [0.0, 0.2, 0.4]
  
  n_clusters_range:
    - auto  # Automatic based on customer/capacity ratio
    - 5
    - 10
  
  recursive_split:
    max_depth: 5
  
  merge:
    min_cluster_size: 3
    max_distance: 10.0  # km
```

### Filtering Clusters

```python
# Get only large clusters (good vehicle utilization)
large_clusters = [c for c in clusters if len(c.customer_ids) >= 10]

# Get clusters for specific configuration
config_clusters = [c for c in clusters if c.config_id == "cfg_005"]

# Get clusters with low route time (fast routes)
fast_clusters = [c for c in clusters if c.route_time <= 6.0]
```

## Extension Points

### Adding a Custom Clustering Algorithm

Implement the `Clusterer` protocol and register:

```python
from fleetmix.registry import register_clusterer
from fleetmix.interfaces import Clusterer
import pandas as pd

@register_clusterer("my_custom_method")
class MyCustomClusterer:
    """Custom clustering based on domain-specific logic."""
    
    def fit(
        self,
        customers: pd.DataFrame,
        *,
        context: CapacitatedClusteringContext,
        n_clusters: int,
    ) -> list[int]:
        """Custom clustering implementation."""
        # Access vehicle capacity
        capacity = context.vehicle_config.capacity
        
        # Access goods
        goods = context.goods
        
        # Your clustering logic here
        # Must return list of cluster labels (ints)
        labels = your_algorithm(customers, n_clusters)
        
        return [int(label) for label in labels]
```

Then use in config:

```yaml
clustering:
  methods:
    - my_custom_method
```

### Modifying Distance Metric

To use alternative distance functions:

1. **Edit `heuristics.py:compute_composite_distance()`**:
```python
def compute_composite_distance(customers, goods, geo_weight, demand_weight):
    # Your custom distance computation
    D = your_distance_function(customers)
    return D
```

2. **Or use geographic-only**:
```yaml
clustering:
  geo_weight: [1.0]  # λ = 1, ignore demand
  demand_weight: [0.0]
```

### Adding Capacitated Clustering

To integrate constraint-aware clustering from the start:

```python
@register_clusterer("capacitated_kmeans")
class CapacitatedKMeansClusterer:
    def fit(self, customers, *, context, n_clusters):
        # Use capacity-constrained K-means
        # E.g., COP-KMeans or similar
        capacity = context.vehicle_config.capacity
        ...
```

**Literature**: Ferreira et al. (2013) - Capacitated K-means

### Fixed Compartment Variant

To implement Henke et al. (2015) fixed compartments:

1. **Modify feasibility check** to verify each compartment's capacity:
```python
def check_compartment_capacity(cluster, vehicle_config):
    for product in goods:
        if cluster.demand[product] > vehicle_config.compartment_capacity[product]:
            return False
    return True
```

2. **Update configuration generation** to include fixed capacities (see [vehicle_configurations.md](vehicle_configurations.md))

## Testing

### Unit Tests

- **Location**: `tests/unit/test_clustering.py`, `tests/unit/test_heuristics.py`
- **Coverage**:
  - Each clustering algorithm produces valid labels
  - Split recursion terminates correctly
  - Merge logic doesn't create infeasible clusters
  - Distance metric is symmetric and non-negative

### Integration Tests

- **Location**: `tests/integration/test_clustering_pipeline.py`
- **Scenarios**:
  - End-to-end cluster generation
  - Feasibility verification
  - Cluster overlap handling

## Comparison with Literature

### Henke et al. (2015, 2019)

**Their approach**: Integrated into VNS/Branch-and-cut metaheuristic

**FleetMix**: Separate cluster-first phase

**Trade-off**: 
- They: Better solution quality (routing integrated)
- We: Better scalability (clustering + MILP is faster)

### Ostermeier & Hübner (2018)

**Their approach**: ALNS with destroy-repair operators

**FleetMix**: Pre-generate clusters, optimize assignment

**Commonality**: Both use multiple clustering methods

## References

### Related Modules

- **[vehicle_configurations.md](vehicle_configurations.md)**: Provides vehicle configs input
- **[route_time_estimation.md](route_time_estimation.md)**: Computes $t_{vk}$ for feasibility
- **[optimization.md](optimization.md)**: Consumes clusters for MILP
- **[protocols.md](protocols.md)**: `Clusterer` protocol definition

### Literature

1. **Beardwood, Halton, Hammersley (1959)** - Shortest path through random points: BHH formula
2. **Henke et al. (2015)** - Multi-compartment VRP problem: VNS approach
3. **Ostermeier & Hübner (2018)** - Vehicle selection: ALNS metaheuristic
4. **Ferreira et al. (2013)** - Capacitated K-means: Constraint-aware clustering

### External Documentation

- [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [kmedoids package](https://github.com/kno10/python-kmedoids)

## See Also

- [← Vehicle Configurations](vehicle_configurations.md)
- [Next: Route Time Estimation →](route_time_estimation.md)
- [↑ Architecture](../ARCHITECTURE.md)
- [Docs Home](../README.md)

---

**Navigation**: [← Vehicle Configs](vehicle_configurations.md) | [↑ Specs Index](../README.md#-module-specifications) | [Route Time →](route_time_estimation.md)

