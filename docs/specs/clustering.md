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
- **Key Equations**: Continuous Approximation formula (BHH)
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
- $D^{\text{geo}}_{ij}$ is the geographical distance (Euclidean on lat/lon coordinates)
- $D^{\text{prod}}_{ij}$ is the cosine distance between demand composition profiles

**Demand Composition Profile** for customer $i$:

$$\text{profile}_i = \frac{1}{\sum_p d_{ip}} \cdot [w_p \cdot d_{ip}]_{p \in P}$$

where $w_p$ are product-specific weights. In the default implementation, all weights are equal ($w_p = 1/3$ for Dry, Chilled, Frozen).

## Design Decisions

### Why Multiple Clustering Algorithms?

- Different algorithms find different cluster shapes
- No single algorithm dominates across all demand patterns
- "Ensemble" approach: Generate diverse candidates, let optimizer choose
- Computational cost is acceptable when parallelized

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
    cluster_id=42,
    config_id="cfg_001",
    vehicle_type="A",
    customers=["c1", "c2", "c3"],
    total_demand={"Dry": 100, "Chilled": 50, "Frozen": 30},
    centroid_latitude=4.6,
    centroid_longitude=-74.0,
    goods_in_config=["Dry", "Chilled", "Frozen"],
    route_time=4.5,  # hours
    method="kmeans",
    tsp_sequence=[],  # Optional, only if TSP used
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
   - Find nearest neighbor clusters (limited to $n_{neighbors}$ candidates)
   - For each neighbor $k_j$:
     - Check if $k_i \cup k_j$ is feasible
     - If yes, add to cluster set
3. Keep both original and merged clusters (optimizer decides)

**Complexity**: $O(m \cdot n_{neighbors})$ where $m$ = number of small clusters

**Parameters**:
- $\eta$: Small cluster size threshold (default: 5)
- $n_{neighbors}$: Number of nearest candidates to consider (default: 50)

## Implementation Notes

### Code Organization

- **Primary Modules**:
  - `src/fleetmix/clustering/generator.py`: Main orchestration
  - `src/fleetmix/clustering/heuristics.py`: Algorithm implementations
  - `src/fleetmix/merging/core.py`: Merge operations

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

**Parallelization**:
- Cluster generation per configuration is independent
- Uses `joblib.Parallel` with `n_jobs=-1` (all cores)

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
  max_depth: 20
  method: combine  # Options: minibatch_kmeans, kmedoids, agglomerative, gaussian_mixture, combine
  distance: euclidean  # Options: euclidean, composite (only for agglomerative)
  geo_weight: 0.7  # Weight for geographical distance (composite-only)
  demand_weight: 0.3  # Weight for demand distance (composite-only)
  route_time_estimation: 'BHH'  # Options: TSP, BHH

# Merge phase (pre-MILP)
pre_small_cluster_size: 5
pre_nearest_merge_candidates: 50
```

### Filtering Clusters

```python
# Get only large clusters (good vehicle utilization)
large_clusters = [c for c in clusters if len(c.customers) >= 10]

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


---

**Navigation**: [← Vehicle Configs](vehicle_configurations.md) | [↑ Specs Index](../README.md#-module-specifications) | [Route Time →](route_time_estimation.md)

