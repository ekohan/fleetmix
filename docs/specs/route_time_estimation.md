# Route Time Estimation

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

Route time estimation provides the duration required to serve a cluster of customers with a given vehicle configuration. This is critical for checking route duration feasibility during cluster generation (Phase 2) and cost computation in the optimization phase (Phase 3).

Two methods are available:
1. **BHH (Beardwood-Halton-Hammersley)**: Fast continuous approximation
2. **TSP**: Exact solution using PyVRP solver

## Paper Connection

- **Primary Reference**: Paper §4.2 "Computation of vehicle route durations"
- **Key Equation**: BHH formula for route time approximation
- **Trade-off Discussion**: Paper mentions both CA and solver-based approaches

## Mathematical Formulation

### Notation

| Symbol | Description |
|--------|-------------|
| $t_{vk}$ | Route time for cluster $k$ with vehicle config $v$ (hours) |
| $\alpha_{vk}$ | Setup time for dispatching vehicle |
| $\delta_{vk}$ | Line-haul time (depot to cluster centroid) |
| $\beta$ | BHH constant (≈ 0.75 for Euclidean space) |
| $n$ | Number of customers in cluster |
| $A$ | Service area (km²) |
| $\gamma$ | Service time per customer (minutes) |

### BHH Continuous Approximation

**Formula** (Paper §4.2):

$$t_{vk} \approx \alpha_{vk} + 2 \cdot \delta_{vk} + \beta \cdot \sqrt{n \cdot A} + \gamma \cdot n$$

**Components**:

1. **Setup time** $\alpha_{vk}$: Fixed time to dispatch vehicle (e.g., loading at depot)
2. **Line-haul** $2 \cdot \delta_{vk}$: Round trip to cluster centroid
   $$\delta_{vk} = \frac{\text{distance(depot, centroid)}}{\text{avg\_speed}}$$
3. **Within-cluster routing** $\beta \cdot \sqrt{n \cdot A}$: Estimated tour length within cluster
4. **Service time** $\gamma \cdot n$: Total stop duration

**Derivation**: Beardwood, Halton, Hammersley (1959) proved that the TSP tour length through $n$ random points in area $A$ is approximately $\beta \cdot \sqrt{n \cdot A}$ with $\beta \approx 0.75$.

### TSP Exact Solution

Uses PyVRP genetic algorithm to solve the Traveling Salesman Problem:

$$\min \sum_{(i,j) \in \text{tour}} d_{ij}$$

subject to visiting each customer exactly once.

**Time conversion**:
$$t_{vk} = \frac{\text{tour\_distance}}{\text{avg\_speed}} + \gamma \cdot n$$

## Design Decisions

### Why Two Methods?

| Aspect | BHH | TSP |
|--------|-----|-----|
| **Speed** | Very fast (~ms) | Slower (~seconds per cluster) |
| **Accuracy** | Approximation (±10-20%) | Exact tour |
| **Scalability** | Handles 1000s of clusters | Feasible but slow |
| **Use case** | Large-scale problems, fast iteration | Small problems, research validation |

**Default choice**: BHH for production, TSP for benchmarking

**Trade-off**: Paper §4.2 notes that BHH is sufficient for tactical fleet design since routing is not explicitly modeled.

### BHH Constant $\beta$

**Theoretical value**: $\beta \approx 0.75$ for 2D Euclidean space

**Empirical calibration**: May adjust based on:
- Urban grid vs irregular street network
- Traffic patterns
- Historical route data

**Current implementation**: $\beta = 0.75$ (default)

### Centroid vs Medoid

For line-haul distance $\delta_{vk}$:
- **Current**: Use geographic centroid (mean lat/lon)
- **Alternative**: Use medoid (customer closest to centroid)

**Rationale**: Centroid is faster to compute and sufficient for approximation.

## Interfaces

### Protocol Definition

```python
class RouteTimeEstimator(Protocol):
    """Protocol for route time estimation algorithms."""
    
    def estimate_route_time(
        self,
        cluster_customers: pd.DataFrame,
        context: RouteTimeContext,
    ) -> tuple[float, list[str]]:
        """
        Estimate route time for serving a cluster.
        
        Args:
            cluster_customers: DataFrame with customer locations and IDs
            context: Route time context (depot, speed, service time, etc.)
            
        Returns:
            (route_time_hours, customer_sequence)
        """
        ...
```

### Input

```python
cluster_customers: pd.DataFrame
# Columns: customer_id, Latitude, Longitude

context: RouteTimeContext
# Fields:
RouteTimeContext(
    depot=DepotLocation(latitude=4.6, longitude=-74.08),
    avg_speed=30.0,  # km/h
    service_time=25.0,  # minutes per stop
    max_route_time=10.0,  # hours
    prune_tsp=False,
)
```

### Output

```python
(route_time: float, sequence: list[str])

# Example:
(6.5, ["c1", "c5", "c3", "c2", "c4"])  # 6.5 hours, in this sequence
```

## Key Algorithms

### BHH Estimator

**Purpose**: Fast route time approximation

**Steps**:

1. **Compute setup time** (typically 0 in current implementation):
```python
setup_time = 0.0  # Can be parameterized
```

2. **Compute centroid**:
```python
centroid_lat = cluster_customers['Latitude'].mean()
centroid_lon = cluster_customers['Longitude'].mean()
```

3. **Compute line-haul distance** (Haversine formula):
```python
depot_to_centroid_km = haversine(
    (depot.latitude, depot.longitude),
    (centroid_lat, centroid_lon)
)
```

4. **Compute line-haul time** (round trip):
```python
line_haul_hours = (2 * depot_to_centroid_km) / avg_speed
```

5. **Compute service area** $A$:
```python
# Convex hull or bounding box
lats = cluster_customers['Latitude']
lons = cluster_customers['Longitude']
A_km2 = compute_area(lats, lons)  # Simplified rectangular area
```

6. **Compute within-cluster tour** (BHH formula):
```python
n = len(cluster_customers)
beta = 0.75
tour_km = beta * sqrt(n * A_km2)
tour_hours = tour_km / avg_speed
```

7. **Compute service time**:
```python
service_hours = (n * service_time_minutes) / 60.0
```

8. **Total time**:
```python
total_time = setup_time + line_haul_hours + tour_hours + service_hours
```

**Complexity**: $O(n)$ where $n$ = cluster size

**Accuracy**: Typically within 10-20% of actual TSP solution

### TSP Estimator

**Purpose**: Exact route time via solving TSP

**Steps**:

1. **Build distance matrix** (cached for efficiency):
```python
D = pairwise_haversine_distances(customers + [depot])
```

2. **Convert to duration matrix**:
```python
T = D / avg_speed
```

3. **Solve TSP using PyVRP**:
```python
model = Model()
# Add depot and clients
# Define vehicle type
# Solve with genetic algorithm
solution = model.solve(...)
```

4. **Extract tour and duration**:
```python
tour_distance = solution.distance()
tour_time = tour_distance / avg_speed
```

5. **Add service time**:
```python
total_time = tour_time + (n * service_time) / 60.0
```

**Complexity**: $O(n^2)$ for distance matrix, then GA (thousands of iterations)

**Typical runtime**: 0.5-2 seconds per cluster

## Implementation Notes

### Code Organization

- **Primary Module**: `src/fleetmix/utils/route_time.py` (631 lines)
- **Key Classes/Functions**:
  - `BHHRouteTimeEstimator`: BHH implementation
  - `TSPRouteTimeEstimator`: PyVRP-based TSP solver
  - `make_rt_context()`: Factory for context objects
  - `build_distance_duration_matrices()`: Precompute for TSP

### Dependencies

- **Internal**: `fleetmix.core_types`, `fleetmix.registry`
- **External**:
  - `haversine`: Geographic distance computation
  - `pyvrp`: TSP solver (state-of-the-art VRP solver)
  - `numpy`, `pandas`: Data structures

### Performance Considerations

**BHH**:
- Per-cluster time: < 1ms
- Suitable for 1000s of clusters
- Bottleneck: None (negligible)

**TSP**:
- Per-cluster time: 0.5-2s
- Distance matrix cache: $O(n^2)$ memory
- Bottleneck: Can dominate runtime for large problems

**Caching strategy**:
- Distance matrices cached globally
- Keyed by `avg_speed` (different vehicles may have different speeds)
- Built once at start if TSP method is used

### Edge Cases

1. **Single customer**: Line-haul only, no within-cluster tour
2. **Customers at depot**: Zero distance, only service time
3. **Large clusters** (n > 100): TSP becomes slow, BHH preferred
4. **Negative times**: Validation catches invalid inputs

## Usage Examples

### Using BHH (Default)

```python
from fleetmix.utils.route_time import BHHRouteTimeEstimator, make_rt_context
from fleetmix.core_types import DepotLocation
import pandas as pd

# Create context
depot = DepotLocation(latitude=4.6, longitude=-74.08)
context = make_rt_context(vehicle_config, depot, prune_tsp=False)

# Prepare cluster data
cluster_df = pd.DataFrame({
    'customer_id': ['c1', 'c2', 'c3'],
    'Latitude': [4.7, 4.65, 4.72],
    'Longitude': [-74.1, -74.05, -74.12],
})

# Estimate route time
estimator = BHHRouteTimeEstimator()
route_time, sequence = estimator.estimate_route_time(cluster_df, context)

print(f"Estimated route time: {route_time:.2f} hours")
# Note: BHH doesn't provide sequence, returns customer_ids as-is
```

### Using TSP

```python
from fleetmix.utils.route_time import (
    TSPRouteTimeEstimator, 
    build_distance_duration_matrices,
)

# Precompute distance matrices (once)
build_distance_duration_matrices(
    customers_df=all_customers,
    depot={'latitude': 4.6, 'longitude': -74.08},
    avg_speed=30.0,
)

# Estimate with TSP
estimator = TSPRouteTimeEstimator()
route_time, sequence = estimator.estimate_route_time(cluster_df, context)

print(f"Exact route time: {route_time:.2f} hours")
print(f"Optimal sequence: {sequence}")
```

### Configuration

```yaml
# config.yaml
route_time:
  method: bhh  # or 'tsp'
  
  bhh:
    beta: 0.75  # BHH constant
  
  tsp:
    max_iterations: 10000
    population_size: 50
```

### Choosing Method

```python
# In your code
if len(customers) > 500 or time_budget_seconds < 60:
    method = "bhh"  # Fast approximation
else:
    method = "tsp"  # Exact solution
```

## Extension Points

### Custom Route Time Estimator

Implement the protocol:

```python
from fleetmix.registry import register_route_time_estimator

@register_route_time_estimator("traffic_aware")
class TrafficAwareEstimator:
    """Route time estimation with real-time traffic."""
    
    def estimate_route_time(
        self,
        cluster_customers: pd.DataFrame,
        context: RouteTimeContext,
    ) -> tuple[float, list[str]]:
        # Fetch traffic data
        traffic_multiplier = get_traffic_multiplier(time_of_day)
        
        # Use BHH as base
        base_time, seq = bhh_estimate(cluster_customers, context)
        
        # Adjust for traffic
        adjusted_time = base_time * traffic_multiplier
        
        return adjusted_time, seq
```

### Calibrating BHH Constant

To fit $\beta$ to historical data:

```python
# Collect actual route times
actual_times = [...]
bhh_estimates = [...]

# Optimize β
from scipy.optimize import minimize_scalar

def objective(beta):
    errors = [
        abs(actual - bhh_formula(cluster, beta))
        for actual, cluster in zip(actual_times, clusters)
    ]
    return sum(errors)

result = minimize_scalar(objective, bounds=(0.5, 1.0))
optimal_beta = result.x
```

### Custom Extensions

For problem-specific constraints, implement custom estimator:

```python
def estimate_route_time_custom(cluster, context):
    # Your custom route time logic
    # Can incorporate any constraints needed
    ...
```

See paper §7 for discussion of problem variants (TODO).

## Testing

### Unit Tests

- **Location**: `tests/unit/test_route_time.py`
- **Coverage**:
  - BHH formula correctness
  - TSP solution validity
  - Edge cases (single customer, collocated customers)
  - Cache consistency

### Integration Tests

- **Location**: `tests/integration/test_route_feasibility.py`
- **Scenarios**:
  - Route times within max limits
  - BHH vs TSP comparison on small instances

## Comparison with Literature

### Beardwood, Halton, Hammersley (1959)

**Original result**: TSP tour length ≈ $\beta \sqrt{n \cdot A}$ with $\beta \approx 0.75$

**Our usage**: Extended to include line-haul and service time

**Accuracy**: Confirmed in subsequent literature (Daganzo 1984, Silva-Febre et al. 2022)

### Practical VRP Solvers

- **PyVRP**: State-of-the-art solver (Wouda et al. 2024)
- **OR-Tools**: Google's solver (also viable alternative)
- **LKH**: Classical TSP heuristic (Helsgaun 2000)

**Choice**: PyVRP for its Python integration and MCV support

## References

### Related Modules

- **[clustering.md](clustering.md)**: Uses route time for feasibility checks
- **[optimization.md](optimization.md)**: Uses route time for cost computation
- **[data_model.md](data_model.md)**: Defines `RouteTimeContext`
- **[protocols.md](protocols.md)**: `RouteTimeEstimator` protocol

### Literature

1. **Beardwood, Halton, Hammersley (1959)** - *The shortest path through many points*: BHH formula
2. **Daganzo (1984)** - *The distance traveled to visit N points...*: Extension to urban routing
3. **Silva-Febre et al. (2022)** - *Tactical demand-aware routing*: Validation of BHH accuracy
4. **Wouda et al. (2024)** - *PyVRP: A high-performance VRP solver*: TSP solver used

### External Documentation

- [PyVRP Documentation](https://pyvrp.readthedocs.io/)
- [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula)

## See Also

- [← Clustering](clustering.md)
- [Next: Optimization →](optimization.md)
- [↑ Architecture](../ARCHITECTURE.md)
- [Docs Home](../README.md)

---

**Navigation**: [← Clustering](clustering.md) | [↑ Specs Index](../README.md#-module-specifications) | [Optimization →](optimization.md)

