# Route Time Estimation

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

Estimates route duration for serving a cluster. Used for feasibility checking (Phase 2) and cost computation (Phase 3).

**Methods**: BHH (fast approximation), TSP (exact via PyVRP)

## Paper Connection

- **Primary Reference**: Paper §4.2 "Computation of vehicle route durations"
- **Formula**: $t_{vk} \approx \alpha_{vk} + 2 \cdot \delta_{vk} + \beta \cdot \sqrt{n \cdot A} + \gamma \cdot n$

## Methods

### BHH (Beardwood-Halton-Hammersley)

**Formula**: $t_{vk} = \text{setup} + 2 \cdot \text{depot\_travel} + \beta \cdot \sqrt{n \cdot A} / \text{speed} + \text{service} \cdot n$

- $\beta = 0.765$ (constant in implementation)
- $A$ = cluster area (approximated as $\pi \cdot r^2$ where $r$ = max distance from centroid)
- Fast: ~ms per cluster

### TSP (PyVRP Solver)

Solves exact TSP and converts distance to time. 
- Slower
- Provides optimal customer sequence


## Function

```python
def estimate_route_time(
    cluster_customers: pd.DataFrame,
    context: RouteTimeContext,
) -> tuple[float, list[str]]:
    """Returns (route_time_hours, customer_sequence)."""
```

**Configuration**:
```yaml
clustering:
  route_time_estimation: 'BHH'  # Options: BHH, TSP
```


## Implementation

- **Module**: `src/fleetmix/utils/route_time.py`
- **Classes**: `BHHEstimator`, `TSPEstimator`
- **Helpers**: `make_rt_context()`, `build_distance_duration_matrices()`
- **Dependencies**: `haversine`, `pyvrp`, `numpy`, `pandas`
- **Caching**: Distance matrices cached globally (keyed by `avg_speed`)

## References

- **Beardwood, Halton, Hammersley (1959)** - *The shortest path through many points*: BHH formula
- **Wouda, Lan, Kool (2024)** - *PyVRP: A high-performance VRP solver package*: TSP solver used

## See Also

- [clustering.md](clustering.md) - Uses for feasibility checks
- [protocols.md](protocols.md) - RouteTimeEstimator protocol

---

**Navigation**: [← Clustering](clustering.md) | [↑ Specs Index](../README.md#-module-specifications) | [Optimization →](optimization.md)

