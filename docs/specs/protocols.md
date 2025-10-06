# Protocol-Based Plugin Architecture

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

FleetMix uses Python Protocols (PEP 544) to define pluggable interfaces for clustering, route time estimation, and MILP solvers.

## Protocols

### 1. Clusterer

```python
class Clusterer(Protocol):
    def fit(
        self,
        customers: pd.DataFrame,
        *,
        context: CapacitatedClusteringContext,
        n_clusters: int,
    ) -> list[int]:
        """Returns list of cluster labels."""
```

**Register with**: `@register_clusterer("name")`

### 2. RouteTimeEstimator

```python
class RouteTimeEstimator(Protocol):
    def estimate_route_time(
        self,
        cluster_customers: pd.DataFrame,
        context: RouteTimeContext,
    ) -> tuple[float, list[str]]:
        """Returns (route_time_hours, customer_sequence)."""
```

**Register with**: `@register_route_time_estimator("name")`

### 3. SolverAdapter

```python
class SolverAdapter(Protocol):
    def get_pulp_solver(self, params: RuntimeParams) -> pulp.LpSolver:
        """Return configured PuLP solver."""
    
    @property
    def name(self) -> str:
        """Solver name."""
    
    @property
    def available(self) -> bool:
        """Check if solver is available."""
```

**Register with**: `@register_solver_adapter("name")`

## Usage

1. **Define your class** implementing the protocol
2. **Decorate with** `@register_*("name")`
3. **Configure** to use it:
   ```yaml
   clustering:
     method: my_custom_method
   ```

## Example

```python
from fleetmix.registry import register_clusterer

@register_clusterer("my_method")
class MyClusterer:
    def fit(self, customers, *, context, n_clusters):
        # Your clustering logic
        labels = ...
        return [int(label) for label in labels]
```

## Implementation

- **Module**: `src/fleetmix/interfaces.py` - Protocol definitions
- **Registry**: `src/fleetmix/registry.py` - Registration system

## See Also

- [clustering.md](clustering.md) - Clusterer usage
- [route_time_estimation.md](route_time_estimation.md) - RouteTimeEstimator usage

---

**Navigation**: [← Configuration](configuration.md) | [↑ Specs Index](../README.md#-module-specifications) | [Data Model →](data_model.md)

