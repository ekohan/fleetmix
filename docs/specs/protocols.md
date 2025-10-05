# Protocol-Based Plugin Architecture

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

FleetMix uses Python Protocols (PEP 544) to define pluggable interfaces for core algorithmic components. This enables researchers and practitioners to customize behavior without modifying the core codebase.

## Why Protocols?

**Protocols vs Abstract Base Classes**:

| Aspect | Protocols | Abstract Base Classes |
|--------|-----------|----------------------|
| Type checking | Structural (duck typing) | Nominal (inheritance) |
| Registration | Automatic if signature matches | Must inherit explicitly |
| External libraries | Easy to adapt | Requires wrapper classes |
| Flexibility | High | Medium |

**FleetMix choice**: Protocols for maximum flexibility

## Available Protocols

### 1. Clusterer

**Purpose**: Implement custom clustering algorithms

**Interface**:
```python
from typing import Protocol
import pandas as pd

class Clusterer(Protocol):
    def fit(
        self,
        customers: pd.DataFrame,
        *,
        context: CapacitatedClusteringContext,
        n_clusters: int,
    ) -> list[int]:
        """
        Cluster customers.
        
        Args:
            customers: DataFrame with Latitude, Longitude, demand columns
            context: Contains vehicle config, goods list, weights
            n_clusters: Target number of clusters
            
        Returns:
            List of cluster labels (integers), one per customer
        """
        ...
```

**Implementation Example**:
```python
from fleetmix.registry import register_clusterer

@register_clusterer("my_method")
class MyClusterer:
    def fit(self, customers, *, context, n_clusters):
        # Your algorithm here
        labels = your_clustering_logic(customers, n_clusters)
        return [int(label) for label in labels]
```

**Registration**: Use `@register_clusterer(name)` decorator

**See**: [clustering.md](clustering.md) for details

### 2. RouteTimeEstimator

**Purpose**: Implement custom route duration estimation

**Interface**:
```python
class RouteTimeEstimator(Protocol):
    def estimate_route_time(
        self,
        cluster_customers: pd.DataFrame,
        context: RouteTimeContext,
    ) -> tuple[float, list[str]]:
        """
        Estimate route time and sequence.
        
        Args:
            cluster_customers: Customers in cluster
            context: Depot, speed, service time, max time
            
        Returns:
            (route_time_hours, customer_sequence)
        """
        ...
```

**Implementation Example**:
```python
from fleetmix.registry import register_route_time_estimator

@register_route_time_estimator("traffic_aware")
class TrafficAwareEstimator:
    def estimate_route_time(self, cluster_customers, context):
        # Fetch traffic data
        traffic = get_current_traffic()
        
        # Compute base time
        base_time = bhh_formula(cluster_customers, context)
        
        # Adjust for traffic
        adjusted_time = base_time * traffic.multiplier
        
        return adjusted_time, list(cluster_customers['customer_id'])
```

**See**: [route_time_estimation.md](route_time_estimation.md)

### 3. SolverAdapter

**Purpose**: Integrate custom MILP solvers

**Interface**:
```python
import pulp

class SolverAdapter(Protocol):
    def get_pulp_solver(self, params: RuntimeParams) -> pulp.LpSolver:
        """Return configured PuLP solver."""
        ...
    
    @property
    def name(self) -> str:
        """Solver name for logging."""
        ...
    
    @property
    def available(self) -> bool:
        """Check if solver is available."""
        ...
```

**Implementation Example**:
```python
from fleetmix.registry import register_solver

@register_solver("my_solver")
class MySolverAdapter:
    @property
    def name(self) -> str:
        return "MySolver"
    
    @property
    def available(self) -> bool:
        try:
            import my_solver_package
            return True
        except ImportError:
            return False
    
    def get_pulp_solver(self, params):
        return pulp.MY_SOLVER(
            timeLimit=params.solver_time_limit,
            msg=params.verbose,
        )
```

**See**: [optimization.md](optimization.md)

## Registry System

### How Registration Works

1. **Decorator registers** the class in a global registry dict:
```python
CLUSTERER_REGISTRY["my_method"] = MyClusterer
```

2. **Config specifies** which implementation to use:
```yaml
clustering:
  methods:
    - my_method  # Your custom method
```

3. **Pipeline loads** from registry:
```python
ClustererClass = CLUSTERER_REGISTRY[method_name]
clusterer = ClustererClass()
result = clusterer.fit(...)
```

### Multiple Implementations

You can have multiple implementations available simultaneously:

```yaml
clustering:
  methods:
    - kmeans
    - kmedoids
    - my_custom_method
    - another_custom_method
```

All will be used to generate diverse cluster candidates.

## Usage Examples

### Complete Custom Clusterer

```python
# my_plugin.py
from fleetmix.registry import register_clusterer
from fleetmix.interfaces import Clusterer
import pandas as pd
from sklearn.cluster import DBSCAN

@register_clusterer("dbscan")
class DBSCANClusterer:
    """DBSCAN clustering for irregular-shaped clusters."""
    
    def fit(
        self,
        customers: pd.DataFrame,
        *,
        context,
        n_clusters: int,
    ):
        # Use geographic coordinates
        X = customers[['Latitude', 'Longitude']].values
        
        # DBSCAN doesn't use n_clusters, use eps instead
        eps = 0.1  # Could parameterize this
        
        model = DBSCAN(eps=eps, min_samples=2)
        labels = model.fit_predict(X)
        
        # DBSCAN uses -1 for noise, map to valid cluster
        labels = [max(0, int(label)) for label in labels]
        
        return labels
```

**Usage**:
```python
# In your main script
import my_plugin  # Registers the clusterer

# Now use it
import fleetmix as fm
solution = fm.optimize(
    demand="customers.csv",
    config="config_with_dbscan.yaml"
)
```

### Custom Route Time with External API

```python
from fleetmix.registry import register_route_time_estimator
import requests

@register_route_time_estimator("google_maps")
class GoogleMapsEstimator:
    """Use Google Maps API for realistic travel times."""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    
    def estimate_route_time(self, cluster_customers, context):
        # Build waypoints
        waypoints = cluster_customers[['Latitude', 'Longitude']].values
        
        # Call Google Maps Directions API
        response = requests.post(
            "https://maps.googleapis.com/maps/api/directions/json",
            params={
                "origin": f"{context.depot.latitude},{context.depot.longitude}",
                "destination": f"{context.depot.latitude},{context.depot.longitude}",
                "waypoints": "|".join(
                    f"{lat},{lon}" for lat, lon in waypoints
                ),
                "key": self.api_key,
            }
        )
        
        data = response.json()
        duration_seconds = data['routes'][0]['legs'][0]['duration']['value']
        duration_hours = duration_seconds / 3600
        
        # Add service time
        n_customers = len(cluster_customers)
        service_hours = (n_customers * context.service_time) / 60
        
        total_time = duration_hours + service_hours
        sequence = list(cluster_customers['customer_id'])
        
        return total_time, sequence
```

## Advanced Topics

### Type Checking with MyPy

Protocols enable static type checking:

```python
def my_function(clusterer: Clusterer) -> list[int]:
    # MyPy verifies that clusterer has fit() method
    return clusterer.fit(...)
```

### Runtime Protocol Checking

```python
from typing import runtime_checkable

@runtime_checkable
class Clusterer(Protocol):
    ...

# Check at runtime
if isinstance(my_object, Clusterer):
    result = my_object.fit(...)
```

### Extending Core Types

To add custom fields to context objects:

```python
from dataclasses import dataclass, replace

@dataclass
class MyEnhancedContext:
    # Include all original fields
    depot: DepotLocation
    avg_speed: float
    # ... other fields from RouteTimeContext
    
    # Add custom fields
    traffic_data: dict
    weather_conditions: str

# Use in your estimator
def estimate_route_time(self, customers, context):
    enhanced = MyEnhancedContext(**context.__dict__, 
                                 traffic_data=fetch_traffic(),
                                 weather_conditions="rainy")
    ...
```

## Testing Custom Implementations

### Unit Test Template

```python
# test_my_clusterer.py
import pytest
from my_plugin import DBSCANClusterer

def test_dbscan_clusterer():
    """Test DBSCAN implementation."""
    clusterer = DBSCANClusterer()
    
    # Prepare test data
    customers = pd.DataFrame({
        'customer_id': ['c1', 'c2', 'c3'],
        'Latitude': [4.6, 4.61, 4.8],
        'Longitude': [-74.0, -74.01, -74.3],
    })
    
    context = make_test_context()
    
    # Run
    labels = clusterer.fit(customers, context=context, n_clusters=2)
    
    # Verify
    assert len(labels) == 3
    assert all(isinstance(label, int) for label in labels)
    assert all(label >= 0 for label in labels)
```

### Integration Test

```python
def test_custom_clusterer_in_pipeline():
    """Test custom clusterer works end-to-end."""
    import my_plugin  # Register
    
    solution = fm.optimize(
        demand=test_customers,
        config=config_with_custom_method,
    )
    
    assert solution.total_cost > 0
    assert len(solution.selected_clusters) > 0
```

## References

### Related Modules

- [clustering.md](clustering.md): Clusterer protocol usage
- [route_time_estimation.md](route_time_estimation.md): RouteTimeEstimator protocol
- [optimization.md](optimization.md): SolverAdapter protocol

### External Documentation

- [PEP 544 - Protocols](https://peps.python.org/pep-0544/)
- [MyPy Protocols](https://mypy.readthedocs.io/en/stable/protocols.html)

---

**Navigation**: [← Configuration](configuration.md) | [↑ Specs Index](../README.md#-module-specifications) | [Data Model →](data_model.md)

