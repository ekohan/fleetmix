# Vehicle Configurations

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

The vehicle configuration module generates all feasible combinations of vehicle types and compartment subsets. This is the first phase of the matheuristic pipeline and establishes the solution space for fleet design optimization.

A **vehicle configuration** is a specific vehicle type equipped with a particular subset of compartments. For example, a truck configured with only dry and chilled compartments (but not frozen) is a different configuration than the same truck with all three compartments.

## Paper Connection

- **Primary Reference**: Paper §4.1 "Generate Vehicle Configurations"
- **Related Sections**: §3 "Problem Definition" (vehicle and compartment definitions)
- **Key Concept**: Enumerating $2^{|M|}-1$ configurations per vehicle type

## Mathematical Formulation

### Notation

| Symbol | Description |
|--------|-------------|
| $V$ | Set of all vehicle configurations |
| $M$ | Set of available compartment types (e.g., dry, chilled, frozen) |
| $r$ | Number of vehicle types available |
| $Q_v$ | Maximum load capacity of configuration $v \in V$ |
| $T_v$ | Maximum route duration for configuration $v \in V$ |

### Generation Rule

For each vehicle type and each non-empty subset $M' \subseteq M$, create a configuration:

$$v = (\text{vehicle\_type}, M', Q, T, \text{costs})$$

where:
- The compartment subset $M'$ determines which product types the configuration can carry
- Capacity $Q$ and time limit $T$ are inherited from the vehicle type
- Costs include fixed cost (vehicle + compartments) and variable cost (per distance/time)

**Total configurations**: $r \cdot (2^{|M|} - 1)$

For example, with $r=3$ vehicle types and $|M|=3$ compartments:
- Configurations per type: $2^3 - 1 = 7$
- Total configurations: $3 \times 7 = 21$

### Allowed Goods Constraint

In practice, not all vehicles can carry all product types. The `allowed_goods` constraint restricts which compartments a vehicle type can have:

$$M'_v \subseteq M_{\text{allowed}}^{\text{vehicle\_type}}$$

This reduces the total number of configurations when certain vehicle types are specialized (e.g., a refrigerated truck that cannot carry dry goods).

## Design Decisions

### Why Enumerate All Configurations?

**Alternative approaches**:
1. **Dynamic configuration**: Decide compartments during optimization
2. **Pre-select configurations**: Only create "likely useful" ones

**Our choice: Complete enumeration**

**Rationale**:
- Makes the MILP formulation tractable (set covering with fixed set $V$)
- Allows the optimizer to choose between specialized (few compartments) vs. versatile (many compartments) vehicles
- Enables comparison with single-compartment vehicle (SCV) fleets naturally (SCV configs are singletons of $M$)
- Computational cost is negligible: even with 5 vehicle types and 5 compartments, we get only $5 \times 31 = 155$ configurations

### Compartment Capacity Flexibility

Unlike Henke et al. (2015) who model fixed compartment capacities, we allow **flexible compartments**:
- Each configuration has total capacity $Q_v$
- How that capacity is divided among compartments is determined **post-optimization**
- The optimization model (§4.3) selects cluster-configuration pairs
- After selection, compartment capacities are set to match cluster demand

**Benefits**:
- Realistic: Modern MCVs have adjustable partitions
- Reduces configurations: Don't need separate configs for "60% dry, 40% chilled" vs. "40% dry, 60% chilled"
- Cited by Ostermeier & Hübner (2018) as important for practical food distribution

## Interfaces

### Input

```python
vehicle_types: dict[str, VehicleSpec]
# Example:
{
    "TruckA": VehicleSpec(
        capacity=2700,
        fixed_cost=100,
        avg_speed=30,
        service_time=25,
        max_route_time=10,
        allowed_goods=["Dry", "Chilled"]  # Optional constraint
    ),
    "TruckB": VehicleSpec(
        capacity=4500,
        fixed_cost=225,
        avg_speed=30,
        service_time=25,
        max_route_time=10,
        allowed_goods=None  # Can carry all goods
    )
}

goods: list[str]
# Example: ["Dry", "Chilled", "Frozen"]
```

### Output

```python
list[VehicleConfiguration]
# Where each VehicleConfiguration has:
VehicleConfiguration(
    config_id="cfg_001",
    vehicle_type="TruckA",
    capacity=2700,
    fixed_cost=100,
    compartments={"Dry": True, "Chilled": True, "Frozen": False},
    avg_speed=30,
    service_time=25,
    max_route_time=10,
)
```

### Data Types

```python
@dataclass
class VehicleConfiguration:
    config_id: str              # Unique identifier
    vehicle_type: str           # Original vehicle type name
    capacity: float             # Total load capacity (kg)
    fixed_cost: float           # Daily fixed cost
    compartments: dict[str, bool]  # Which compartments are active
    avg_speed: float            # km/h
    service_time: float         # minutes per stop
    max_route_time: float       # hours
```

## Key Algorithms

### Configuration Generation

**Purpose**: Enumerate all feasible vehicle-compartment combinations

**Complexity**: $O(r \cdot 2^{|M|})$ where $r$ = # vehicle types, $|M|$ = # compartments

**Steps**:

1. **For each vehicle type**:
   ```
   For vt in vehicle_types:
       allowed = vt.allowed_goods if specified else all_goods
   ```

2. **Generate compartment options**:
   ```
   # Binary vector: each position = include compartment or not
   options = itertools.product([0, 1], repeat=len(allowed))
   ```

3. **Create configurations** (excluding empty set):
   ```
   For option in options:
       if sum(option) > 0:  # At least one compartment
           config = VehicleConfiguration(...)
           configurations.append(config)
   ```

4. **Assign unique IDs**: `cfg_001`, `cfg_002`, ...

**Implementation Notes**: 
- Uses `itertools.product` for efficient enumeration
- Respects `allowed_goods` constraint by only generating valid combinations
- Initializes all goods to `False`, then sets allowed goods based on option

### Example

**Input**:
- Vehicle types: `["TruckA", "TruckB"]`
- Goods: `["Dry", "Chilled", "Frozen"]`
- TruckA `allowed_goods`: `["Dry", "Chilled"]`
- TruckB `allowed_goods`: `None` (all goods)

**Output** (11 configurations total):

TruckA (3 configurations from $2^2 - 1$):
1. `{Dry: True, Chilled: False, Frozen: False}`
2. `{Dry: False, Chilled: True, Frozen: False}`
3. `{Dry: True, Chilled: True, Frozen: False}`

TruckB (7 configurations from $2^3 - 1$):
4. `{Dry: True, Chilled: False, Frozen: False}`
5. `{Dry: False, Chilled: True, Frozen: False}`
6. `{Dry: False, Chilled: False, Frozen: True}`
7. `{Dry: True, Chilled: True, Frozen: False}`
8. `{Dry: True, Chilled: False, Frozen: True}`
9. `{Dry: False, Chilled: True, Frozen: True}`
10. `{Dry: True, Chilled: True, Frozen: True}`

## Implementation Notes

### Code Organization

- **Primary Module**: `src/fleetmix/utils/vehicle_configurations.py`
- **Key Function**: 
  - `generate_vehicle_configurations()`: Main entry point (54 lines)
- **Data Types**: `src/fleetmix/core_types.py`
  - `VehicleConfiguration`: Output type
  - `VehicleSpec`: Input type

### Dependencies

- **Internal**: `fleetmix.core_types`, `fleetmix.utils.common`
- **External**: `itertools` (standard library)

### Performance Considerations

- **Time complexity**: Negligible for practical fleet sizes (< 1ms for typical inputs)
- **Space complexity**: $O(r \cdot 2^{|M|})$ configurations stored in memory
- **Bottleneck**: Not a bottleneck; clustering phase dominates runtime

### Edge Cases

1. **No allowed goods specified**: Treats as "can carry all goods"
2. **Empty allowed goods**: Would generate 0 configurations (validation should catch this)
3. **Single compartment**: Effectively creates SCV fleet
4. **All compartments**: Creates MCV fleet

## Usage Examples

### Basic Usage

```python
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.core_types import VehicleSpec

# Define vehicle types
vehicles = {
    "SmallTruck": VehicleSpec(
        capacity=2700,
        fixed_cost=100,
        avg_speed=30,
        service_time=25,
        max_route_time=10,
    ),
    "LargeTruck": VehicleSpec(
        capacity=4500,
        fixed_cost=225,
        avg_speed=30,
        service_time=25,
        max_route_time=12,
    ),
}

goods = ["Dry", "Chilled", "Frozen"]

# Generate all configurations
configs = generate_vehicle_configurations(vehicles, goods)

print(f"Generated {len(configs)} configurations")
# Output: Generated 14 configurations (2 × 7)
```

### With Allowed Goods Constraint

```python
# Specialized fleet
vehicles = {
    "DryTruck": VehicleSpec(
        capacity=2700,
        fixed_cost=100,
        allowed_goods=["Dry"],  # Only dry goods
        ...
    ),
    "ColdTruck": VehicleSpec(
        capacity=3300,
        fixed_cost=175,
        allowed_goods=["Chilled", "Frozen"],  # Only cold goods
        ...
    ),
}

configs = generate_vehicle_configurations(vehicles, goods)

# DryTruck generates 1 configuration: {Dry: True}
# ColdTruck generates 3 configurations: {Chilled}, {Frozen}, {Chilled, Frozen}
# Total: 4 configurations
```

### Filtering Configurations

```python
# Get only multi-compartment configs
mcv_configs = [
    cfg for cfg in configs 
    if sum(cfg.compartments.values()) > 1
]

# Get only single-compartment (SCV) configs
scv_configs = [
    cfg for cfg in configs 
    if sum(cfg.compartments.values()) == 1
]
```

## Extension Points

### Adding Fixed Compartments

To implement fixed compartment capacities (like Henke et al. 2015):

1. **Modify `VehicleConfiguration`** to include compartment sizes:
```python
@dataclass
class VehicleConfiguration:
    ...
    compartment_capacities: dict[str, float]  # NEW: fixed sizes
```

2. **Generate capacity variants** in addition to presence/absence:
```python
# Instead of binary [0,1], use capacity levels
capacity_levels = [0, 0.3, 0.5, 0.7]  # % of total capacity
for combo in itertools.product(capacity_levels, repeat=len(goods)):
    if sum(combo) == 1.0:  # Must sum to 100%
        # Create config with this capacity split
```

**Trade-off**: Exponentially more configurations, but models physical constraints

### Adding Compartment Setup Costs

To penalize using more compartments:

1. **Add to `VehicleSpec`**:
```python
compartment_setup_cost: float  # Cost per active compartment
```

2. **Update configuration fixed cost**:
```python
config.fixed_cost = (
    vt_info.fixed_cost + 
    sum(compartments.values()) * vt_info.compartment_setup_cost
)
```

This is mentioned in Ostermeier & Hübner (2018) and can affect MCV vs SCV trade-off.

## Testing

### Unit Tests

- **Location**: `tests/unit/test_vehicle_configurations.py`
- **Coverage**:
  - Correct number of configurations generated
  - `allowed_goods` constraint respected
  - Configuration IDs are unique
  - Edge cases (single compartment, all compartments)

### Integration Tests

- **Location**: `tests/integration/test_pipeline.py`
- **Scenario**: Configurations feed into clustering phase correctly

## Comparison with Literature

### Ostermeier & Hübner (2018)

**Their approach**: Only two vehicle types (SCV and MCV), homogeneous within type

**FleetMix**: Arbitrary number of heterogeneous vehicle types, each with different:
- Capacity
- Speed
- Time limits  
- Costs
- Allowed goods

**Generalization**: Our approach subsumes theirs as a special case

### Henke et al. (2015)

**Their approach**: Fixed compartment capacities predefined at manufacturing

**FleetMix**: Flexible compartments with capacity determined post-optimization

**Trade-off**: We sacrifice physical realism for computational tractability and match modern adjustable-partition vehicles

## References

### Related Modules

- **[data_model.md](data_model.md)**: Defines `VehicleConfiguration` and `VehicleSpec`
- **[clustering.md](clustering.md)**: Uses configurations to generate clusters
- **[optimization.md](optimization.md)**: Selects optimal configuration subset

### Literature

1. **Ostermeier & Hübner (2018)** - *Vehicle selection in a city distribution center*: Two-type fleet (SCV/MCV)
2. **Henke et al. (2015)** - *Multi-compartment vehicle routing with flexible compartments*: Fixed capacities
3. **Beardwood, Halton, Hammersley (1959)**: Continuous approximation (used in route time)

## See Also

- [← Back to Architecture](../ARCHITECTURE.md)
- [Next: Clustering →](clustering.md)
- [Docs Home](../README.md)
- [Paper Mapping](../mapping.md)

---

**Navigation**: [← Architecture](../ARCHITECTURE.md) | [↑ Specs Index](../README.md#-module-specifications) | [Clustering →](clustering.md)

