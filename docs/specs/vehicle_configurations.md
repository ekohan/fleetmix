# Vehicle Configurations

> **Status**: Stable  
> **Last Updated**: 2025-10-05

## Purpose

Generates all feasible combinations of vehicle types and compartment subsets. First phase of the matheuristic pipeline (Paper §4.1).

A **vehicle configuration** is a vehicle type with a specific subset of compartments. For example, a truck with only dry+chilled compartments vs. the same truck with all three compartments.

## Mathematical Formulation

For each vehicle type and non-empty compartment subset $M' \subseteq M$:

$$v = (\text{vehicle type}, M', Q, T, \text{costs})$$

**Total configurations**: $r \cdot (2^{|M|} - 1)$ where $r$ = vehicle types, $|M|$ = compartments

**Allowed goods constraint**: $M'_v \subseteq M_{\text{allowed}}^{\text{vehicle type}}$ (reduces configurations for specialized vehicles)

## Key Design Choices

1. **Complete enumeration** vs. dynamic configuration
   - Enables tractable MILP formulation
   - Allows optimizer to choose specialized vs. versatile vehicles
   - Negligible cost: 5 types × 5 compartments = 155 configurations

2. **Flexible compartments** vs. fixed capacities
   - Total capacity $Q_v$ divided post-optimization to match demand
   - Realistic for modern adjustable-partition vehicles
   - Reduces configuration explosion

## Implementation

**Location**: `src/fleetmix/utils/vehicle_configurations.py`  
**Function**: `generate_vehicle_configurations()` (54 lines)  
**Data types**: `VehicleConfiguration`, `VehicleSpec` in `src/fleetmix/core_types.py`

**Algorithm**:
```python
for vehicle_type in vehicle_types:
    allowed_goods = vehicle_type.allowed_goods or all_goods
    for option in itertools.product([0,1], repeat=len(allowed_goods)):
        if sum(option) > 0:  # Skip empty configuration
            create_configuration(vehicle_type, option)
```

## Usage

```python
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.core_types import VehicleSpec

vehicles = {
    "Truck": VehicleSpec(capacity=2700, fixed_cost=100, allowed_goods=["Dry", "Chilled"])
}
goods = ["Dry", "Chilled", "Frozen"]

configs = generate_vehicle_configurations(vehicles, goods)
# Generates 3 configurations: {Dry}, {Chilled}, {Dry,Chilled}
```

## References

- **[data_model.md](data_model.md)**: Data structures
- **[clustering.md](clustering.md)**: Next pipeline phase
- **[optimization.md](optimization.md)**: Uses configurations

---

**Navigation**: [← Architecture](../ARCHITECTURE.md) | [↑ Specs Index](../README.md#-module-specifications) | [Clustering →](clustering.md)

