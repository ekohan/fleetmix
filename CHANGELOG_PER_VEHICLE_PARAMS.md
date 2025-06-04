# Per-Vehicle Operational Parameters Refactoring

## Summary

This refactoring moves operational parameters (`avg_speed`, `service_time`, `max_route_time`) from global parameters to per-vehicle specifications, enabling different vehicle types (bikes, trucks, drones) to have different operational characteristics.

## Changes Made

### 1. Data Model Changes

#### VehicleSpec (internal_types.py)
- Added required fields: `avg_speed`, `service_time`, `max_route_time`
- Reordered fields to satisfy dataclass inheritance rules (required fields before optional)

#### VehicleConfiguration (types.py)
- Added the same three fields to the public API type

#### Parameters (config/parameters.py)
- Removed global `avg_speed`, `service_time`, `max_route_time` fields
- Updated `from_yaml()` to require these fields per vehicle in the YAML configuration

#### ClusteringContext (internal_types.py)
- Added `avg_speed`, `service_time`, `max_route_time` fields after `depot`
- These are populated per-vehicle during cluster generation

### 2. Configuration File Format

Old format:
```yaml
vehicles:
  SmallTruck:
    capacity: 1000
    fixed_cost: 300

avg_speed: 30.0
service_time: 25.0
max_route_time: 10.0
```

New format:
```yaml
vehicles:
  SmallTruck:
    capacity: 1000
    fixed_cost: 300
    avg_speed: 30.0
    service_time: 25.0
    max_route_time: 10.0
```

### 3. Code Updates

#### Cluster Generation (clustering/generator.py)
- `process_configuration()` now overrides context with per-vehicle values using `replace()`
- `_get_clustering_context_list()` creates base context with placeholder values (0.0)
- Fixed TSP matrix precomputation to use average speed from all configurations

#### Merge Phase (post_optimization/merge_phase.py)
- `_get_merged_route_time()` now accepts vehicle-specific parameters
- `generate_merge_phase_clusters()` extracts parameters from configurations
- `validate_merged_cluster()` uses vehicle-specific parameters

#### Vehicle Configurations (utils/vehicle_configurations.py)
- Updated to include operational parameters in generated configurations

#### API Conversion (api.py)
- `_to_public_solution()` extracts per-vehicle parameters when converting to public types

#### Save Results (utils/save_results.py)
- Updated to display per-vehicle parameters instead of global ones

#### GUI (gui.py)
- Removed global operational parameters section
- Added per-vehicle parameter inputs in vehicle configuration
- Updated parameter collection to handle per-vehicle values

#### CLI (utils/cli.py)
- Removed `--avg-speed`, `--service-time`, `--max-route-time` arguments
- Updated help text to indicate these are now per-vehicle

### 4. Test Updates

- Updated all test fixtures to include per-vehicle parameters
- Fixed test configurations in:
  - `tests/gui/test_gui.py`
  - `tests/integration/test_benchmarking_workflows.py`
  - `tests/integration/test_cli_workflows.py`
  - `tests/core/test_fsm_scenarios.py`
  - `tests/unit/test_merge_phase_edge_cases.py`
  - `tests/unit/test_optimization_core.py`

### 5. Components Using Placeholder Values

Several components still use placeholder values (marked with TODO comments) as they don't yet have access to vehicle-specific parameters:
- VRP solver (uses first vehicle's parameters)
- CVRP/MCVRP converters (create VehicleSpec with defaults)

These can be addressed in future updates as needed.

## Migration Guide

To update existing configurations:

1. Move `avg_speed`, `service_time`, and `max_route_time` from root level to each vehicle definition
2. Ensure all vehicles have these three parameters defined
3. Update any code that accessed `params.avg_speed`, etc. to get values from vehicle specs

## Benefits

- Different vehicle types can have different speeds (e.g., bikes vs trucks)
- Service times can vary by vehicle type (e.g., drones vs manual delivery)
- Route time limits can be vehicle-specific (e.g., electric vehicles with battery constraints)
- More realistic modeling of heterogeneous fleets

## Breaking Changes

This is a breaking change (v2.0) that affects:
- Configuration file format
- Public API (VehicleConfiguration type)
- Internal data structures
- Most core modules that use operational parameters

No backward compatibility is maintained as requested. 