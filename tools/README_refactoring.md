# Per-Vehicle Parameters Refactoring Scripts

This directory contains scripts to refactor the fleetmix codebase to support per-vehicle operational parameters (avg_speed, service_time, max_route_time) instead of global parameters.

## Scripts

### 1. `migrate_vehicle_yamls.py`
- **Purpose**: Updates all YAML configuration files to include per-vehicle operational parameters
- **What it does**:
  - Adds `avg_speed`, `service_time`, and `max_route_time` to each vehicle in YAML configs
  - Uses default values: avg_speed=30.0, service_time=25.0, max_route_time=10.0
  - Processes all test configs and the default config

### 2. `refactor_per_vehicle_params.py`
- **Purpose**: Comprehensive refactoring script for the initial migration attempt
- **What it does**:
  - Updates YAML configs (calls migrate_vehicle_yamls functionality)
  - Updates VehicleSpec instantiations to include new parameters
  - Attempts to update Parameters usage throughout the codebase
  - Updates test fixtures
  - Updates CVRP/MCVRP converters
  - Updates route time calculations

### 3. `fix_per_vehicle_params.py`
- **Purpose**: Targeted fixes for specific issues after the initial refactoring
- **What it does**:
  - Fixes GUI parameter access (uses placeholder values)
  - Fixes VRP solver to extract params from first vehicle
  - Fixes merge phase to use placeholder values (TODO: needs proper implementation)
  - Fixes clustering generator to use default avg_speed
  - Comments out test assertions on removed Parameters attributes
  - Removes params.max_route_time assignments in converters

### 4. `fix_test_fixtures.py`
- **Purpose**: Fixes remaining test fixture issues
- **What it does**:
  - Updates test_parameters.py to not test removed attributes
  - Fixes GUI tests that expect params to have avg_speed/service_time
  - Fixes test_fsm_scenarios.py params.service_time access
  - Updates public API test VehicleConfiguration instantiation
  - Fixes save results tests VehicleSpec instantiations

## Usage

Run the scripts in this order:
```bash
# 1. First, migrate all YAML files
python tools/migrate_vehicle_yamls.py

# 2. Run the comprehensive refactoring
python tools/refactor_per_vehicle_params.py

# 3. Apply targeted fixes
python tools/fix_per_vehicle_params.py

# 4. Fix remaining test issues
python tools/fix_test_fixtures.py
```

## Current Status

After running all scripts, the following components have been updated:
- ✅ YAML configuration files include per-vehicle parameters
- ✅ VehicleSpec and VehicleConfiguration include the new fields
- ✅ Parameters class no longer has global operational parameters
- ✅ CVRP/MCVRP converters create vehicles with operational parameters
- ✅ Most test fixtures have been updated

## Remaining TODOs

Several components use placeholder values and need proper implementation:
1. **Merge phase** (`src/fleetmix/post_optimization/merge_phase.py`):
   - Currently uses hardcoded values (30.0, 25.0, 10.0)
   - Needs to get values from the vehicle configuration being evaluated

2. **GUI** (`src/fleetmix/gui.py`):
   - Currently uses hardcoded values
   - Needs to aggregate or select appropriate values from vehicle specs

3. **Clustering generator** (`src/fleetmix/clustering/generator.py`):
   - Uses hardcoded avg_speed for distance matrix calculation
   - Should use context-appropriate values

4. **Route time calculations**:
   - Many functions still expect global parameters
   - Need to be updated to accept per-vehicle context

## Breaking Changes

This is a v2.0 breaking change that affects:
- Configuration file format (YAML files must include per-vehicle operational parameters)
- Parameters class API (no longer has avg_speed, service_time, max_route_time)
- VehicleSpec constructor (requires 3 additional parameters)
- Public VehicleConfiguration type (requires 3 additional parameters) 