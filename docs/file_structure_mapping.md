# Fleetmix File Structure Mapping

## Overview

This document maps the file structure changes made during the public API implementation.

## Core Changes

### Type System Refactoring

**Before:**
```
src/fleetmix/
├── core_types.py          # All internal dataclasses
├── __init__.py            # Exported many internal modules
└── api.py                 # Returned internal FleetmixSolution
```

**After:**
```
src/fleetmix/
├── internal_types.py      # Internal dataclasses (renamed from core_types.py)
├── types.py              # NEW: Public API dataclasses
├── __init__.py           # Exports only 10 public symbols
└── api.py                # Returns public FleetmixSolution
```

## Import Mapping

### Internal Types (for package development)

| Old Import | New Import |
|------------|------------|
| `from fleetmix.core_types import FleetmixSolution` | `from fleetmix.internal_types import FleetmixSolution` |
| `from fleetmix.core_types import Cluster` | `from fleetmix.internal_types import Cluster` |
| `from fleetmix.core_types import VehicleSpec` | `from fleetmix.internal_types import VehicleSpec` |
| `from fleetmix.core_types import DepotLocation` | `from fleetmix.internal_types import DepotLocation` |
| `from fleetmix.core_types import ClusteringContext` | `from fleetmix.internal_types import ClusteringContext` |
| `from fleetmix.core_types import RouteTimeContext` | `from fleetmix.internal_types import RouteTimeContext` |
| `from fleetmix.core_types import BenchmarkType` | `from fleetmix.internal_types import BenchmarkType` |
| `from fleetmix.core_types import VRPSolution` | `from fleetmix.internal_types import VRPSolution` |

### Public API (for package users)

| Purpose | Import |
|---------|--------|
| Main optimization | `from fleetmix import optimize` |
| Vehicle configurations | `from fleetmix import VehicleConfiguration` |
| Cluster assignments | `from fleetmix import ClusterAssignment` |
| Solution object | `from fleetmix import FleetmixSolution` |
| Parameters | `from fleetmix import Parameters` |
| Stage 1 | `from fleetmix import generate_vehicle_configurations` |
| Stage 2 | `from fleetmix import generate_feasible_clusters` |
| Stage 3 | `from fleetmix import optimize_fleet_selection` |
| Stage 4 | `from fleetmix import improve_solution` |

## Files Updated

### Source Files (30+ files updated)
- All files importing from `core_types` were updated to import from `internal_types`
- `app.py` was updated to use the public `FleetmixSolution` type
- `api.py` was updated to convert internal solution to public solution

### Test Files
- `tests/cli/test_cli.py` - Updated to use public FleetmixSolution
- All other test files continue to use internal types for detailed testing

### New Files Created
1. `src/fleetmix/types.py` - Public API types
2. `tests/public_api/test_api_surface.py` - Tests for API exports
3. `tests/public_api/test_optimize_return_type.py` - Tests for return types
4. `examples/public_api_demo.py` - Demo of new API usage
5. `docs/public_api_summary.md` - API documentation
6. `docs/file_structure_mapping.md` - This file

## Key Differences

### Internal FleetmixSolution (in `internal_types.py`)
- `selected_clusters`: pandas DataFrame
- Many detailed attributes
- Used internally by optimization algorithms

### Public FleetmixSolution (in `types.py`)
- `selected_clusters`: List[ClusterAssignment]
- `configurations_used`: List[VehicleConfiguration]
- Simplified attributes for ease of use
- Returned by public `optimize()` function 