# CLAUDE.md — FleetMix Development Guide

> This file provides guidance for Claude Code when working with the FleetMix codebase.

## Project Overview

FleetMix is a **production-ready, research-grade Python library** for multi-compartment vehicle fleet optimization. It implements a *cluster-first → MILP-second* matheuristic for heterogeneous fleet design in urban food distribution.

**Core workflow**: Demand data → Vehicle configurations → Feasible clusters → MILP optimization → Post-optimization merge → Fleet solution

## Quick Reference

**Always use `uv run` to execute commands** — it handles the virtual environment automatically.

```bash
# Run tests (ALWAYS run before concluding changes)
uv run pytest                              # Full test suite
uv run pytest tests/unit/                  # Unit tests only  
uv run pytest -k "test_name" -x            # Single test, stop on failure
uv run pytest --cov=src --cov-report=term-missing  # With coverage

# Type checking
uv run mypy src/fleetmix/

# Linting and formatting
uv run ruff check src/
uv run ruff format src/

# Run the CLI
uv run fleetmix --help
uv run fleetmix optimize --demand examples/bogota_demand.csv
uv run fleetmix gui                        # Web interface

# Sync dependencies (when pyproject.toml changes)
uv sync --all-extras
```

## Engineering Philosophy

**Gold standard, simple solutions.** This codebase is open-source and aims for research reproducibility. Channel your inner Jeff Dean: elegant, performant, maintainable code.

- **Don't overcomplicate** — good engineering is simple engineering
- **Spec-driven** — modules connect paper math to implementation
- **Protocol-based** — structural typing over inheritance
- **Type-safe** — full type hints, mypy-validated

## Code Architecture

### Directory Structure

```
src/fleetmix/
├── api.py                   # Public Python API (fm.optimize())
├── app.py                   # CLI (Typer-based)
├── gui.py                   # Web interface (Streamlit)
├── core_types.py            # Dataclasses: Customer, Cluster, VehicleConfiguration
├── interfaces.py            # Protocols: Clusterer, RouteTimeEstimator, SolverAdapter
├── registry.py              # Plugin registration system
├── clustering/              # Cluster generation (k-means, k-medoids, GMM, etc.)
├── optimization/            # MILP fleet size & mix solver
├── post_optimization/       # Iterative merge improvement phase
├── merging/                 # Cluster merging utilities
├── preprocess/              # Split-stop preprocessing
├── utils/                   # Common utilities, route time estimation
├── config/                  # YAML configuration loading & validation
└── benchmarking/            # CVRP/MCVRP parsers, converters, datasets
```

### Key Types

```python
# Core domain (src/fleetmix/core_types.py)
Customer          # Customer with demands, location
Cluster           # Group of customers assigned to vehicle config
VehicleConfiguration  # Vehicle type with compartment assignment
FleetmixSolution  # Optimization result

# Contexts (passed to algorithms)
CapacitatedClusteringContext  # For clusterers
RouteTimeContext              # For route time estimators
```

### Plugin Architecture

FleetMix uses **Python Protocols (PEP 544)** for extension points. Any class matching the protocol signature works — no inheritance required.

```python
from fleetmix.registry import register_clusterer

@register_clusterer("my_method")
class MyClusterer:
    def fit(self, customers: pd.DataFrame, *, 
            context: CapacitatedClusteringContext, 
            n_clusters: int) -> list[int]:
        # Your implementation
        return labels
```

Available protocols:
- `Clusterer` — customer grouping algorithms
- `RouteTimeEstimator` — route duration estimation (BHH, TSP-based)
- `SolverAdapter` — MILP solver wrappers (Gurobi, CBC)

## Coding Conventions

### Type Hints (Required)

All code must pass `mypy` with strict settings:

```python
# ✅ Good
def process_clusters(
    clusters: list[Cluster],
    params: FleetmixParams,
) -> FleetmixSolution:
    ...

# ❌ Bad — missing types
def process_clusters(clusters, params):
    ...
```

### Style

- **Line length**: 88 characters (ruff)
- **Quotes**: Double quotes for strings
- **Imports**: Sorted by ruff (stdlib → third-party → local)
- **Docstrings**: Module-level required; function-level for public APIs
- **Dataclasses**: Preferred over dicts for structured data

```python
from dataclasses import dataclass

@dataclass
class VehicleSpec:
    capacity: int
    fixed_cost: float
    compartments: dict[str, bool]
```

### Naming

- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_single_underscore`
- Config IDs: String format (`"V1"`, `"config_123"`)

## Testing Requirements

**Run the full test suite before concluding any change is successful.**

```bash
# Full suite (required before any PR)
pytest

# Fast feedback during development
pytest tests/unit/ -x --tb=short

# Specific test file
pytest tests/unit/test_optimization_core.py -v

# With coverage
pytest --cov=src --cov-report=term-missing
```

### Test Organization

```
tests/
├── conftest.py          # Shared fixtures
├── unit/                # Fast, isolated tests
├── integration/         # Pipeline tests
├── component/           # Module integration
└── _assets/             # Test configs, data files
```

### Fixture Pattern

```python
@pytest.fixture
def simple_clusters_df():
    return pd.DataFrame({
        "Cluster_ID": ["C1", "C2"],
        "Customers": [["Cust1", "Cust2"], ["Cust3"]],
        "Config_ID": ["V1", "V1"],
        "Total_Demand": [{"Dry": 20}, {"Dry": 15}],
        "Route_Time": [2.5, 1.8],
    })
```

## Configuration

YAML-based configuration with Pydantic validation:

```yaml
# config.yaml
vehicles:
  SmallVan:
    capacity: 1800
    fixed_cost: 80
    compartments: 2
  LargeTruck:
    capacity: 4500
    fixed_cost: 200
    compartments: 3

goods: [Dry, Chilled, Frozen]

optimization:
  solver: gurobi  # or 'cbc'
  time_limit: 300

clustering:
  methods: [kmeans, kmedoids, gmm, agglomerative]
```

Load with: `params = load_fleetmix_params("config.yaml")`

## Common Tasks

### Adding a New Clustering Method

1. Create class implementing `Clusterer` protocol
2. Register with `@register_clusterer("name")`
3. Add tests in `tests/unit/`
4. Update config to use: `clustering.method: name`

### Modifying the Optimization Model

- Core MILP in `src/fleetmix/optimization/core.py`
- Cost calculation: `_calculate_cluster_cost()`
- Model building: `_create_model()`
- Solution extraction: `_extract_solution()`

### Debugging

```python
from fleetmix.utils.logging import FleetmixLogger, setup_logging, LogLevel

setup_logging(LogLevel.DEBUG)
FleetmixLogger.debug("Detailed message")
FleetmixLogger.detail("Progress message")  # For verbose mode
```

## Dependencies

**Package manager**: Use `uv` exclusively.

```bash
uv add package-name                    # Add dependency
uv sync --all-extras                   # Sync environment
uv run <command>                       # Run any command in the environment
```

If pip is absolutely needed: `uv run pip install --index-url https://pypi.org/simple/ package`

**Key libraries**:
- `numpy`, `pandas`, `scipy` — numerics
- `scikit-learn`, `kmedoids` — clustering
- `pulp`, `gurobipy` — optimization
- `typer`, `rich` — CLI
- `streamlit` — GUI
- `pytest`, `hypothesis` — testing

## Documentation

- `docs/ARCHITECTURE.md` — System design overview
- `docs/mapping.md` — Paper section ↔ code cross-reference
- `docs/specs/` — Module specifications with math formulations
- `docs/REPRODUCIBILITY.md` — Reproduce paper experiments

## Common Pitfalls

1. **Config ID types** — Always use strings for `config_id` comparisons
2. **Empty clusters** — Check `if not clusters:` before optimization
3. **Solver availability** — CBC is fallback when Gurobi unavailable
4. **Route time units** — Hours internally, minutes for service time
5. **Coordinate order** — `(latitude, longitude)` tuple order

## Git Workflow

- Branch naming: `feature/`, `fix/`, `refactor/`
- Commits: Clear, atomic, imperative mood
- Before any commit, run:
  ```bash
  uv run pytest
  uv run mypy src/fleetmix/
  uv run ruff check src/
  ```

