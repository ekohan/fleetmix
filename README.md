# fleetmix — Fleet Size & Mix Optimizer for Multi‑Compartment Fleets

*Written for transparent research, hardened for production use.*

Fast, reproducible tooling for **multi‑compartment vehicle fleet design** in urban food distribution. This repository supports our forthcoming paper *Designing Multi‑Compartment Vehicle Fleets for Last‑Mile Food Distribution Systems* and doubles as a production‑grade library for industry users.

---

## Why fleetmix?

* **Scales** — >1,000 customers solved in seconds via a *cluster‑first → MILP‑second* matheuristic.
* **Extensible** — pluggable clustering engines, route‑time estimators, solver back‑ends.
* **Reproducible** — entire paper can be re-run with a single script.
* **User-friendly** — clean CLI, idiomatic Python API, and a lightweight web GUI.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Command‑Line Usage](#command-line-usage)
5. [Python API](#python-api)
6. [Benchmarking Suite](#benchmarking-suite)
7. [Repository Layout](#repository-layout)
8. [Paper ↔ Code Map](#paper-↔-code-map)
9. [Contributing](#contributing)
10. [Citation](#citation)
11. [License](#license)

---

## Installation

### From PyPI (coming soon)
```bash
pip install fleetmix
```

### From Source (Development)
```bash
# Clone and setup environment
git clone https://github.com/ekohan/fleetmix.git && cd fleetmix
./init.sh

# Install the package in editable mode
pip install -e .
```

---

## Quick Start

### Command Line Interface

```bash
# Run optimization on customer demand data
fleetmix optimize --demand customers.csv --config fleet.yaml

# Run the full MCVRP benchmark suite
fleetmix benchmark mcvrp

# Convert VRP instance to FSM format
fleetmix convert --type cvrp --instance X-n101-k25 --benchmark-type split

# Check version
fleetmix version
```

### Python API

```python
import fleetmix

# Run optimization
solution = fleetmix.optimize(
    demand="customers.csv",
    config="fleet_config.yaml"
)

print(f"Total cost: ${solution['total_cost']:,.2f}")
print(f"Vehicles used: {len(solution['vehicles_used'])}")
```

### Web Interface

Fleetmix includes a web-based GUI for interactive optimization:

```bash
# Launch web interface
fleetmix gui

# Or specify a custom port
fleetmix gui --port 8080
```

The GUI provides:
- Drag-and-drop CSV upload
- Interactive parameter configuration
- Real-time optimization progress
- Visual results with maps
- Excel/JSON export

---

## Architecture Overview

```mermaid
graph LR
    A[Read Demand] --> B[Generate feasible clusters]
    B --> C[MILP fleet‑selection]
    C --> D[Merge improvement phase]
    D --> E["Results (JSON | XLSX | HTML)"]
```

*Full algorithmic details are in §4 of the paper.*

---

## Command‑Line Usage

### Main Commands

#### `fleetmix optimize`
Run fleet optimization on customer demand data.

```bash
fleetmix optimize \
  --demand customers.csv \      # Customer demand CSV file
  --config fleet.yaml \         # Configuration YAML file
  --output results/ \           # Output directory
  --format excel \              # Output format (excel or json)
  --verbose                     # Enable verbose output
```

#### `fleetmix benchmark`
Run the **full benchmark suites** shipped with Fleetmix (batch-mode over all instances). Use `fleetmix convert` for one-off runs on individual instances.

```bash
# Run all MCVRP instances
fleetmix benchmark mcvrp

# Run all CVRP instances  
fleetmix benchmark cvrp
```

#### `fleetmix convert`
Convert a **single** CVRP / MCVRP instance—either from the built-in datasets *or your own .vrp/.dat file*—into FSM format, run optimisation, and export results. Ideal for ad-hoc experimentation; use `fleetmix benchmark` for full-suite runs.

```bash
# Convert MCVRP instance
fleetmix convert --type mcvrp --instance 10_3_3_3_\(01\)

# Convert CVRP instance with specific benchmark type
fleetmix convert \
  --type cvrp \
  --instance X-n101-k25 \
  --benchmark-type split \
  --num-goods 3
```

### Legacy Scripts (Deprecated)

The following direct script executions still work but will show deprecation warnings:

```bash
# Old way (deprecated)
python -m fleetmix.cli.main --demand-file data/customers.csv

# New way (recommended)
fleetmix optimize --demand data/customers.csv
```

---

## Python API

### Basic Usage

```python
import fleetmix
import pandas as pd

# Option 1: Using file paths
solution = fleetmix.optimize(
    demand="customers.csv",
    config="config.yaml",
    output_dir="results",
    format="excel"
)

# Option 2: Using DataFrame directly
customers_df = pd.DataFrame({
    'Customer_ID': [1, 2, 3],
    'Latitude': [40.7128, 40.7580, 40.7614],
    'Longitude': [-74.0060, -73.9855, -73.9776],
    'Dry_Demand': [100, 150, 200],
    'Chilled_Demand': [50, 75, 100],
    'Frozen_Demand': [25, 50, 0]
})

solution = fleetmix.optimize(
    demand=customers_df,
    config="config.yaml"
)

# Access solution details
print(f"Total cost: ${solution['total_cost']:,.2f}")
print(f"Fixed cost: ${solution['total_fixed_cost']:,.2f}")
print(f"Variable cost: ${solution['total_variable_cost']:,.2f}")
print(f"Vehicles used: {solution['vehicles_used']}")
```

### Error Handling

The API provides helpful error messages for common issues:

```python
try:
    solution = fleetmix.optimize(demand="customers.csv", config="config.yaml")
except FileNotFoundError as e:
    print(f"File error: {e}")
except ValueError as e:
    print(f"Configuration or optimization error: {e}")
```

---

## Benchmarking Suite

Located under `src/fleetmix/benchmarking/`

* **Converters** – turn `.vrp` / `.dat` instances into Fleet‑Size‑and‑Mix demand tables.
* **Parsers & Models** – light dataclasses for CVRP / MCVRP metadata.
* **Solvers** – PyVRP wrapper providing single‑ and multi‑compartment baselines.
* **Scripts** – batch runners producing JSON/XLSX artifacts in `results/`.

Upper‑ and lower‑bound reference solutions are generated automatically for sanity checks.

---

## Repository Layout

```
src/fleetmix/
  api.py                 # Python API facade
  app.py                 # CLI application (Typer)
  clustering/            # capacity & time‑feasible cluster generation
  optimization/          # MILP core pulp / gurobi backend
  post_optimization/     # merge‑phase heuristic
  benchmarking/          # datasets • converters • baseline solvers
  cli/                   # legacy entry points (deprecated)
  utils/                 # I/O, logging, route‑time estimation, etc.
  config/                # default_config.yaml + Parameters dataclass
  pipeline/              # thin orchestration wrappers
tests/                   # >150 unit / integration tests
docs/                    # code↔paper map • design notes
data/                    # sample data files
tools/                   # utility scripts
```

---

## Paper ↔ Code Map

See `docs/mapping.md` for a line‑by‑line crosswalk between paper sections and implementation.

---

## Contributing

1. Fork → feature branch → PR against **main**.
2. All tests (`pytest -q --cov=src`) **must** stay green.
3. Follow *PEP‑8*, add type hints, and keep public APIs doc‑commented.

Bug reports and ideas via Issues are welcome.

---

## Citation

If you use this software, please cite the companion article (pre‑print DOI forthcoming):

```latex
@article{Kohan2025FleetMix,
  author  = {Eric Kohan},
  title   = {Designing Multi‑Compartment Vehicle Fleets for Last‑Mile Food Distribution Systems},
  journal = {To appear},
  year    = {2025}
}
```

---

## License

`MIT` — free for academic & commercial use. See `LICENSE` for details.
