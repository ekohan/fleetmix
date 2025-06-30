# 🚚 **fleetmix** — *Fleet Size & Mix Optimizer for Multi‑Compartment Fleets*

[![PyPI](https://img.shields.io/pypi/v/fleetmix.svg?label=PyPI)](https://pypi.org/project/fleetmix/)
[![CI](https://img.shields.io/github/actions/workflow/status/ekohan/fleetmix/ci.yml?label=CI)](https://github.com/ekohan/fleetmix/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/ekohan/fleetmix?label=coverage)](https://codecov.io/gh/ekohan/fleetmix)

*Written for transparent research, hardened for production use.*

Fast, reproducible tooling for **multi‑compartment vehicle fleet design** in urban food distribution.
This repository supports our forthcoming paper *Designing Multi‑Compartment Vehicle Fleets for Last‑Mile Food Distribution Systems* and doubles as a production‑grade library for industry users.

---

<!-- GIF Demo -->

<p align="center">
  <img src="docs/images/fleetmix_demo.png" alt="Fleetmix demo animation" width="80%"/>
  <br><em>(interactive demo – coming soon)</em>
</p>

---

## ✨ Why fleetmix?

* ⚡ **Scales** — >1,000 customers solved in seconds via a *cluster‑first → MILP‑second* matheuristic.
* 🧩 **Extensible** — pluggable clustering engines, route‑time estimators, and solver back‑ends.
* 🔄 **Reproducible** — every experiment in the journal article re‑runs with one script.
* 🖥️ **User‑friendly** — clean CLI, idiomatic Python API, and a lightweight web GUI.

---

## 🗺️ Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Command‑Line Usage](#command-line-usage)
5. [Python API](#python-api)
6. [Configuration](#configuration)
7. [Composability & Extensibility](#composability--extensibility)
8. [Benchmarking Suite](#benchmarking-suite)
9. [Repository Layout](#repository-layout)
10. [Paper ↔ Code Map](#paper-↔-code-map)
11. [Contributing](#contributing)
12. [Citation](#citation)
13. [License](#license)

---

## ⚙️ Installation

### From PyPI *(coming soon)*

```bash
pip install fleetmix
```

### From Source *(development)*

```bash
# Clone and set up environment
git clone https://github.com/ekohan/fleetmix.git && cd fleetmix
./init.sh

# Install in editable mode
pip install -e .
```

---

## 🚀 Quick Start

### Command‑Line Interface

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
import fleetmix as fm

customers_df = ...  # build a DataFrame
solution = fm.optimize(demand=customers_df, config="config.yaml")
```

Retrieve metrics via `solution[...]` keys (see docstring for full schema).

### Web Interface

```bash
# Launch web interface
fleetmix gui

# Or specify a custom port
fleetmix gui --port 8080
```

The GUI provides:

* 📥 Drag‑and‑drop CSV upload
* 🎛️ Interactive parameter tweaking (including allowed goods per vehicle and split-stop configuration)
* 🔎 Real‑time optimization progress
* 🗺️ Map‑based visual results
* 📊 Excel/JSON export

---

## 🏗️ Architecture Overview

```mermaid
graph LR
    A[Read Demand] --> B[Generate feasible clusters]
    B --> C[MILP fleet‑selection]
    C --> D[Merge improvement phase]
    D --> E["Results (JSON | XLSX | HTML)"]
```

*Full algorithmic details are in §4 of the paper.*

---

## 🔧 Command‑Line Usage

### `fleetmix optimize`

Run fleet optimization on customer demand data.

```bash
fleetmix optimize \
  --demand customers.csv \
  --config fleet.yaml \
  --output results/ \
  --format excel \
  --verbose
```

### `fleetmix benchmark`

Run the **full benchmark suites** shipped with Fleetmix (batch mode over all instances).

```bash
fleetmix benchmark mcvrp   # All MCVRP instances
fleetmix benchmark cvrp    # All CVRP instances
fleetmix benchmark case    # All case study instances
```

You can also specify a config file to use custom parameters:

```bash
fleetmix benchmark mcvrp --config custom_config.yaml
fleetmix benchmark case --config baseline_config.yaml --format excel
```

### `fleetmix convert`

Convert a **single** CVRP / MCVRP instance into FSM format, run optimisation, and export results.

```bash
fleetmix convert --type mcvrp --instance 10_3_3_3_\(01\)
```

> *Legacy direct‑script calls still work but show deprecation warnings.*

---

## 🐍 Python API

```python
import fleetmix as fm

customers_df = ...  # build a DataFrame
solution = fm.optimize(demand=customers_df, config="config.yaml")
```

Retrieve metrics via `solution[...]` keys (see docstring for full schema).

---

## ⚙️ Configuration

FleetMix uses YAML configuration files to define fleet composition, optimization parameters, and operational constraints.

### Vehicle-Specific Goods Capability

Vehicles can be configured to carry only specific subsets of goods, enabling realistic modeling of specialized fleets:

```yaml
vehicles:
  # Dry goods only truck
  DryTruck:
    capacity: 2700
    fixed_cost: 100
    avg_speed: 30
    service_time: 25
    max_route_time: 10
    allowed_goods: ["Dry"]  # Can only carry dry goods
  
  # Refrigerated truck for cold chain
  RefrigeratedTruck:
    capacity: 3300
    fixed_cost: 175
    avg_speed: 30
    service_time: 25
    max_route_time: 10
    allowed_goods: ["Chilled", "Frozen"]  # No dry goods capability
  
  # Multi-temperature truck (no allowed_goods = can carry all goods)
  MultiTempTruck:
    capacity: 4500
    fixed_cost: 225
    avg_speed: 30
    service_time: 25
    max_route_time: 10
    # No allowed_goods specified - can carry all goods

goods:
  - Dry
  - Chilled
  - Frozen
```

**Key features:**
- If `allowed_goods` is not specified, the vehicle can carry all goods (backward compatible)
- If specified, must be a non-empty subset of the global `goods` list
- Automatically generates only feasible compartment combinations
- Optimization respects these constraints when assigning customers to vehicles

See `src/fleetmix/config/example_allowed_goods_config.yaml` for a complete example.

---

## 🧩 Composability & Extensibility

FleetMix uses a **Protocol-based plugin architecture** that makes it easy to add custom implementations for core components.

### Adding a Custom Clustering Algorithm

```python
from fleetmix.registry import register_clusterer
from fleetmix.interfaces import Clusterer
import pandas as pd
from typing import List

@register_clusterer("my_custom_clustering")
class MyCustomClusterer:
    """Custom clustering implementation."""
    
    def fit(self, customers: pd.DataFrame, *, context, n_clusters: int) -> List[int]:
        """Implement your clustering logic here."""
        # Your custom clustering algorithm
        # Must return a list of cluster labels (integers)
        labels = your_clustering_logic(customers, n_clusters)
        return labels

# Now use it in your config.yaml:
# clustering:
#   method: my_custom_clustering
```

The plugin system supports:
- **Clustering algorithms**: K-means, K-medoids, Agglomerative, Gaussian Mixture, or your own
- **Route time estimators**: Legacy, BHH, TSP-based, or custom (e.g., with traffic data)
- **Solvers**: Gurobi, CBC, or any PuLP-compatible solver

---

## 📊 Benchmarking Suite

Located under `src/fleetmix/benchmarking/`.

* **Converters** – `.vrp` / `.dat` → FSM tables
* **Parsers & Models** – light dataclasses for CVRP / MCVRP metadata
* **Solvers** – PyVRP wrapper providing single‑ & multi‑compartment baselines
* **Case Studies** – real-world demand patterns from food distribution
* **Scripts** – batch runners producing JSON/XLSX artifacts in `results/`

The suite includes three benchmark types:
- **CVRP**: Classic vehicle routing instances
- **MCVRP**: Multi-compartment vehicle routing instances  
- **Case**: Real-world food distribution demand patterns

Upper‑ and lower‑bound reference solutions are generated automatically for sanity checks.

---

## 🗂️ Repository Layout

```
src/fleetmix/
  api.py                # Python API facade
  app.py                # CLI (Typer)
  clustering/           # capacity & time‑feasible cluster generation
  optimization/         # MILP core (PuLP/Gurobi)
  post_optimization/    # merge‑phase heuristic
  benchmarking/         # datasets • converters • baselines
  gui.py                # lightweight web GUI
  utils/                # I/O, logging, etc.
docs/                   # code↔paper map • design notes
```

---

## 📝 Paper ↔ Code Map

See `docs/mapping.md` for a line‑by‑line crosswalk between paper sections and implementation.

---

## 🤝 Contributing

1. Fork → feature branch → PR against **main**.
2. `pytest -q --cov=src` **must** stay green.
3. Follow *PEP‑8*, add type hints, and keep public APIs doc‑commented.

Bug reports and ideas via **Issues** are welcome.

---

## 📚 Citation

```latex
@article{Kohan2025FleetMix,
  author  = {Eric Kohan},
  title   = {Designing Multi‑Compartment Vehicle Fleets for Last‑Mile Food Distribution Systems},
  journal = {To appear},
  year    = {2025}
}
```

---

## 🪪 License

`MIT` — free for academic & commercial use. See [`LICENSE`](LICENSE) for details.
