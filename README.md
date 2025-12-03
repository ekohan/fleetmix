# 🚚 **fleetmix** — *Designing Multi‑Compartment Last‑Mile Vehicle Fleets: An Open‑Source Matheuristic*

[![PyPI](https://img.shields.io/pypi/v/fleetmix.svg?label=PyPI)](https://pypi.org/project/fleetmix/)
[![CI](https://img.shields.io/github/actions/workflow/status/ekohan/fleetmix/ci.yml?label=CI)](https://github.com/ekohan/fleetmix/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/ekohan/fleetmix?label=coverage)](https://codecov.io/gh/ekohan/fleetmix)

*Written for transparent research, hardened for production use.*

Fast, reproducible tooling for **multi‑compartment vehicle fleet design** in urban food distribution.
This repository supports our paper *Designing Multi‑Compartment Last‑Mile Vehicle Fleets: An Open‑Source Matheuristic* and provides a production-ready library for practitioners.

---

## ✨ Why FleetMix?

* ⚡ **Scales** — >1,000 customers solved in seconds via a *cluster‑first → MILP‑second* matheuristic
* 🧩 **Extensible** — pluggable clustering engines, route‑time estimators, and solver back‑ends  
* 🔄 **Reproducible** — every experiment in the journal article re‑runs with one script
* 🖥️ **User‑friendly** — clean CLI, idiomatic Python API, and a lightweight web GUI
* 📐 **Spec-driven** — comprehensive module specifications connecting paper to code

---

## 🗺️ Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Running Examples](#-running-examples)
4. [📐 Specifications & Documentation](#-specifications--documentation) ⭐
5. [🔬 Reproducing Paper Experiments](#-reproducing-paper-experiments) ⭐
6. [Configuration](#configuration)
7. [Composability & Extensibility](#composability--extensibility)
8. [Citation](#citation)
9. [License](#license)

---

## ⚙️ Installation

### From PyPI

```bash
uv pip install fleetmix
```

### From Source (development)

```bash
# Clone and set up environment
git clone https://github.com/ekohan/fleetmix.git && cd fleetmix

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Create virtual environment and install all extras
uv venv fleetmix-env
source fleetmix-env/bin/activate
uv sync --all-extras --active

# Test
fleetmix --help
```

---

## 🚀 Quick Start

### Command‑Line Interface

```bash
# Run optimization on customer demand data
fleetmix optimize --demand customers.csv --config fleet.yaml

# Reproduce paper results
fleetmix reproduce-paper

# List all available commands
fleetmix --help
```

### Python API

```python
import fleetmix as fm

customers_df = ...  # build a DataFrame
solution = fm.optimize(demand=customers_df, config="config.yaml")
```

### Web Interface

```bash
fleetmix gui
```

The GUI provides drag‑and‑drop CSV upload, interactive parameter tweaking, real‑time optimization progress, and map‑based visual results.

---

## 💡 Running Examples

The `examples/` directory contains standalone scripts demonstrating various features. We recommend running them using `uv run` to ensure they use the correct environment:

```bash
# Heterogeneous fleet demo
uv run python examples/heterogeneous_fleet.py

# Custom clustering plugin (Round Robin)
uv run python examples/custom_clustering.py

# Custom route-time estimator (Straight Line)
uv run python examples/custom_route_time.py

# Custom solver adapter (Naive CBC)
uv run python examples/custom_solver_adapter.py
```

*Note: If you have already activated your virtual environment (via `source fleetmix-env/bin/activate`), you can simply use `python examples/...`.*

---

## 📐 Specifications & Documentation

FleetMix is a **spec-driven codebase**: each module has detailed specifications connecting the paper's mathematical formulations to the implementation.

### For Researchers

| Document | Purpose |
|----------|---------|
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System overview and module interactions |
| **[docs/mapping.md](docs/mapping.md)** | Complete paper section ↔ code cross-reference |
| **[docs/specs/](docs/specs/)** | Detailed module specifications with math formulations |
| **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)** | Reproduce all paper experiments |

### For Practitioners

| Document | Purpose |
|----------|---------|
| **[docs/quickstart.md](docs/quickstart.md)** | Get started in 5 minutes |
| **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** | Complete end-to-end workflow guide |
| **[docs/specs/configuration.md](docs/specs/configuration.md)** | All parameters explained |
| **[docs/specs/protocols.md](docs/specs/protocols.md)** | Plugin development guide |

---

## 🔬 Reproducing Paper Experiments

All experiments from the paper can be reproduced using the `fleetmix reproduce-paper` command suite:

```bash
# View available experiments
fleetmix reproduce-paper --help

# Experiment 1: MCVRP benchmark instances (Section: Effectiveness of the Matheuristic Approach)
fleetmix reproduce-paper mcvrp-instances

# Experiment 2: Sensitivity analysis (Section: Benefits of using MCVs)
fleetmix reproduce-paper sensitivity-analysis

# Experiment 3: Fleet composition analysis (Section: Impact of Cost Structure on Fleet Composition)
fleetmix reproduce-paper fleet-composition
```

**📖 Full documentation:** [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)

---

## ⚙️ Configuration

FleetMix uses YAML configuration files to define fleet composition, optimization parameters, and operational constraints:

```yaml
vehicles:
  SmallVan:
    capacity: 1800
    fixed_cost: 80
    compartments: 2
  LargeTruck:
    capacity: 4500
    fixed_cost: 200
    compartments: 3
    allowed_goods: ["Dry", "Chilled"]  # Optional: restrict goods types

goods: [Dry, Chilled, Frozen]
```

**📖 Full reference:** [docs/specs/configuration.md](docs/specs/configuration.md) · [default_config.yaml](src/fleetmix/config/default_config.yaml)

---

## 🧩 Composability & Extensibility

FleetMix uses a **Protocol-based plugin architecture** (see `src/fleetmix/interfaces.py`) that makes it easy to add custom implementations for core components.

The plugin system supports:
- **Clustering algorithms**: K-means, K-medoids, Agglomerative, Gaussian Mixture, or your own
- **Route time estimators**: BHH, TSP-based, or custom (e.g., with traffic data)
- **Solvers**: Gurobi, CBC, or any PuLP-compatible solver

**See also:**
- **[docs/specs/protocols.md](docs/specs/protocols.md)** for detailed interface definitions
- **[`src/fleetmix_example_plugins/`](src/fleetmix_example_plugins/)** for working examples (`round_robin.py`, `straight_line.py`, `naive_solver.py`)


---

## 📚 Citation

If using FleetMix in your research:

```bibtex
@article{Kohan2025FleetMix,
  author  = {Eric Kohan and Fabricio Torres and Victor Silva-Febre and J.C. Pina-Pardo},
  title   = {Designing Multi-Compartment Last-Mile Vehicle Fleets: An Open-Source Matheuristic},
  journal = {TBD},
  year    = {2025},
  note    = {To be submitted}
}
```

**Repository**: [github.com/ekohan/fleetmix](https://github.com/ekohan/fleetmix)  

---

## 🪪 License

`MIT` — free for academic & commercial use. See [`LICENSE`](LICENSE) for details.
