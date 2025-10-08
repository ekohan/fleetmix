# 🚀 Quick Start Guide

Welcome to **FleetMix**! This 5-minute tutorial will walk you through running your first optimisation on a tiny 10-customer dataset.

<!-- TODO: revisar esta, hacerla bien. -->

---

## 1. Installation

If you cloned the repository already, activate the environment created by `uv`:

```bash
source fleetmix-env/bin/activate  # macOS / Linux
```

Otherwise, install from source:

```bash
git clone https://github.com/ekohan/fleetmix.git && cd fleetmix
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv venv fleetmix-env
source fleetmix-env/bin/activate
uv sync --all-extras
```

---

## 2. Download the Toy Dataset

A 10-customer CSV is provided in the test assets. Copy it to a convenient location:

```bash
cp tests/_assets/smoke/mini_demand.csv ./mini_demand.csv
```

---

## 3. Optimise via the Python API (recommended)

```python
import fleetmix as fm

solution = fm.optimize(
    demand="mini_demand.csv",          # 10 customers
    config="src/fleetmix/config/default_config.yaml",
)

print(solution.summary())  # Cost, fleet composition, KPIs
```

---

## 4. …or via the CLI

```bash
fleetmix optimize \
  --demand mini_demand.csv \
  --config src/fleetmix/config/default_config.yaml
```

You'll find JSON/XLSX results in the newly-created `results/` folder.

---

## 5. Visualise the Routes (optional)

```bash
fleetmix gui --results results/latest
```
<!-- TODO: implement this gui feature -->
An interactive map will open in your browser, showing clusters, routes, and assigned vehicles.

---

## Next Steps

* Try the examples in the `examples/` folder (heterogeneous fleets, custom clustering, split-stops).
* Read the [Concepts](../docs/concepts/matheuristic.md) section to understand the algorithm.
* Dive into the [API Reference](../docs/api/) for advanced usage.

Happy optimising! 🚚 