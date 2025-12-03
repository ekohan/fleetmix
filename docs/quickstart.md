# Quick Start Guide

Welcome to **FleetMix**! This 5-minute tutorial walks you through running your first optimization.

---

## 1. Installation

```bash
git clone https://github.com/ekohan/fleetmix.git && cd fleetmix
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv venv fleetmix-env
source fleetmix-env/bin/activate
uv sync --all-extras
```

---

## 2. Run Your First Optimization

**Via Python API:**

```python
import fleetmix as fm

solution = fm.optimize(
    demand="tests/_assets/smoke/mini_demand.csv",
    config="src/fleetmix/config/default_config.yaml",
)

print(f"Total cost: ${solution.total_cost:,.2f}")
print(f"Vehicles used: {solution.total_vehicles}")
```

**Via CLI:**

```bash
fleetmix optimize \
  --demand tests/_assets/smoke/mini_demand.csv \
  --config src/fleetmix/config/default_config.yaml
```

Results are saved to `results/` as JSON (or XLSX with `--format xlsx`).

---

## Next Steps

* Explore the `examples/` folder for heterogeneous fleets and custom configurations.
* See `docs/USER_GUIDE.md` for detailed usage scenarios.
* Review `docs/specs/` for algorithm documentation.
