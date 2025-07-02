"""Custom solver-adapter example for FleetMix.

Registers a *naive* solver adapter (thin wrapper around CBC with a relaxed
0.2 relative gap) and activates it via the ``FSM_SOLVER`` environment
variable.

Run:
    python examples/custom_solver_adapter.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Use our relaxed CBC adapter
os.environ["FSM_SOLVER"] = "cbc"

# Import plugin for side-effect registration
import fleetmix as fm
import fleetmix_example_plugins.naive_solver  # noqa: F401


def main() -> None:  # pragma: no cover – example script
    demand_file = Path("tests/_assets/smoke/mini_demand.csv")

    params = fm.Parameters.from_yaml("src/fleetmix/config/default_config.yaml")

    solution = fm.optimize(demand=demand_file, config=params)

    print("\nSolved using *naive* CBC adapter – total cost:", solution.total_cost)
    print("Solver used:", solution.solver_name)


if __name__ == "__main__":
    main()
