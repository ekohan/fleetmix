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

import fleetmix as fm

# Import plugin for side-effect registration
import fleetmix_example_plugins.naive_solver  # noqa: F401

from fleetmix.config import load_fleetmix_params
import dataclasses


def main() -> None:  # pragma: no cover – example script
    demand_file = Path("tests/_assets/smoke/mini_demand.csv")

    # Start with default config
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")
    # Update solver using dataclasses.replace for immutable params
    params_with_custom_solver = dataclasses.replace(
        params,
        runtime=dataclasses.replace(params.runtime, solver="naive"),
    )

    solution = fm.optimize(demand=demand_file, config=params_with_custom_solver)

    print("\nSolved using *naive* CBC adapter – total cost:", solution.total_cost)
    print("Solver used:", solution.solver_name)


if __name__ == "__main__":
    main()
