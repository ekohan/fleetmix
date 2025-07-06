"""Custom route-time estimator example for FleetMix.

Shows how to plug in a user-defined *RouteTimeEstimator*.  The estimator lives
in ``src/fleetmix_example_plugins/straight_line.py`` and assumes the vehicle
travels in straight lines at its average speed.

Run:
    python examples/custom_route_time.py

The demo forces serial execution so the plugin is visible without installing it
as a package.  See docs/parallelism.md for details.
"""

from __future__ import annotations

import os
from pathlib import Path

# Ensure the plugin is visible (serial mode) *before* FleetMix import, see docs/parallelism.md.
os.environ.setdefault("FLEETMIX_N_JOBS", "1")

# Register the straight-line estimator (import for side-effect)
import fleetmix as fm
import fleetmix_example_plugins.straight_line  # noqa: F401
from fleetmix.config import load_fleetmix_params
import dataclasses


def main() -> None:  # pragma: no cover – example script
    demand_file = Path("tests/_assets/smoke/mini_demand.csv")

    # Start with default config
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")
    # Update route time estimation using dataclasses.replace for immutable params
    params_with_custom_route_time_estimator = dataclasses.replace(
        params,
        algorithm=dataclasses.replace(
            params.algorithm, route_time_estimation="straight_line"
        ),
    )

    solution = fm.optimize(
        demand=demand_file, config=params_with_custom_route_time_estimator
    )

    print("\nSolved using *straight_line* estimator – total cost:", solution.total_cost)


if __name__ == "__main__":
    main()
