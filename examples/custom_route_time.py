"""Custom route-time estimator example for FleetMix.

Shows how to plug in a user-defined *RouteTimeEstimator*.  The estimator lives
in ``src/fleetmix_example_plugins/straight_line.py`` and assumes the vehicle
travels in straight lines at its average speed.

Run:
    uv run python examples/custom_route_time.py
"""

from __future__ import annotations

import os
from pathlib import Path

# 1. Configure environment *before* importing fleetmix.
# We force serial execution (N_JOBS=1) because our custom plugin is registered
# at runtime in this script's process. Worker processes in parallel mode would
# not see this registration unless the plugin was a properly installed package.
os.environ.setdefault("FLEETMIX_N_JOBS", "1")

# Suppress info logs for a cleaner demo output
os.environ["FLEETMIX_EFFECTIVE_LOG_LEVEL"] = "QUIET"

import dataclasses

import fleetmix as fm

# 2. Register the straight-line estimator (import for side-effect)
import fleetmix_example_plugins.straight_line  # noqa: F401
from fleetmix.config import load_fleetmix_params


def main() -> None:  # pragma: no cover – example script
    demand_file = Path("examples/bogota_demand.csv")

    # 1. Load default config
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")

    # 2. Update route time estimation using dataclasses.replace for immutable params
    params = dataclasses.replace(
        params,
        algorithm=dataclasses.replace(
            params.algorithm, route_time_estimation="straight_line"
        ),
    )

    print("Running optimization with custom 'straight_line' estimator...")
    print(f"Dataset: {demand_file}")

    # 3. Run optimization
    solution = fm.optimize(demand=demand_file, config=params)

    # 4. Display results
    print("\n" + "=" * 40)
    print("       CUSTOM ROUTE TIME RESULTS       ")
    print("=" * 40)

    print(f"\nTotal Cost: ${solution.total_cost:,.2f}")
    print(f"Vehicles Used: {len(solution.selected_clusters)}")

    print("\nCluster Assignments:")
    for i, cluster in enumerate(solution.selected_clusters):
        print(f"  Cluster {i + 1}:")
        print(f"    Vehicle: {cluster.vehicle_type} (Config {cluster.config_id})")
        print(f"    Customers: {', '.join(cluster.customers)}")
        print(f"    Route Time: {cluster.route_time:.2f} hours (Straight Line)")
        total_demand_str = ", ".join(
            [f"{k}={v}" for k, v in cluster.total_demand.items() if v > 0]
        )
        print(f"    Total Demand: {total_demand_str}")

    print("\n" + "-" * 40)
    print(f"Detailed results saved to: {params.io.results_dir}")


if __name__ == "__main__":
    main()
