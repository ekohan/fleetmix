"""Custom solver-adapter example for FleetMix.

Registers a *naive* solver adapter (thin wrapper around CBC with a relaxed
0.2 relative gap) and activates it via the ``runtime.solver`` parameter.

Run:
    uv run python examples/custom_solver_adapter.py
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

# 2. Import plugin for side-effect registration
import fleetmix_example_plugins.naive_solver  # noqa: F401
from fleetmix.config import load_fleetmix_params


def main() -> None:  # pragma: no cover – example script
    demand_file = Path("examples/bogota_demand.csv")

    # 1. Load default config
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")

    # 2. Update solver using dataclasses.replace for immutable params
    params = dataclasses.replace(
        params,
        runtime=dataclasses.replace(params.runtime, solver="naive"),
    )

    print("Running optimization with custom 'naive' solver adapter...")
    print(f"Dataset: {demand_file}")

    # 3. Run optimization
    solution = fm.optimize(demand=demand_file, config=params)

    # 4. Display results
    print("\n" + "=" * 40)
    print("       CUSTOM SOLVER RESULTS       ")
    print("=" * 40)

    print(f"\nTotal Cost: ${solution.total_cost:,.2f}")
    print(f"Solver Used: {solution.solver_name}")
    print(f"Vehicles Used: {len(solution.selected_clusters)}")

    print("\nCluster Assignments:")
    for i, cluster in enumerate(solution.selected_clusters):
        print(f"  Cluster {i + 1}:")
        print(f"    Vehicle: {cluster.vehicle_type} (Config {cluster.config_id})")
        print(f"    Customers: {', '.join(cluster.customers)}")
        total_demand_str = ", ".join(
            [f"{k}={v}" for k, v in cluster.total_demand.items() if v > 0]
        )
        print(f"    Total Demand: {total_demand_str}")

    print("\n" + "-" * 40)
    print(f"Detailed results saved to: {params.io.results_dir}")


if __name__ == "__main__":
    main()
