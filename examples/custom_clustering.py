"""Custom clustering plugin example for FleetMix.

Demonstrates how to register a user-defined clustering algorithm through the
`fleetmix.registry` decorator, then run a small optimisation using it.

Run with:
    uv run python examples/custom_clustering.py
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

# 2. Import the plugin module for its side-effect: registering 'round_robin'.
import fleetmix_example_plugins.round_robin  # noqa: F401
from fleetmix.config import load_fleetmix_params


def main() -> None:
    # Use the standard example dataset
    demand_file = Path("examples/bogota_demand.csv")

    # 1. Load default configuration
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")

    # 2. Switch to our custom 'round_robin' clustering method
    # We use dataclasses.replace because FleetmixParams are immutable
    params = dataclasses.replace(
        params,
        algorithm=dataclasses.replace(
            params.algorithm, clustering_method="round_robin"
        ),
    )

    print("Running optimization with custom 'round_robin' clusterer...")
    print(f"Dataset: {demand_file}")

    # 3. Run optimization
    solution = fm.optimize(demand=demand_file, config=params)

    # 4. Display results
    print("\n" + "=" * 40)
    print("       CUSTOM CLUSTERING RESULTS       ")
    print("=" * 40)

    print(f"\nTotal Cost: ${solution.total_cost:,.2f}")
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
