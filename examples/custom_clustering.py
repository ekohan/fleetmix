"""Custom clustering plugin example for FleetMix.

Demonstrates how to register a user-defined clustering algorithm through the
`fleetmix.registry` decorator, then run a small optimisation using it.

Run with:
    python examples/custom_clustering.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Ensure the plugin is visible (serial mode) *before* FleetMix import, see docs/parallelism.md.
os.environ.setdefault("FLEETMIX_N_JOBS", "1")

import dataclasses

import fleetmix as fm

# Plugin module import – executed for its side-effect of registering itself.
import fleetmix_example_plugins.round_robin  # noqa: F401
from fleetmix.config import load_fleetmix_params


def main():
    """Main execution function."""
    demand_file = Path("tests/_assets/smoke/mini_demand.csv")

    # Start with default config
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")
    # Update clustering method using dataclasses.replace for immutable params
    params_with_custom_clusterer = dataclasses.replace(
        params,
        algorithm=dataclasses.replace(
            params.algorithm, clustering_method="round_robin"
        ),
    )

    solution = fm.optimize(demand=demand_file, config=params_with_custom_clusterer)

    print("Selected clusters (ID -> customers):")
    for cluster in solution.selected_clusters:
        print(
            f"  Cluster {cluster.cluster_id} (Config {cluster.config_id}): {cluster.customers} | Total Demand: {cluster.total_demand}"
        )

    print(
        "Optimisation complete with custom clusterer – total cost:", solution.total_cost
    )


if __name__ == "__main__":
    main()
