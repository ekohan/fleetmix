"""Custom clustering plugin example for FleetMix.

Demonstrates how to register a user-defined clustering algorithm through the
`fleetmix.registry` decorator, then run a small optimisation using it.

Run with:
    python examples/custom_clustering.py
"""
# ---------------------------------------------------------------------------
# Example script needs to run serially so the dynamically-registered clusterer
# is visible.  Setting the env-var **before** importing FleetMix ensures this.
# Details: docs/parallelism.md
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("FLEETMIX_N_JOBS", "1")

# Plugin module import – executed for its side-effect of registering itself.
import fleetmix as fm
import fleetmix_example_plugins.round_robin  # noqa: F401
from fleetmix.config import load_fleetmix_params
import dataclasses


def main():
    """Main execution function."""
    # ---------------------------------------------------------------------------
    # 1.  Custom round-robin clusterer (demo)
    # ---------------------------------------------------------------------------
    # The plugin is imported above; nothing to define here

    # ---------------------------------------------------------------------------
    # 2.  Prepare demand file
    # ---------------------------------------------------------------------------
    demand_file = Path("tests/_assets/smoke/mini_demand.csv")
    
    # ---------------------------------------------------------------------------
    # 3.  Run optimisation with custom clusterer
    # ---------------------------------------------------------------------------

    # Start with default config
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")
    # Update clustering method using dataclasses.replace for immutable params
    params = dataclasses.replace(
        params,
        algorithm=dataclasses.replace(
            params.algorithm,
            clustering_method="round_robin"
        )
    )

    solution = fm.optimize(demand=demand_file, config=params)

    # ---------------------------------------------------------------------------
    # 4.  Display selected clusters
    # ---------------------------------------------------------------------------
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
