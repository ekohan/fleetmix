"""Heterogeneous-fleet demo for *FleetMix*.

Builds a **mixed fleet** (drones → refrigerated trucks) entirely in-memory – no YAML editing required.

Run it: $ uv run python examples/heterogeneous_fleet.py

"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict

import fleetmix as fm
from fleetmix import VehicleSpec
from fleetmix.config import FleetmixParams, load_fleetmix_params


def build_demo_parameters_heterogeneous_fleet() -> FleetmixParams:
    """Return a :class:`FleetmixParams` instance with a heterogeneous fleet."""

    # Start with default config
    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")

    # Define mixed fleet
    fleet: Dict[str, VehicleSpec] = {
        "Drone": VehicleSpec(
            capacity=5,
            fixed_cost=10,
            avg_speed=60,  # km/h – fast but tiny
            max_route_time=0.5,  # 30-minute battery
            allowed_goods=["Dry"],
        ),
        "E-Bike": VehicleSpec(
            capacity=50,
            fixed_cost=20,
            avg_speed=25,
            max_route_time=4,
            allowed_goods=["Dry", "Chilled"],
        ),
        "Van": VehicleSpec(
            capacity=500,
            fixed_cost=80,
            avg_speed=40,
            max_route_time=8,
        ),
        "Refrigerated_Truck": VehicleSpec(
            capacity=2000,
            fixed_cost=200,
            avg_speed=30,
            max_route_time=10,
            allowed_goods=["Chilled", "Frozen"],
        ),
    }

    # Update the fleet using dataclasses.replace for immutable params
    params = dataclasses.replace(
        params, problem=dataclasses.replace(params.problem, vehicles=fleet)
    )
    return params


def main() -> None:  # pragma: no cover – example script
    # Use the custom Bogotá demand file we just created
    demand_file = Path("examples/bogota_demand.csv")

    params = build_demo_parameters_heterogeneous_fleet()

    print(f"Depot location: {params.problem.depot}")

    solution = fm.optimize(demand=demand_file, config=params)

    print("\n=== Fleet Composition ===")
    for vehicle, count in solution.vehicles_used.items():
        print(f"{vehicle:>20}: {count}")

    print("\n=== Cost Breakdown ===")
    print(f"Total cost: ${solution.total_cost:,.2f}")

    print(
        f"\nOptimisation complete – review the generated JSON and HTML results in {params.io.results_dir}"
    )


if __name__ == "__main__":
    main()
