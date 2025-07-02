"""Heterogeneous-fleet demo for *FleetMix*.

Highlights
==========
1. Builds a **mixed fleet** (drones → refrigerated trucks) entirely in-memory – no YAML editing required.
2. Uses the bundled ``mini_demand.csv`` test file for convenience – edit the path as you like.
3. Stays <50 lines while still showcasing FleetMix's high-level API.

Run it:
    $ python examples/heterogeneous_fleet.py

Tip: afterwards launch ``fleetmix gui`` to visualise routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import fleetmix as fm
from fleetmix.core_types import VehicleSpec


def build_demo_parameters() -> fm.Parameters:
    """Return a :class:`fleetmix.Parameters` instance with a heterogeneous fleet."""

    params = fm.Parameters.from_yaml("src/fleetmix/config/default_config.yaml")

    # Define mixed fleet    – deliberately diverse to showcase capability
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

    params.vehicles = fleet
    return params


def main() -> None:  # pragma: no cover – example script
    demand_file = Path("tests/_assets/smoke/mini_demand.csv")

    params = build_demo_parameters()

    solution = fm.optimize(demand=demand_file, config=params)

    # --- Pretty print results -------------------------------------------------
    print("\n=== Fleet Composition ===")
    for vehicle, count in solution.vehicles_used.items():
        print(f"{vehicle:>20}: {count}")

    print("\n=== Cost Breakdown ===")
    print(f"Total cost: ${solution.total_cost:,.2f}")

    print("\nOptimisation complete – run `fleetmix gui` to visualise routes!")


if __name__ == "__main__":
    main()
