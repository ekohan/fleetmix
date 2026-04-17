#!/usr/bin/env python3
"""
Sanity check: TSP tour of ALL customers per Henke 2015 n=10 instance.

Justifies the 10h max-route-duration choice in the heterogeneous experiment
by showing that a single vehicle serving every customer would exceed 10h
(i.e., the constraint is non-trivial and binding).

Writes: results/henke_heterogeneous/tsp_of_all.csv

Usage:
    uv run python scripts/henke_tsp_of_all.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "src/fleetmix/benchmarking/datasets/mcvrp"
OUT_CSV = PROJECT_ROOT / "results/henke_heterogeneous/tsp_of_all.csv"

AVG_SPEED = 30.0       # km/h — Bogotá case study
SERVICE_TIME = 25.0    # minutes — Bogotá case study
MAX_ROUTE_TIME = 10.0  # hours — the threshold we want to check


def main() -> None:
    from fleetmix.benchmarking.converters.vrp import VRPType, convert_vrp_to_fsm
    from fleetmix.core_types import Customer
    from fleetmix.utils.route_time import estimate_route_time

    instances: list[str] = []
    for s in (1, 2, 3):
        for i in range(1, 51):
            instances.append(f"2015_10_3_3_{s}_({i:02d})")

    rows: list[dict[str, str]] = []
    exceed_count = 0
    tour_hours: list[float] = []

    print(f"Computing TSP-of-all on {len(instances)} instances...")
    for idx, inst in enumerate(instances, 1):
        dat_path = DATASETS_DIR / f"{inst}.dat"
        customers_df, spec = convert_vrp_to_fsm(
            VRPType.MCVRP, instance_path=dat_path
        )
        customers = Customer.from_dataframe(customers_df)
        depot = {
            "latitude": spec.depot.latitude,
            "longitude": spec.depot.longitude,
        }

        tour_time_hours, _ = estimate_route_time(
            cluster_customers=customers_df,
            depot=depot,
            service_time=SERVICE_TIME,
            avg_speed=AVG_SPEED,
            method="TSP",
        )

        exceeds = tour_time_hours > MAX_ROUTE_TIME
        if exceeds:
            exceed_count += 1
        tour_hours.append(tour_time_hours)

        rows.append({
            "instance": inst,
            "supply": str(int(inst.split("_")[4])),
            "n_customers": str(len(customers)),
            "tsp_tour_hours": f"{tour_time_hours:.3f}",
            "exceeds_10h": "1" if exceeds else "0",
        })
        mark = "✓" if exceeds else "✗"
        print(f"  [{idx}/{len(instances)}] {inst}: {tour_time_hours:.2f}h {mark}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["instance", "supply", "n_customers",
                        "tsp_tour_hours", "exceeds_10h"],
        )
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"TSP-of-all summary over {len(instances)} instances:")
    print(f"  min:  {min(tour_hours):.2f}h")
    print(f"  mean: {statistics.mean(tour_hours):.2f}h")
    print(f"  max:  {max(tour_hours):.2f}h")
    print(f"  exceeds 10h: {exceed_count}/{len(instances)} "
          f"({100*exceed_count/len(instances):.0f}%)")
    print(f"CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
