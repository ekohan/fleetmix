#!/usr/bin/env python3
"""
Henke 2015 n=10 Heterogeneous-Fleet Experiment (reviewer revision).

Runs matheuristic + exhaustive on the 150 Henke 2015 n=10 instances with a
heterogeneous A+B fleet (Bogotá case-study cost structure, 10h max route
duration). Both BHH and TSP route-time methods. Resumable.

CSV schema includes per-instance fleet composition (n_A, n_B) for both
pipelines, plus cluster counts fed to the MILP.

Usage:
    FSM_SOLVER=cbc uv run python scripts/run_henke_heterogeneous.py
    uv run python scripts/run_henke_heterogeneous.py --report-only
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = (
    PROJECT_ROOT
    / "src/fleetmix/config/experiments/henke_heterogeneous/base_config.yaml"
)
DATASETS_DIR = PROJECT_ROOT / "src/fleetmix/benchmarking/datasets/mcvrp"
OUTPUT_DIR = PROJECT_ROOT / "results/henke_heterogeneous"
CSV_PATH = OUTPUT_DIR / "hetero_all.csv"

CSV_FIELDS = [
    "instance",
    "supply",
    "method",
    # Matheuristic
    "h_vehicles",
    "h_cost",
    "h_clusters",
    "h_time",
    "h_n_A",
    "h_n_B",
    # Exhaustive
    "e_vehicles",
    "e_cost",
    "e_clusters",
    "e_time",
    "e_n_A",
    "e_n_B",
    # Gap
    "v_gap",
    "cost_gap_pct",
]


def get_experiment_plan() -> list[tuple[str, str]]:
    """150 Henke 2015 n=10 instances × {BHH, TSP}."""
    plan: list[tuple[str, str]] = []
    for method in ("BHH", "TSP"):
        for s in (1, 2, 3):
            for i in range(1, 51):
                plan.append((f"2015_10_3_3_{s}_({i:02d})", method))
    return plan


def parse_supply(instance_name: str) -> int:
    return int(instance_name.split("_")[4])


def run_instance(instance_name: str, method: str, config) -> dict[str, str]:
    """Run heuristic + exhaustive on one (instance, method); return CSV row."""
    from fleetmix.benchmarking.converters.vrp import VRPType, convert_vrp_to_fsm
    from fleetmix.clustering import generate_feasible_clusters
    from fleetmix.clustering.exhaustive import generate_exhaustive_clusters
    from fleetmix.core_types import Customer
    from fleetmix.merging.core import _merged_route_time_cache
    from fleetmix.optimization import optimize_fleet
    from fleetmix.post_optimization import improve_solution
    from fleetmix.utils.route_time import clear_matrix_cache, clear_tsp_result_cache
    from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations

    # Clear module-level caches that would otherwise leak state between
    # benchmark instances that reuse customer IDs (Henke's "1..10" pattern).
    _merged_route_time_cache.clear()
    clear_tsp_result_cache()
    clear_matrix_cache()

    dat_path = DATASETS_DIR / f"{instance_name}.dat"
    customers_df, spec = convert_vrp_to_fsm(VRPType.MCVRP, instance_path=dat_path)
    params = config.apply_instance_spec(spec)

    # Restore YAML fleet — the MCVRP converter injects a single "MCVRP"
    # vehicle via InstanceSpec; for this experiment we want A+B from the YAML.
    params = dataclasses.replace(
        params,
        problem=dataclasses.replace(
            params.problem,
            vehicles=config.problem.vehicles,
            goods=config.problem.goods,
            allow_split_stops=False,
        ),
        algorithm=dataclasses.replace(
            params.algorithm, route_time_estimation=method
        ),
    )

    cfgs = generate_vehicle_configurations(
        params.problem.vehicles, params.problem.goods
    )
    customers = Customer.from_dataframe(customers_df)

    # Matheuristic
    t0 = time.perf_counter()
    h_cl = generate_feasible_clusters(
        customers=customers, configurations=cfgs, params=params
    )
    h_sol = optimize_fleet(
        clusters=h_cl, configurations=cfgs, customers=customers, parameters=params
    )
    if params.algorithm.post_optimization:
        h_sol = improve_solution(h_sol, cfgs, customers, params)
    h_time = time.perf_counter() - t0

    # Exhaustive
    t0 = time.perf_counter()
    e_cl = generate_exhaustive_clusters(
        customers=customers, configurations=cfgs, params=params
    )
    e_params = dataclasses.replace(
        params,
        algorithm=dataclasses.replace(params.algorithm, post_optimization=False),
    )
    e_sol = optimize_fleet(
        clusters=e_cl, configurations=cfgs, customers=customers, parameters=e_params
    )
    e_time = time.perf_counter() - t0

    cost_gap = (
        (h_sol.total_cost - e_sol.total_cost) / e_sol.total_cost * 100
        if e_sol.total_cost > 0
        else 0.0
    )

    h_used = dict(h_sol.vehicles_used)
    e_used = dict(e_sol.vehicles_used)

    return {
        "instance": instance_name,
        "supply": str(parse_supply(instance_name)),
        "method": method,
        "h_vehicles": str(h_sol.total_vehicles),
        "h_cost": f"{h_sol.total_cost:.2f}",
        "h_clusters": str(len(h_cl)),
        "h_time": f"{h_time:.1f}",
        "h_n_A": str(h_used.get("A", 0)),
        "h_n_B": str(h_used.get("B", 0)),
        "e_vehicles": str(e_sol.total_vehicles),
        "e_cost": f"{e_sol.total_cost:.2f}",
        "e_clusters": str(len(e_cl)),
        "e_time": f"{e_time:.1f}",
        "e_n_A": str(e_used.get("A", 0)),
        "e_n_B": str(e_used.get("B", 0)),
        "v_gap": str(h_sol.total_vehicles - e_sol.total_vehicles),
        "cost_gap_pct": f"{cost_gap:.2f}",
    }


def load_results() -> dict[tuple[str, str], dict[str, str]]:
    results: dict[tuple[str, str], dict[str, str]] = {}
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                results[(row["instance"], row["method"])] = dict(row)
    return results


def save_results(results: dict[tuple[str, str], dict[str, str]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(results.values(), key=lambda r: (r["method"], r["instance"]))
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Henke heterogeneous experiment")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N pending (instance, method) pairs (smoke test)."
    )
    args = parser.parse_args()

    from fleetmix.config import load_fleetmix_params
    config = load_fleetmix_params(CONFIG_PATH)

    plan = get_experiment_plan()
    results = load_results()
    todo = [p for p in plan if p not in results]
    if args.limit:
        todo = todo[: args.limit]

    print(f"Henke Heterogeneous Experiment (Vehicle A+B)")
    print(f"  Config:     {CONFIG_PATH}")
    print(f"  Planned:    {len(plan)} ({len(plan) - len([p for p in plan if p not in results])} done)")
    print(f"  Running:    {len(todo)} pair(s)")
    print()

    for idx, (inst, meth) in enumerate(todo, 1):
        print(f"  [{idx}/{len(todo)}] {inst} ({meth})...", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            row = run_instance(inst, meth, config)
            elapsed = time.perf_counter() - t0
            results[(inst, meth)] = row
            print(
                f"v_gap={row['v_gap']} cost_gap={row['cost_gap_pct']}% "
                f"fleet={row['h_n_A']}A+{row['h_n_B']}B "
                f"cl={row['h_clusters']}/{row['e_clusters']} "
                f"({elapsed:.1f}s)"
            )
            save_results(results)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print()
    print(f"CSV: {CSV_PATH} ({len(results)} rows)")


if __name__ == "__main__":
    main()
