"""
Runner for Henke 2015 heterogeneous-fleet experiments (paper Section 6.3).

Two modes:

- ``experiment`` (default): runs the matheuristic and exhaustive enumeration on
  all 150 Henke 2015 n=10 instances with a heterogeneous A+B fleet (Bogotá
  case-study cost structure, 10h max route duration) under BHH and TSP route
  times. Produces ``hetero_all.csv``, the source of Table 5 numbers (78/150
  optimal, 1.67% average gap, 150/150 fleet match, 139/150 composition match,
  7.75x average speedup over the complete model).

- ``tsp-of-all``: computes the TSP tour over all customers for each instance.
  Produces ``tsp_of_all.csv`` and the range / mean / exceed-10h statistics
  quoted in Section 6.3 (7.93-11.36 h, mean 9.89 h, 59 of 150 exceed 10 h).
"""

from __future__ import annotations

import csv
import dataclasses
import statistics
import time
from pathlib import Path
from typing import Any

__all__ = ["run_henke_heterogeneous", "run_tsp_of_all"]


def _project_paths() -> tuple[Path, Path, Path]:
    """Resolve (config_path, datasets_dir, default_output_dir)."""
    from fleetmix import app as fleetmix_app

    pkg_root = Path(fleetmix_app.__file__).parent
    config_path = (
        pkg_root
        / "config"
        / "experiments"
        / "henke_heterogeneous"
        / "base_config.yaml"
    )
    datasets_dir = pkg_root / "benchmarking" / "datasets" / "mcvrp"
    default_output = Path("results/henke_heterogeneous")
    return config_path, datasets_dir, default_output


# ---------------------------------------------------------------------------
# Mode 1: experiment (matheuristic vs exhaustive over all n=10 instances)
# ---------------------------------------------------------------------------

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


def _get_experiment_plan() -> list[tuple[str, str]]:
    """150 Henke 2015 n=10 instances × {BHH, TSP}."""
    plan: list[tuple[str, str]] = []
    for method in ("BHH", "TSP"):
        for s in (1, 2, 3):
            for i in range(1, 51):
                plan.append((f"2015_10_3_3_{s}_({i:02d})", method))
    return plan


def _parse_supply(instance_name: str) -> int:
    return int(instance_name.split("_")[4])


def _run_instance(
    instance_name: str,
    method: str,
    config: Any,
    datasets_dir: Path,
) -> dict[str, str]:
    """Run matheuristic + exhaustive on one (instance, method); return CSV row."""
    from fleetmix.benchmarking.converters.vrp import VRPType, convert_vrp_to_fsm
    from fleetmix.clustering import generate_feasible_clusters
    from fleetmix.clustering.exhaustive_enumerator import generate_exhaustive_clusters
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

    dat_path = datasets_dir / f"{instance_name}.dat"
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
        "supply": str(_parse_supply(instance_name)),
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


def _load_results(csv_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    results: dict[tuple[str, str], dict[str, str]] = {}
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                results[(row["instance"], row["method"])] = dict(row)
    return results


def _save_results(
    csv_path: Path, results: dict[tuple[str, str], dict[str, str]]
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(results.values(), key=lambda r: (r["method"], r["instance"]))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_henke_heterogeneous(
    output_dir: Path | None = None,
    limit: int | None = None,
) -> None:
    """Matheuristic vs exhaustive over all 150 Henke 2015 n=10 instances."""
    from fleetmix.config import load_fleetmix_params

    config_path, datasets_dir, default_output = _project_paths()
    if output_dir is None:
        output_dir = default_output
    csv_path = output_dir / "hetero_all.csv"

    config = load_fleetmix_params(config_path)

    plan = _get_experiment_plan()
    results = _load_results(csv_path)
    todo = [p for p in plan if p not in results]
    if limit:
        todo = todo[:limit]

    print("Henke Heterogeneous Experiment (Vehicle A+B)")
    print(f"  Config:     {config_path}")
    print(f"  Planned:    {len(plan)} ({len(plan) - len(todo)} done)")
    print(f"  Running:    {len(todo)} pair(s)")
    print()

    for idx, (inst, meth) in enumerate(todo, 1):
        print(f"  [{idx}/{len(todo)}] {inst} ({meth})...", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            row = _run_instance(inst, meth, config, datasets_dir)
            elapsed = time.perf_counter() - t0
            results[(inst, meth)] = row
            print(
                f"v_gap={row['v_gap']} cost_gap={row['cost_gap_pct']}% "
                f"fleet={row['h_n_A']}A+{row['h_n_B']}B "
                f"cl={row['h_clusters']}/{row['e_clusters']} "
                f"({elapsed:.1f}s)"
            )
            _save_results(csv_path, results)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()

    print()
    print(f"CSV: {csv_path} ({len(results)} rows)")


# ---------------------------------------------------------------------------
# Mode 2: tsp-of-all (single-vehicle TSP justification for 10h max-route)
# ---------------------------------------------------------------------------

AVG_SPEED = 30.0  # km/h — Bogotá case study
SERVICE_TIME = 25.0  # minutes — Bogotá case study
MAX_ROUTE_TIME = 10.0  # hours — the threshold checked


def run_tsp_of_all(output_dir: Path | None = None) -> None:
    """Single-vehicle TSP over all customers per Henke 2015 n=10 instance."""
    from fleetmix.benchmarking.converters.vrp import VRPType, convert_vrp_to_fsm
    from fleetmix.core_types import Customer
    from fleetmix.utils.route_time import estimate_route_time

    _, datasets_dir, default_output = _project_paths()
    if output_dir is None:
        output_dir = default_output
    out_csv = output_dir / "tsp_of_all.csv"

    instances: list[str] = []
    for s in (1, 2, 3):
        for i in range(1, 51):
            instances.append(f"2015_10_3_3_{s}_({i:02d})")

    rows: list[dict[str, str]] = []
    exceed_count = 0
    tour_hours: list[float] = []

    print(f"Computing TSP-of-all on {len(instances)} instances...")
    for idx, inst in enumerate(instances, 1):
        dat_path = datasets_dir / f"{inst}.dat"
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

        rows.append(
            {
                "instance": inst,
                "supply": str(int(inst.split("_")[4])),
                "n_customers": str(len(customers)),
                "tsp_tour_hours": f"{tour_time_hours:.3f}",
                "exceeds_10h": "1" if exceeds else "0",
            }
        )
        mark = "✓" if exceeds else "✗"
        print(f"  [{idx}/{len(instances)}] {inst}: {tour_time_hours:.2f}h {mark}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "supply",
                "n_customers",
                "tsp_tour_hours",
                "exceeds_10h",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"TSP-of-all summary over {len(instances)} instances:")
    print(f"  min:  {min(tour_hours):.2f}h")
    print(f"  mean: {statistics.mean(tour_hours):.2f}h")
    print(f"  max:  {max(tour_hours):.2f}h")
    print(
        f"  exceeds 10h: {exceed_count}/{len(instances)} "
        f"({100 * exceed_count / len(instances):.0f}%)"
    )
    print(f"CSV: {out_csv}")
