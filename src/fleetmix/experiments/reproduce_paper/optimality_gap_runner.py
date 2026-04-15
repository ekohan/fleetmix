"""
Runner for optimality gap analysis on small Henke instances.

Compares the matheuristic solution against the provably optimal solution
obtained by exhaustively enumerating all feasible clusters and feeding
them to the MILP. Primary metric: fleet size (number of vehicles).

Usage (CLI)::

    uv run python -m fleetmix.experiments.reproduce_paper.optimality_gap_runner

Or from Python::

    from fleetmix.experiments.reproduce_paper.optimality_gap_runner import (
        run_optimality_gap_analysis,
    )
    run_optimality_gap_analysis()
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from fleetmix.benchmarking.converters.vrp import VRPType, convert_vrp_to_fsm
from fleetmix.clustering import generate_feasible_clusters
from fleetmix.clustering.exhaustive import generate_exhaustive_clusters
from fleetmix.config import FleetmixParams, load_fleetmix_params
from fleetmix.core_types import Customer, FleetmixSolution
from fleetmix.experiments.reproduce_paper.utils import (
    ProgressTracker,
    ensure_output_dir,
    print_summary_stats,
    save_summary_table,
)
from fleetmix.optimization import optimize_fleet
from fleetmix.post_optimization import improve_solution
from fleetmix.utils.logging import FleetmixLogger, log_error, log_success
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations

__all__ = ["run_optimality_gap_analysis", "run_single_gap_instance"]

console = Console()


def _get_henke_instances(max_customers: int = 15) -> list[Path]:
    """Get Henke instance .dat files up to a given customer count."""
    from fleetmix import app

    datasets_dir = Path(app.__file__).parent / "benchmarking" / "datasets" / "mcvrp"
    if not datasets_dir.exists():
        return []

    paths = []
    for dat_file in sorted(datasets_dir.glob("*.dat")):
        # Parse customer count from filename: YEAR_CUSTOMERS_PRODUCTS_COMPARTMENTS_...
        parts = dat_file.stem.split("_")
        if len(parts) >= 2:
            try:
                n_customers = int(parts[1])
            except ValueError:
                continue
            if n_customers <= max_customers:
                paths.append(dat_file)
    return paths


def _solve_with_clusters(
    clusters: list,
    configs: list,
    customers: list,
    params: FleetmixParams,
    apply_post_opt: bool = False,
) -> FleetmixSolution:
    """Run the MILP (and optionally post-optimization) on a cluster pool."""
    solution = optimize_fleet(
        clusters=clusters,
        configurations=configs,
        customers=customers,
        parameters=params,
    )
    if apply_post_opt and params.algorithm.post_optimization:
        solution = improve_solution(solution, configs, customers, params)
    return solution


def run_single_gap_instance(
    dat_path: Path,
    config: FleetmixParams,
) -> dict[str, Any] | None:
    """Run both matheuristic and exhaustive pipelines on one instance.

    Returns a dict with fleet-size and cost comparisons, or None on failure.
    """
    instance_name = dat_path.stem
    try:
        # ---- Convert instance ----
        customers_df, instance_spec = convert_vrp_to_fsm(
            VRPType.MCVRP, instance_path=dat_path
        )
        params = config.apply_instance_spec(instance_spec)

        # Disable split-stops for this analysis
        params = dataclasses.replace(
            params,
            problem=dataclasses.replace(params.problem, allow_split_stops=False),
        )

        configs = generate_vehicle_configurations(
            params.problem.vehicles, params.problem.goods
        )
        customers = Customer.from_dataframe(customers_df)

        n_customers = len(customers)

        # ---- Matheuristic pipeline ----
        t0 = time.perf_counter()
        heuristic_clusters = generate_feasible_clusters(
            customers=customers, configurations=configs, params=params
        )
        heuristic_solution = _solve_with_clusters(
            heuristic_clusters, configs, customers, params, apply_post_opt=True
        )
        heuristic_time = time.perf_counter() - t0

        # ---- Exhaustive pipeline ----
        t0 = time.perf_counter()
        exhaustive_clusters = generate_exhaustive_clusters(
            customers=customers, configurations=configs, params=params
        )
        # No post-optimization needed — full cluster set already contains the
        # optimal, and post-opt merging can only produce clusters that are
        # already in the exhaustive set.
        exhaustive_params = dataclasses.replace(
            params,
            algorithm=dataclasses.replace(params.algorithm, post_optimization=False),
        )
        exhaustive_solution = _solve_with_clusters(
            exhaustive_clusters, configs, customers, exhaustive_params
        )
        exhaustive_time = time.perf_counter() - t0

        # ---- Compare ----
        h_vehicles = heuristic_solution.total_vehicles
        e_vehicles = exhaustive_solution.total_vehicles
        h_cost = heuristic_solution.total_cost
        e_cost = exhaustive_solution.total_cost

        vehicle_gap = h_vehicles - e_vehicles  # positive = heuristic uses more
        cost_gap_pct = (
            ((h_cost - e_cost) / e_cost * 100) if e_cost > 0 else 0.0
        )

        result: dict[str, Any] = {
            "instance": instance_name,
            "n_customers": n_customers,
            "expected_vehicles": params.problem.expected_vehicles,
            # Matheuristic
            "heuristic_vehicles": h_vehicles,
            "heuristic_cost": round(h_cost, 2),
            "heuristic_clusters": len(heuristic_clusters),
            "heuristic_time_sec": round(heuristic_time, 2),
            # Exhaustive (optimal)
            "exhaustive_vehicles": e_vehicles,
            "exhaustive_cost": round(e_cost, 2),
            "exhaustive_clusters": len(exhaustive_clusters),
            "exhaustive_time_sec": round(exhaustive_time, 2),
            # Gap
            "vehicle_gap": vehicle_gap,
            "cost_gap_pct": round(cost_gap_pct, 2),
            "status": "success",
        }

        symbol = "=" if vehicle_gap == 0 else ("+" if vehicle_gap > 0 else "-")
        log_success(
            f"{instance_name}: heuristic={h_vehicles}v, optimal={e_vehicles}v "
            f"({symbol}{abs(vehicle_gap)}), cost gap={cost_gap_pct:+.1f}%"
        )
        return result

    except Exception as e:
        log_error(f"Error processing {instance_name}: {e}")
        return {"instance": instance_name, "status": "error", "error": str(e)}


def run_optimality_gap_analysis(
    config_path: Path | None = None,
    output_dir: Path | None = None,
    max_customers: int = 15,
    instances: list[str] | None = None,
    route_time_method: str | None = None,
) -> None:
    """Run the full optimality gap experiment.

    Parameters
    ----------
    config_path:
        YAML config for the solver. Defaults to the synthetic-test-instances
        base config.
    output_dir:
        Where to write per-instance JSON and summary CSV/parquet.
    max_customers:
        Only run instances with ≤ this many customers (default 15).
    instances:
        Optional explicit list of instance stems to run.
    route_time_method:
        Override route-time estimation ('BHH' or 'TSP'). If None, uses
        whatever the config specifies.
    """
    # ---- Resolve paths ----
    if config_path is None:
        from fleetmix import app

        config_path = (
            Path(app.__file__).parent
            / "config"
            / "experiments"
            / "synthetic_test_instances"
            / "base_config.yaml"
        )
    if output_dir is None:
        output_dir = Path("results/paper/optimality_gap")

    output_dir = ensure_output_dir(output_dir)

    # ---- Load config ----
    config = load_fleetmix_params(config_path)

    # Override route-time method if requested
    if route_time_method is not None:
        config = dataclasses.replace(
            config,
            algorithm=dataclasses.replace(
                config.algorithm, route_time_estimation=route_time_method
            ),
        )

    # ---- Gather instances ----
    all_paths = _get_henke_instances(max_customers=max_customers)
    if instances:
        all_paths = [p for p in all_paths if p.stem in instances]

    if not all_paths:
        console.print("[red]No instances found.[/red]")
        return

    console.print("\n[bold]Optimality Gap Analysis[/bold]")
    console.print(f"Config: {config_path}")
    console.print(f"Route-time: {config.algorithm.route_time_estimation}")
    console.print(f"Output: {output_dir}")
    console.print(f"Instances: {len(all_paths)} (max {max_customers} customers)\n")

    # ---- Run ----
    results: list[dict[str, Any]] = []

    with ProgressTracker("Optimality gap", len(all_paths)) as progress:
        for dat_path in all_paths:
            progress.set_description(f"Running {dat_path.stem}")
            result = run_single_gap_instance(dat_path, config)
            if result:
                results.append(result)
                # Save per-instance JSON
                out_file = output_dir / f"gap_{dat_path.stem}.json"
                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2)
            progress.update()

    # ---- Summary ----
    import pandas as pd

    df = pd.DataFrame(results)
    success_df = df[df["status"] == "success"]

    if success_df.empty:
        console.print("[red]No successful results.[/red]")
        return

    # Summary table
    table = Table(title="Optimality Gap Summary by Category")
    table.add_column("Category")
    table.add_column("Instances", justify="right")
    table.add_column("Avg Heuristic Vehicles", justify="right")
    table.add_column("Avg Optimal Vehicles", justify="right")
    table.add_column("Exact Match", justify="right")
    table.add_column("Avg Cost Gap %", justify="right")

    # Group by customer count
    for n_cust, group in success_df.groupby("n_customers"):
        exact_match = (group["vehicle_gap"] == 0).sum()
        table.add_row(
            f"n={n_cust}",
            str(len(group)),
            f"{group['heuristic_vehicles'].mean():.1f}",
            f"{group['exhaustive_vehicles'].mean():.1f}",
            f"{exact_match}/{len(group)}",
            f"{group['cost_gap_pct'].mean():.2f}%",
        )

    # Overall
    exact_total = (success_df["vehicle_gap"] == 0).sum()
    table.add_row(
        "[bold]Overall[/bold]",
        str(len(success_df)),
        f"{success_df['heuristic_vehicles'].mean():.1f}",
        f"{success_df['exhaustive_vehicles'].mean():.1f}",
        f"[bold]{exact_total}/{len(success_df)}[/bold]",
        f"[bold]{success_df['cost_gap_pct'].mean():.2f}%[/bold]",
    )

    console.print(table)

    # Save summary
    save_summary_table(df, output_dir / "optimality_gap_summary")

    stats = {
        "Total instances": len(results),
        "Successful": len(success_df),
        "Errors": len(df) - len(success_df),
        "Fleet-size exact matches": f"{exact_total}/{len(success_df)}",
        "Mean cost gap": f"{success_df['cost_gap_pct'].mean():.2f}%",
        "Max cost gap": f"{success_df['cost_gap_pct'].max():.2f}%",
    }
    print_summary_stats("Optimality Gap Statistics", stats)

    console.print(f"\n[green]Results saved to {output_dir}[/green]\n")


if __name__ == "__main__":
    run_optimality_gap_analysis()
