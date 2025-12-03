"""
Runner for MCVRP benchmark instances (Paper Section: Matheuristic Effectiveness).

Reproduces results from Table 1 comparing against Henke (2015, 2019).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from fleetmix import api
from fleetmix.benchmarking.converters.vrp import VRPType
from fleetmix.benchmarking.converters.vrp import convert_vrp_to_fsm as convert_to_fsm
from fleetmix.config import FleetmixParams, load_fleetmix_params
from fleetmix.experiments.reproduce_paper.utils import (
    ProgressTracker,
    ensure_output_dir,
    print_summary_stats,
    skip_if_exists,
)
from fleetmix.utils.logging import log_error, log_success
from fleetmix.utils.save_results import save_optimization_results

__all__ = ["get_mcvrp_instances", "run_mcvrp_instances", "run_single_mcvrp_instance"]

console = Console()


def get_mcvrp_instances() -> list[str]:
    """Get list of all MCVRP instance names."""
    # Find the datasets directory
    from fleetmix import app

    datasets_dir = Path(app.__file__).parent / "benchmarking" / "datasets" / "mcvrp"

    if not datasets_dir.exists():
        return []

    instances = [f.stem for f in sorted(datasets_dir.glob("*.dat"))]
    return instances


def run_single_mcvrp_instance(
    instance: str,
    config: FleetmixParams,
    output_dir: Path,
) -> dict[str, Any] | None:
    """
    Run a single MCVRP instance.

    Args:
        instance: Instance name (without .dat extension)
        config: FleetMix parameters
        output_dir: Output directory for results

    Returns:
        Dictionary with results, or None if failed
    """
    from fleetmix import app

    datasets_dir = Path(app.__file__).parent / "benchmarking" / "datasets" / "mcvrp"
    dat_path = datasets_dir / f"{instance}.dat"

    if not dat_path.exists():
        log_error(f"Instance file not found: {dat_path}")
        return None

    # Check if output already exists
    output_path = output_dir / f"mcvrp_{instance}.json"
    if skip_if_exists(output_path):
        return {"instance": instance, "status": "skipped"}

    try:
        # Convert instance to FSM format
        customers_df, instance_spec = convert_to_fsm(
            VRPType.MCVRP, instance_path=dat_path
        )

        # Update params with instance spec
        params = config.apply_instance_spec(instance_spec)

        # Override output directory
        params = dataclasses.replace(
            params, io=dataclasses.replace(params.io, results_dir=output_dir)
        )

        # Run optimization
        solution = api.optimize(demand=customers_df, config=params, output_dir=None)

        # Save results
        save_optimization_results(
            solution=solution,
            parameters=params,
            filename=str(output_path),
            format="json",
            is_benchmark=True,
            expected_vehicles=params.problem.expected_vehicles,
        )

        log_success(f"Saved result: {output_path}")

        return {
            "instance": instance,
            "status": "success",
            "total_vehicles": solution.total_vehicles,
            "total_cost": solution.total_cost,
            "runtime_sec": solution.solver_runtime_sec or 0.0,
            "expected_vehicles": params.problem.expected_vehicles,
        }

    except Exception as e:
        log_error(f"Error processing {instance}: {e}")
        return {"instance": instance, "status": "error", "error": str(e)}


def run_mcvrp_instances(
    config_path: Path | None = None,
    output_dir: Path | None = None,
    instances: list[str] | None = None,
    list_instances: bool = False,
    skip_existing: bool = True,
) -> None:
    """
    Run MCVRP benchmark instances from Henke (2015, 2019).

    Args:
        config_path: Path to config file (default: synthetic_test_instances/base_config.yaml)
        output_dir: Output directory (default: results/paper/mcvrp_instances/)
        instances: List of specific instances to run (None = all)
        list_instances: Just list available instances and exit
        skip_existing: Skip instances that already have results
    """
    # Get all available instances
    all_instances = get_mcvrp_instances()

    if list_instances:
        console.print("\n[bold]Available MCVRP instances:[/bold]\n")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Instance", style="white")
        table.add_column("Category", style="yellow")

        for instance in all_instances:
            # Parse category from instance name (e.g., "10_3_3_1_01" -> "10_3_3_1")
            parts = instance.split("_")
            if len(parts) >= 4:
                category = "_".join(parts[:4])
            else:
                category = "Unknown"
            table.add_row(instance, category)

        console.print(table)
        console.print(f"\n[green]Total instances: {len(all_instances)}[/green]\n")
        return

    # Filter instances if specified
    if instances:
        missing = [inst for inst in instances if inst not in all_instances]
        if missing:
            log_error(f"Unknown instances: {missing}")
            console.print(
                f"[yellow]Available instances: {', '.join(all_instances[:5])}...[/yellow]"
            )
            raise SystemExit(1)
        instances_to_run = instances
    else:
        instances_to_run = all_instances

    # Setup paths
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
        output_dir = Path("results/paper/mcvrp_instances")

    output_dir = ensure_output_dir(output_dir)

    # Load config
    config = load_fleetmix_params(config_path)

    # Print run info
    console.print("\n[bold]Running MCVRP benchmark instances[/bold]")
    console.print(
        f"Config: {config_path.relative_to(Path.cwd()) if config_path.is_relative_to(Path.cwd()) else config_path}"
    )
    console.print(f"Output: {output_dir}")
    console.print(f"Instances: {len(instances_to_run)}")
    console.print(f"Skip existing: {skip_existing}\n")

    # Run instances
    results = []
    errors = []

    with ProgressTracker("Running MCVRP instances", len(instances_to_run)) as progress:
        for instance in instances_to_run:
            progress.set_description(f"Running {instance}")

            result = run_single_mcvrp_instance(
                instance=instance,
                config=config,
                output_dir=output_dir,
            )

            if result:
                results.append(result)
                if result["status"] == "error":
                    errors.append(instance)

            progress.update()

    # Print summary
    console.print("\n[bold green]✓ MCVRP benchmark completed[/bold green]\n")

    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = len(errors)

    stats = {
        "Total instances": len(instances_to_run),
        "Successfully run": success_count,
        "Skipped (existing)": skipped_count,
        "Errors": error_count,
    }

    if success_count > 0:
        successful_results = [r for r in results if r["status"] == "success"]
        # total_runtime = sum(r.get("runtime_sec", 0) for r in successful_results)

    print_summary_stats("MCVRP Benchmark Summary", stats)

    if errors:
        console.print(f"\n[yellow]Failed instances: {', '.join(errors)}[/yellow]")

    console.print(f"\n[green]Results saved to: {output_dir}[/green]\n")
