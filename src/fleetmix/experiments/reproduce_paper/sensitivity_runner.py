"""
Runner for sensitivity analysis experiments (Paper Section: Benefits of MCVs).

Reproduces results from Figure 4 and Table 2: parameter variation analysis.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from rich.console import Console

from fleetmix import api
from fleetmix.config import load_fleetmix_params
from fleetmix.experiments.reproduce_paper.utils import (
    ProgressTracker,
    aggregate_results_to_dataframe,
    ensure_output_dir,
    print_summary_stats,
    save_summary_table,
    skip_if_exists,
)
from fleetmix.utils.data_processing import load_customer_demand
from fleetmix.utils.logging import log_error
from fleetmix.utils.save_results import save_optimization_results

__all__ = [
    "get_demand_files",
    "get_sensitivity_configs",
    "run_config_on_demand_day",
    "run_sensitivity_analysis",
]

console = Console()


def get_sensitivity_configs(
    base_dir: Path,
    parameters: list[str] | None = None,
    fleet_types: list[str] | None = None,
    variations: list[str] | None = None,
) -> list[tuple[str, str, Path]]:
    """
    Get list of sensitivity analysis config files matching criteria.

    Args:
        base_dir: Base directory for sensitivity configs
        parameters: List of parameters to include (e.g., ['capacity', 'service_time'])
        fleet_types: List of fleet types ('mcv', 'scv', or both)
        variations: List of variations ('minus_50', 'minus_20', 'baseline', 'plus_20', 'plus_50')

    Returns:
        List of tuples: (parameter_name, config_name, config_path)
    """
    if parameters is None:
        parameters = ["capacity", "service_time", "max_route_duration", "variable_cost"]

    if fleet_types is None:
        fleet_types = ["mcv", "scv"]

    if variations is None:
        variations = ["minus_50", "minus_20", "baseline", "plus_20", "plus_50"]

    configs = []

    # Add baselines
    if "baseline" in variations:
        baseline_dir = base_dir / "baselines"
        if baseline_dir.exists():
            for fleet_type in fleet_types:
                baseline_path = baseline_dir / f"baseline_{fleet_type}.yaml"
                if baseline_path.exists():
                    configs.append(
                        ("baseline", f"baseline_{fleet_type}", baseline_path)
                    )

    # Add parameter variations
    for param in parameters:
        param_dir = base_dir / param
        if not param_dir.exists():
            log_error(f"Parameter directory not found: {param_dir}")
            continue

        for config_path in sorted(param_dir.glob("*.yaml")):
            config_name = config_path.stem

            # Check fleet type filter
            if fleet_types and not any(f"{ft}_" in config_name for ft in fleet_types):
                continue

            # Check variation filter
            variation_matched = False
            for variation in variations:
                if variation == "baseline":
                    continue  # Already handled
                if variation in config_name:
                    variation_matched = True
                    break

            if variations and not variation_matched:
                continue

            configs.append((param, config_name, config_path))

    return configs


def get_demand_files(
    demand_dir: Path, specific_days: list[str] | None = None
) -> list[Path]:
    """
    Get list of demand CSV files.

    Args:
        demand_dir: Directory containing demand files
        specific_days: List of specific day names (without .csv), or None for all

    Returns:
        List of demand file paths
    """
    if specific_days:
        files = []
        for day in specific_days:
            if not day.endswith(".csv"):
                day = f"{day}.csv"

            # Try exact match first
            demand_path = demand_dir / day
            if not demand_path.exists():
                # Try with synthetic_sales_ prefix
                if not day.startswith("synthetic_sales_"):
                    day_prefixed = f"synthetic_sales_{day}"
                    if (demand_dir / day_prefixed).exists():
                        demand_path = demand_dir / day_prefixed

            if demand_path.exists():
                files.append(demand_path)
            else:
                log_error(f"Demand file not found: {demand_path}")
        return files
    else:
        return sorted(demand_dir.glob("synthetic_sales_*.csv"))


def run_config_on_demand_day(
    config_path: Path,
    demand_path: Path,
    output_dir: Path,
    skip_existing: bool = True,
) -> dict[str, Any] | None:
    """
    Run a single config on a single demand day.

    Args:
        config_path: Path to config YAML
        demand_path: Path to demand CSV
        output_dir: Output directory
        skip_existing: Whether to skip if output file exists

    Returns:
        Dictionary with results, or None if failed
    """
    config_name = config_path.stem
    demand_name = demand_path.stem

    # Create output path
    output_path = output_dir / f"{config_name}_{demand_name}.json"

    if skip_existing and skip_if_exists(output_path):
        return {"config": config_name, "demand_day": demand_name, "status": "skipped"}

    try:
        # Load config
        params = load_fleetmix_params(config_path)

        # Load demand
        customers_df = load_customer_demand(str(demand_path))

        # Override output directory and demand file
        params = dataclasses.replace(
            params,
            io=dataclasses.replace(
                params.io, results_dir=output_dir, demand_file=str(demand_path)
            ),
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
        )

        # Extract key metrics
        return {
            "config": config_name,
            "demand_day": demand_name,
            "status": "success",
            "total_cost": solution.total_cost,
            "total_vehicles": solution.total_vehicles,
            "total_fixed_cost": solution.total_fixed_cost,
            "total_variable_cost": solution.total_variable_cost,
            "num_customers": len(customers_df),
            "runtime_sec": solution.solver_runtime_sec or 0.0,
        }

    except Exception as e:
        log_error(f"Error processing {config_name} × {demand_name}: {e}")
        return {
            "config": config_name,
            "demand_day": demand_name,
            "status": "error",
            "error": str(e),
        }


def run_sensitivity_analysis(
    output_dir: Path | None = None,
    parameters: str | None = None,
    fleet_types: str | None = None,
    variations: str | None = None,
    demand_days: str | None = None,
    skip_existing: bool = True,
) -> None:
    """
    Run sensitivity analysis experiments.

    Args:
        output_dir: Output directory
        parameters: Comma-separated parameters or 'all'
        fleet_types: Comma-separated fleet types: mcv, scv, or 'both'
        variations: Comma-separated variations or 'all'
        demand_days: Comma-separated demand days or 'all'
        skip_existing: Skip configs that already have results
    """
    # Parse inputs
    from fleetmix import app

    base_config_dir = (
        Path(app.__file__).parent / "config" / "experiments" / "sensitivity_analysis"
    )
    demand_dir = Path(app.__file__).parent / "benchmarking" / "datasets" / "case"

    # Parse parameters
    if parameters is None or parameters.lower() == "all":
        param_list = ["capacity", "service_time", "max_route_duration"]
    else:
        param_list = [p.strip() for p in parameters.split(",")]

    # Parse fleet types
    if fleet_types is None or fleet_types.lower() == "both":
        fleet_list = ["mcv", "scv"]
    else:
        fleet_list = [ft.strip() for ft in fleet_types.split(",")]

    # Parse variations
    if variations is None or variations.lower() == "all":
        variation_list = ["minus_50", "minus_20", "baseline", "plus_20", "plus_50"]
    else:
        variation_list = [v.strip() for v in variations.split(",")]

    # Get configs and demand files
    configs = get_sensitivity_configs(
        base_config_dir, param_list, fleet_list, variation_list
    )

    # Parse demand days
    if demand_days and demand_days.lower() != "all":
        day_list = [d.strip() for d in demand_days.split(",")]
        demand_files = get_demand_files(demand_dir, day_list)
    else:
        demand_files = get_demand_files(demand_dir, None)

    if not configs:
        log_error("No configs found matching criteria")
        return

    if not demand_files:
        log_error("No demand files found")
        return

    # Setup output directory
    if output_dir is None:
        output_dir = Path("results/paper/sensitivity_analysis")

    output_dir = ensure_output_dir(output_dir)

    # Print run info
    console.print("\n[bold]Running sensitivity analysis[/bold]")
    console.print(f"Parameters: {', '.join(param_list)}")
    console.print(f"Fleet types: {', '.join(fleet_list)}")
    console.print(f"Variations: {', '.join(variation_list)}")
    console.print(f"Configs: {len(configs)}")
    console.print(f"Demand days: {len(demand_files)}")
    console.print(f"Total runs: {len(configs) * len(demand_files)}")
    console.print(f"Output: {output_dir}")
    console.print(f"Skip existing: {skip_existing}\n")

    # Run experiments
    results = []
    total_runs = len(configs) * len(demand_files)

    with ProgressTracker("Running sensitivity analysis", total_runs) as progress:
        for param, config_name, config_path in configs:
            # Create subdirectory for this parameter
            param_output_dir = output_dir / param / config_name
            # Ensure the directory exists
            param_output_dir.mkdir(parents=True, exist_ok=True)

            for demand_path in demand_files:
                progress.set_description(
                    f"{config_name} × {demand_path.stem.replace('sales_', '').replace('_demand', '')}"
                )

                result = run_config_on_demand_day(
                    config_path=config_path,
                    demand_path=demand_path,
                    output_dir=param_output_dir,
                    skip_existing=skip_existing,
                )

                if result:
                    result["parameter"] = param
                    results.append(result)

                progress.update()

    # Print summary
    console.print("\n[bold green]✓ Sensitivity analysis completed[/bold green]\n")

    success_count = sum(1 for r in results if r.get("status") == "success")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    error_count = sum(1 for r in results if r.get("status") == "error")

    stats = {
        "Total runs": len(results),
        "Successfully run": success_count,
        "Skipped (existing)": skipped_count,
        "Errors": error_count,
    }

    if success_count > 0:
        successful_results = [r for r in results if r.get("status") == "success"]
        total_runtime = sum(r.get("runtime_sec", 0) for r in successful_results)
        stats["Total runtime"] = f"{total_runtime / 3600:.1f}h"
        stats["Average runtime"] = f"{total_runtime / success_count:.1f}s per run"

    print_summary_stats("Sensitivity Analysis Summary", stats)

    # Save summary
    if results:
        df = aggregate_results_to_dataframe(results)
        summary_path = output_dir / "summary"
        save_summary_table(df, summary_path, formats=["parquet", "csv"])

    console.print(f"\n[green]Results saved to: {output_dir}[/green]\n")
