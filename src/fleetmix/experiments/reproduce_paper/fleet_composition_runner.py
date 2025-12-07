"""
Runner for fleet composition analysis (Paper Section: Cost Structure Impact).

Reproduces results from Figure 7 and Table 3: alpha-C grid analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console

from fleetmix.experiments.fleet_composition.config import (
    ALPHA_GRID as DEFAULT_ALPHA_GRID,
)
from fleetmix.experiments.fleet_composition.config import C_VALUES as DEFAULT_C_VALUES
from fleetmix.experiments.fleet_composition.fleet_templates import (
    make_mixed_fleet,
    make_scv_fleet,
)
from fleetmix.experiments.fleet_composition.run_grid_mixed import (
    _collect_day_summary,
)
from fleetmix.experiments.reproduce_paper.utils import (
    ProgressTracker,
    ensure_output_dir,
    print_summary_stats,
    skip_if_exists,
)
from fleetmix.utils.logging import log_error

__all__ = [
    "aggregate_results",
    "get_demand_files",
    "parse_grid_values",
    "run_fleet_composition",
    "run_mixed_grid",
    "run_scv_baselines",
]

console = Console()


def parse_grid_values(
    value_str: str | None, default_values: list[float]
) -> list[float]:
    """
    Parse comma-separated values or return default.

    Args:
        value_str: Comma-separated values or 'default'
        default_values: Default values to use

    Returns:
        List of float values
    """
    if value_str is None or value_str.lower() == "default":
        return default_values

    try:
        values = [float(v.strip()) for v in value_str.split(",")]
        return values
    except ValueError as e:
        raise ValueError(f"Invalid grid values: {value_str}. Error: {e}")


def get_demand_files(
    demand_dir: Path, specific_days: list[str] | None = None
) -> list[Path]:
    """
    Get list of demand CSV files.

    Args:
        demand_dir: Directory containing demand files
        specific_days: List of specific day names, or None for all

    Returns:
        List of demand file paths
    """
    if specific_days and specific_days[0].lower() != "all":
        files = []
        for day in specific_days:
            # Handle with or without .csv extension
            if not day.endswith(".csv"):
                day = f"{day}.csv"
            # Also try with synthetic_sales_ prefix if not present
            if not day.startswith("synthetic_sales_") and not day.startswith("sales_"):
                day = f"synthetic_sales_{day}"
            demand_path = demand_dir / day
            if demand_path.exists():
                files.append(demand_path)
            else:
                log_error(f"Demand file not found: {demand_path}")
        return sorted(files)
    else:
        return sorted(demand_dir.glob("synthetic_sales_*.csv"))


def run_scv_baselines(
    demand_files: list[Path],
    output_dir: Path,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """
    Run SCV baseline for each demand day.

    Args:
        demand_files: List of demand file paths
        output_dir: Output directory for results
        skip_existing: Skip if result already exists

    Returns:
        List of result dictionaries
    """
    results = []

    console.print("\n[bold]Phase 1: Running SCV baselines[/bold]")

    with ProgressTracker("SCV baselines", len(demand_files)) as progress:
        for demand_path in demand_files:
            json_path = output_dir / f"{demand_path.stem}_SCV_BASE.json"

            if skip_existing and skip_if_exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                results.append(data)
            else:
                params = make_scv_fleet(demand_path.stem)

                data = _collect_day_summary(
                    demand_path, params, "SCV_BASE", alpha=1.0, C=0.0
                )
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
                results.append(data)

            progress.update()

    console.print(f"[green]✓ Completed {len(results)} SCV baselines[/green]\n")
    return results


def run_mixed_grid(
    demand_files: list[Path],
    alpha_grid: list[float],
    c_values: list[float],
    output_dir: Path,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """
    Run mixed fleet analysis across alpha-C grid.

    Args:
        demand_files: List of demand file paths
        alpha_grid: List of alpha values
        c_values: List of C values
        output_dir: Output directory for results
        skip_existing: Skip if result already exists

    Returns:
        List of result dictionaries
    """
    results = []
    total_runs = len(demand_files) * len(alpha_grid) * len(c_values)

    console.print("\n[bold]Phase 2: Running mixed fleet grid[/bold]")
    console.print(f"Alpha values: {len(alpha_grid)} points")
    console.print(f"C values: {len(c_values)} points")
    console.print(f"Demand days: {len(demand_files)}")
    console.print(f"Total runs: {total_runs}\n")

    with ProgressTracker("Mixed fleet grid", total_runs) as progress:
        for demand_path in demand_files:
            for alpha in alpha_grid:
                for C in c_values:
                    json_path = (
                        output_dir
                        / f"{demand_path.stem}_MIXED_{alpha:.2f}_{C:.0f}.json"
                    )

                    desc = f"{demand_path.stem.replace('sales_', '').replace('_demand', '')} α={alpha:.1f} C={C:.0f}"
                    progress.set_description(desc)

                    if skip_existing and skip_if_exists(json_path):
                        with open(json_path, "r") as f:
                            data = json.load(f)
                        results.append(data)
                    else:
                        params = make_mixed_fleet(
                            alpha=alpha,
                            C=C,
                            demand_day=demand_path.stem,
                            allow_split=True,
                        )

                        data = _collect_day_summary(
                            demand_path, params, "MIXED", alpha=alpha, C=C
                        )
                        with open(json_path, "w") as f:
                            json.dump(data, f, indent=2)
                        results.append(data)

                    progress.update()

    console.print(f"\n[green]✓ Completed {len(results)} mixed fleet runs[/green]\n")
    return results


def aggregate_results(
    scv_results: list[dict[str, Any]],
    mixed_results: list[dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Aggregate results and compute deltas vs SCV baseline.

    Args:
        scv_results: List of SCV baseline results
        mixed_results: List of mixed fleet results
        output_dir: Output directory

    Returns:
        Aggregated DataFrame
    """
    console.print("[bold]Aggregating results...[/bold]")

    # Convert to DataFrames
    scv_df = pd.DataFrame(scv_results)
    mixed_df = pd.DataFrame(mixed_results)

    # Compute delta vs SCV
    baselines = scv_df[["instance", "total_cost"]].rename(
        columns={"total_cost": "scv_cost"}
    )
    mixed_with_baseline = mixed_df.merge(baselines, on="instance", how="left")
    mixed_with_baseline["delta_cost_pct_vs_scv"] = (
        100.0
        * (mixed_with_baseline["total_cost"] - mixed_with_baseline["scv_cost"])
        / mixed_with_baseline["scv_cost"]
    )

    # Combine
    final_df = pd.concat([mixed_with_baseline, scv_df], ignore_index=True)

    # Save summary
    summary_path = output_dir / "summary_mixed.parquet"
    final_df.to_parquet(summary_path)
    console.print(f"[green]✓ Saved summary to {summary_path}[/green]")

    # Also save CSV
    csv_path = output_dir / "summary_mixed.csv"
    final_df.to_csv(csv_path, index=False)
    console.print(f"[green]✓ Saved summary to {csv_path}[/green]")

    return final_df


def run_fleet_composition(
    output_dir: Path | None = None,
    alpha_grid: str | None = None,
    c_values: str | None = None,
    demand_days: str | None = None,
    skip_existing: bool = True,
) -> None:
    """
    Run fleet composition analysis across alpha-C grid.

    Args:
        output_dir: Output directory
        alpha_grid: Comma-separated alpha values or 'default'
        c_values: Comma-separated C values or 'default'
        demand_days: Comma-separated demand days or 'all'
        skip_existing: Skip parameter combinations that already have results
    """
    # Parse grid parameters
    alpha_list = parse_grid_values(alpha_grid, DEFAULT_ALPHA_GRID)
    c_list = parse_grid_values(c_values, DEFAULT_C_VALUES)

    # Get demand files
    from fleetmix import app

    demand_dir = Path(app.__file__).parent / "benchmarking" / "datasets" / "case"

    if demand_days and demand_days.lower() != "all":
        day_list = [d.strip() for d in demand_days.split(",")]
        demand_files = get_demand_files(demand_dir, day_list)
    else:
        demand_files = get_demand_files(demand_dir, None)

    if not demand_files:
        log_error("No demand files found")
        return

    # Setup output directory
    if output_dir is None:
        output_dir = Path("results/paper/fleet_composition")

    output_dir = ensure_output_dir(output_dir)

    # Create raw results subdirectory
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Print run info
    console.print("\n[bold]Running fleet composition analysis[/bold]")
    console.print(
        f"Alpha grid: {len(alpha_list)} values ({min(alpha_list):.1f} to {max(alpha_list):.1f})"
    )
    console.print(
        f"C values: {len(c_list)} values ({min(c_list):.0f} to {max(c_list):.0f})"
    )
    console.print(f"Demand days: {len(demand_files)}")
    console.print(
        f"Total runs: {len(demand_files)} SCV + {len(alpha_list) * len(c_list) * len(demand_files)} mixed"
    )
    console.print(f"Output: {output_dir}")
    console.print(f"Skip existing: {skip_existing}\n")

    # Run experiments
    scv_results = run_scv_baselines(
        demand_files=demand_files,
        output_dir=raw_dir,
        skip_existing=skip_existing,
    )

    mixed_results = run_mixed_grid(
        demand_files=demand_files,
        alpha_grid=alpha_list,
        c_values=c_list,
        output_dir=raw_dir,
        skip_existing=skip_existing,
    )

    # Aggregate results
    final_df = aggregate_results(scv_results, mixed_results, output_dir)

    # Print summary statistics aligned with Paper Section 6.3 (Heatmap & Cost Structure)
    console.print("\n[bold green]✓ Fleet composition analysis completed[/bold green]\n")

    mixed_only = final_df[final_df["fleet_type"] == "MIXED"]
    if not mixed_only.empty:
        # Calculate metrics consistent with Figure 7 (Heatmap)
        mean_mcv_share = mixed_only["mcv_share"].mean()

        # Count adoption days (share > 0)
        adoption_rate = (mixed_only["mcv_share"] > 0).mean()

        # Count pure MCV days (share >= 0.99) - "Star" metric in heatmap
        pure_mcv_rate = (mixed_only["mcv_share"] >= 0.99).mean()

        stats = {
            "Total experiments": len(final_df),
            "SCV baselines": len(scv_results),
            "Mixed fleet runs": len(mixed_results),
            "Mean MCV Share": f"{mean_mcv_share:.1%}",
            "Adoption Rate (>0% MCV)": f"{adoption_rate:.1%}",
            "Pure MCV Rate (≥99% MCV)": f"{pure_mcv_rate:.1%}",
        }
        print_summary_stats("Fleet Composition Summary", stats)

    console.print(f"\n[green]Results saved to: {output_dir}[/green]")
    console.print(f"[green]Raw results: {output_dir / 'raw'}[/green]")
    console.print(f"[green]Summary: {output_dir / 'summary_mixed.parquet'}[/green]\n")
