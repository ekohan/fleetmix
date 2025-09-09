#!/usr/bin/env python3
"""
Run specific (alpha, C) blocks for targeted analysis of paradoxical cases.

A "block" means a specific (alpha, C) combination run across ALL 70 demand days.

Usage:
    python run_specific_blocks.py --alpha 1.0 --c 50  # Run single block (70 days)
    python run_specific_blocks.py --alpha 1.0,1.1 --c 30,50  # Run 4 blocks (280 runs)
    python run_specific_blocks.py --alpha 1.0 --c 50 --days sales_2024-06-01_demand  # Test single day
    python run_specific_blocks.py --paradox  # Auto-detect and run paradoxical blocks

Note: Each block runs 70 demand days + 70 SCV baselines = 140 optimization runs
      Approximate runtime: 30-45 minutes per block (alpha, C combination)


TODO: kill when done.
"""

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from fleetmix.api import optimize
from fleetmix.config import load_fleetmix_params
from fleetmix.experiments.alpha_analysis.config import DEMAND_FILES
from fleetmix.experiments.alpha_analysis.fleet_templates import (
    make_mixed_fleet,
    make_scv_fleet,
)
from fleetmix.experiments.alpha_analysis.metrics import (
    average_visits_per_customer,
    cost_per_drop,
    cost_per_kg,
    split_rate,
)
from fleetmix.utils.data_processing import load_customer_demand
from fleetmix.utils.logging import LogLevel, setup_logging

PKG_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PKG_DIR / "results"
RESULTS_RAW = RESULTS_DIR / "raw_mixed"
RESULTS_RAW.mkdir(parents=True, exist_ok=True)


def convert_numpy_types(obj):
    """Convert numpy types and complex objects to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif is_dataclass(obj):
        return convert_numpy_types(asdict(obj))  # type: ignore
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        conv = [convert_numpy_types(x) for x in obj]
        return conv if isinstance(obj, list) else tuple(conv)
    elif hasattr(obj, "__dict__"):
        return convert_numpy_types(obj.__dict__)
    return obj


def _vehicle_classification(params) -> Dict[str, str]:
    """Map vehicle type name -> 'MCV' or 'SCV' based on allowed_goods cardinality."""
    goods_set = set(params.problem.goods)
    mapping: Dict[str, str] = {}
    for name, spec in params.problem.vehicles.items():
        allowed = spec.allowed_goods
        if allowed is None:  # carries all goods -> MCV
            mapping[name] = "MCV"
        else:
            allowed_set = set(allowed)
            mapping[name] = (
                "MCV" if len(allowed_set) >= 2 or allowed_set == goods_set else "SCV"
            )
    return mapping


def _collect_day_summary(
    demand_path: Path, params, fleet_label: str, alpha: float, C: float
) -> Any:
    """Run optimization for one day and collect comprehensive metrics."""
    customers_df = load_customer_demand(str(demand_path))
    num_customers = len(customers_df)
    demand_cols = [c for c in customers_df.columns if c.endswith("_Demand")]
    total_kg = float(customers_df[demand_cols].sum().sum()) if demand_cols else 0.0

    solution = optimize(customers_df, params)

    # Metrics
    cp_drop = cost_per_drop(solution.total_cost, num_customers)
    cp_kg = cost_per_kg(solution.total_cost, total_kg)
    sr = split_rate(solution)
    avg_visits = average_visits_per_customer(solution)

    # Vehicle composition
    vt_class = _vehicle_classification(params)  # vt_name -> 'MCV'/'SCV'
    used: Dict[str, int] = solution.vehicles_used or {}
    scv_used = sum(cnt for vt, cnt in used.items() if vt_class.get(vt, "SCV") == "SCV")
    mcv_used = sum(cnt for vt, cnt in used.items() if vt_class.get(vt, "SCV") == "MCV")
    total_used = scv_used + mcv_used
    mcv_share = float(mcv_used / total_used) if total_used > 0 else 0.0

    # Express α and C as percentages relative to base SCV fixed cost
    base_conf = load_fleetmix_params(
        Path("src/fleetmix/config/default_config_experiments.yaml")
    )
    base_fc = float(next(iter(base_conf.problem.vehicles.values())).fixed_cost)
    alpha_pct = 100.0 * (alpha - 1.0)
    c_pct_scv = 100.0 * (C / base_fc) if base_fc else 0.0

    # Additional derived metrics
    vph = params.problem.variable_cost_per_hour
    total_route_time_hours = (
        float(solution.total_variable_cost / vph) if vph > 0 else 0.0
    )

    # Calculate vehicle utilization from selected clusters
    vehicle_utilizations = []
    scv_utilizations = []
    mcv_utilizations = []
    total_distance = 0.0

    if solution.selected_clusters:
        # Create a mapping of config_id to capacity
        config_capacities = (
            {str(cfg.config_id): cfg.capacity for cfg in solution.configurations}
            if hasattr(solution, "configurations")
            else {}
        )

        # If configurations not in solution, get from params
        if not config_capacities:
            from fleetmix.utils.vehicle_configurations import (
                generate_vehicle_configurations,
            )

            configs = generate_vehicle_configurations(
                params.problem.vehicles, params.problem.goods
            )
            config_capacities = {str(cfg.config_id): cfg.capacity for cfg in configs}

        for cluster in solution.selected_clusters:
            # Calculate total demand weight for this cluster
            cluster_demand = (
                sum(cluster.total_demand.values()) if cluster.total_demand else 0.0
            )

            # Get vehicle capacity from config
            capacity = config_capacities.get(
                str(cluster.config_id), 1000
            )  # Default to 1000 if not found

            # Calculate utilization percentage
            utilization = (cluster_demand / capacity * 100) if capacity > 0 else 0.0
            vehicle_utilizations.append(utilization)

            # Track by vehicle type
            if vt_class.get(cluster.vehicle_type, "SCV") == "SCV":
                scv_utilizations.append(utilization)
            else:
                mcv_utilizations.append(utilization)

            # Estimate distance from route time (using avg speed)
            avg_speed = (
                params.problem.vehicles.get(cluster.vehicle_type, {}).avg_speed
                if hasattr(
                    params.problem.vehicles.get(cluster.vehicle_type, {}), "avg_speed"
                )
                else 30.0
            )
            total_distance += cluster.route_time * avg_speed

    # Calculate average utilizations
    avg_utilization = (
        float(np.mean(vehicle_utilizations)) if vehicle_utilizations else 0.0
    )
    avg_scv_utilization = float(np.mean(scv_utilizations)) if scv_utilizations else 0.0
    avg_mcv_utilization = float(np.mean(mcv_utilizations)) if mcv_utilizations else 0.0

    # Calculate utilization statistics
    utilization_stats = {
        "min": float(np.min(vehicle_utilizations)) if vehicle_utilizations else 0.0,
        "max": float(np.max(vehicle_utilizations)) if vehicle_utilizations else 0.0,
        "median": float(np.median(vehicle_utilizations))
        if vehicle_utilizations
        else 0.0,
        "std": float(np.std(vehicle_utilizations)) if vehicle_utilizations else 0.0,
    }

    return convert_numpy_types(
        {
            "instance": demand_path.stem,
            "fleet_type": fleet_label,  # "SCV_BASE" or "MIXED"
            "alpha": float(alpha),
            "C": float(C),
            "alpha_pct": float(alpha_pct),
            "C_pct_scv": float(c_pct_scv),
            "allow_split_stops": bool(params.problem.allow_split_stops),
            "total_cost": float(solution.total_cost),
            "total_vehicles": int(solution.total_vehicles),
            "vehicles_used": used,
            "scv_vehicles": int(scv_used),
            "mcv_vehicles": int(mcv_used),
            "mcv_share": float(mcv_share),
            "num_customers": int(num_customers),
            "total_demand": float(total_kg),
            "cost_per_drop": float(cp_drop),
            "cost_per_kg": float(cp_kg),
            "split_rate": float(sr),
            "average_visits_per_customer": float(avg_visits),
            "total_route_time_hours": float(total_route_time_hours),
            # Vehicle utilization metrics
            "avg_vehicle_utilization_pct": float(avg_utilization),
            "avg_scv_utilization_pct": float(avg_scv_utilization),
            "avg_mcv_utilization_pct": float(avg_mcv_utilization),
            "utilization_min_pct": float(utilization_stats["min"]),
            "utilization_max_pct": float(utilization_stats["max"]),
            "utilization_median_pct": float(utilization_stats["median"]),
            "utilization_std_pct": float(utilization_stats["std"]),
            "total_distance_km": float(total_distance),
            # Additional solution details
            "total_fixed_cost": float(solution.total_fixed_cost),
            "total_variable_cost": float(solution.total_variable_cost),
            "total_penalties": float(solution.total_penalties),
            "total_light_load_penalties": float(solution.total_light_load_penalties),
            "total_compartment_penalties": float(solution.total_compartment_penalties),
            "solver_runtime_sec": float(solution.solver_runtime_sec or 0.0),
            "solver_status": str(solution.solver_status or "Unknown"),
            "optimality_gap": float(solution.optimality_gap or 0.0),
        }
    )


def find_paradoxical_blocks(
    threshold_mcv_share: float = 0.4,
) -> List[Tuple[float, float]]:
    """
    Find (alpha, C) blocks exhibiting paradoxical behavior:
    1. More visits than baseline but lower cost
    2. Low MCV share but MCVs still selected
    """
    paradox_blocks = set()

    # Check existing results for paradoxical patterns
    for json_file in RESULTS_RAW.glob("*_MIXED_*.json"):
        if "_SCV_BASE" in str(json_file):
            continue

        # Parse alpha and C from filename
        parts = json_file.stem.split("_MIXED_")
        if len(parts) != 2:
            continue
        alpha_c = parts[1].split("_")
        if len(alpha_c) != 2:
            continue

        try:
            alpha = float(alpha_c[0])
            C = float(alpha_c[1])

            with open(json_file, "r") as f:
                mixed_data = json.load(f)

            # Find corresponding SCV baseline
            scv_file = json_file.parent / f"{parts[0]}_SCV_BASE.json"
            if not scv_file.exists():
                continue

            with open(scv_file, "r") as f:
                scv_data = json.load(f)

            # Check for paradoxical behavior
            more_visits = mixed_data.get(
                "average_visits_per_customer", 0
            ) > scv_data.get("average_visits_per_customer", 0)
            lower_cost = mixed_data.get("total_cost", float("inf")) < scv_data.get(
                "total_cost", float("inf")
            )
            low_mcv_share = 0 < mixed_data.get("mcv_share", 0) <= threshold_mcv_share

            if (more_visits and lower_cost) or (
                low_mcv_share and mixed_data.get("mcv_vehicles", 0) > 0
            ):
                paradox_blocks.add((alpha, C))

        except (ValueError, KeyError, json.JSONDecodeError):
            continue

    return sorted(list(paradox_blocks))


def run_specific_blocks(
    alphas: List[float],
    cs: List[float],
    demand_days: Optional[List[str]] = None,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """Run optimization for specific (alpha, C) blocks."""

    # Setup logging
    setup_logging()

    # Filter demand files if specific days requested
    if demand_days:
        demand_files = [
            d for d in DEMAND_FILES if any(day in d.stem for day in demand_days)
        ]
    else:
        demand_files = DEMAND_FILES

    all_results = []
    total_runs = len(alphas) * len(cs) * len(demand_files) + len(demand_files)

    print("\n=== Running Specific Blocks ===")
    print(f"Alphas: {alphas}")
    print(f"C values: {cs}")
    print(
        f"Blocks to run: {len(alphas)} × {len(cs)} = {len(alphas) * len(cs)} (alpha, C) combinations"
    )
    print(f"Demand days per block: {len(demand_files)}")
    print(f"Total optimization runs: {total_runs}")
    print(f"  - SCV baselines: {len(demand_files)} (reused across blocks)")
    print(f"  - Mixed fleet runs: {len(alphas) * len(cs) * len(demand_files)}")
    print(f"Force rerun: {force_rerun}")
    if len(demand_files) == 70:
        print(
            f"Estimated runtime: {len(alphas) * len(cs) * 30}-{len(alphas) * len(cs) * 45} minutes"
        )
    print()

    with tqdm(total=total_runs, desc="Running optimizations") as pbar:
        # 1) Run SCV baselines for selected days
        for demand_path in demand_files:
            json_path = RESULTS_RAW / f"{demand_path.stem}_SCV_BASE.json"

            if json_path.exists() and not force_rerun:
                with open(json_path, "r") as f:
                    data = json.load(f)
                pbar.set_description(f"Loading {demand_path.stem} SCV baseline")
            else:
                pbar.set_description(f"Running {demand_path.stem} SCV baseline")
                params = make_scv_fleet(demand_path.stem)
                data = _collect_day_summary(
                    demand_path, params, "SCV_BASE", alpha=1.0, C=0.0
                )
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)

            all_results.append(data)
            pbar.update(1)

        # 2) Run mixed fleet for specific (alpha, C) combinations
        for demand_path in demand_files:
            for alpha in alphas:
                for C in cs:
                    json_path = (
                        RESULTS_RAW
                        / f"{demand_path.stem}_MIXED_{alpha:.2f}_{C:.0f}.json"
                    )

                    if json_path.exists() and not force_rerun:
                        with open(json_path, "r") as f:
                            data = json.load(f)
                        pbar.set_description(
                            f"Loading {demand_path.stem} α={alpha} C={C}"
                        )
                    else:
                        pbar.set_description(
                            f"Running {demand_path.stem} α={alpha} C={C}"
                        )
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

                    all_results.append(data)
                    pbar.update(1)

    # Create DataFrame with results
    df = pd.DataFrame(all_results)

    # Add delta vs SCV baseline
    baselines = df[df["fleet_type"] == "SCV_BASE"][
        ["instance", "total_cost", "average_visits_per_customer"]
    ].rename(
        columns={
            "total_cost": "scv_cost",
            "average_visits_per_customer": "scv_avg_visits",
        }
    )
    mixed = df[df["fleet_type"] == "MIXED"].merge(baselines, on="instance", how="left")
    mixed["delta_cost_pct"] = (
        100.0 * (mixed["total_cost"] - mixed["scv_cost"]) / mixed["scv_cost"]
    )
    mixed["delta_visits_pct"] = (
        100.0
        * (mixed["average_visits_per_customer"] - mixed["scv_avg_visits"])
        / mixed["scv_avg_visits"]
    )

    # Combine results
    final_df = pd.concat([mixed, df[df["fleet_type"] == "SCV_BASE"]], ignore_index=True)

    return final_df


def main():
    parser = argparse.ArgumentParser(
        description="Run specific (alpha, C) blocks for targeted analysis"
    )
    parser.add_argument(
        "--alpha", type=str, help="Alpha values (comma-separated, e.g., '1.0,1.1')"
    )
    parser.add_argument(
        "--c", type=str, help="C values (comma-separated, e.g., '30,50')"
    )
    parser.add_argument(
        "--days", type=str, help="Specific demand days (comma-separated)"
    )
    parser.add_argument(
        "--paradox", action="store_true", help="Auto-detect and run paradoxical blocks"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rerun even if results exist"
    )
    parser.add_argument("--output", type=str, help="Output CSV file path")

    args = parser.parse_args()

    # Determine which blocks to run
    if args.paradox:
        # Auto-detect paradoxical blocks
        paradox_blocks = find_paradoxical_blocks()
        if not paradox_blocks:
            print("No paradoxical blocks found in existing results.")
            print("Run the full grid first or specify blocks manually.")
            return

        print(f"\n=== Found {len(paradox_blocks)} Paradoxical Blocks ===")
        for alpha, C in paradox_blocks[:10]:  # Show first 10
            print(f"  α={alpha:.2f}, C={C:.0f}")
        if len(paradox_blocks) > 10:
            print(f"  ... and {len(paradox_blocks) - 10} more")

        alphas = list(set(a for a, _ in paradox_blocks))
        cs = list(set(c for _, c in paradox_blocks))
    else:
        # Parse command line arguments
        if not args.alpha or not args.c:
            print("Error: Must specify --alpha and --c, or use --paradox")
            parser.print_help()
            return

        alphas = [float(x.strip()) for x in args.alpha.split(",")]
        cs = [float(x.strip()) for x in args.c.split(",")]

    # Parse demand days if specified
    demand_days = None
    if args.days:
        demand_days = [x.strip() for x in args.days.split(",")]

    # Run the blocks
    results_df = run_specific_blocks(alphas, cs, demand_days, force_rerun=args.force)

    # Save results if output specified
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

    # Print summary statistics
    mixed_only = results_df[results_df["fleet_type"] == "MIXED"]
    if not mixed_only.empty:
        print("\n=== Summary Statistics ===")
        print(f"Total runs: {len(mixed_only)}")
        print(f"Average MCV share: {mixed_only['mcv_share'].mean():.1%}")
        print(f"Average cost delta: {mixed_only['delta_cost_pct'].mean():.1f}%")
        print(f"Average visit delta: {mixed_only['delta_visits_pct'].mean():.1f}%")

        # Paradoxical cases
        paradox_cases = mixed_only[
            (mixed_only["delta_visits_pct"] > 0) & (mixed_only["delta_cost_pct"] < 0)
        ]
        if not paradox_cases.empty:
            print(
                f"\nParadoxical cases (more visits, lower cost): {len(paradox_cases)}/{len(mixed_only)}"
            )
            # Check if utilization columns exist (new data)
            if "avg_vehicle_utilization_pct" in paradox_cases.columns:
                print(
                    f"  Avg utilization in paradox cases: {paradox_cases['avg_vehicle_utilization_pct'].mean():.1f}%"
                )
                print(
                    f"  Avg MCV utilization: {paradox_cases['avg_mcv_utilization_pct'].mean():.1f}%"
                )
                print(
                    f"  Avg SCV utilization: {paradox_cases['avg_scv_utilization_pct'].mean():.1f}%"
                )
            else:
                print(
                    "  (Utilization data not available - rerun with --force to generate)"
                )


if __name__ == "__main__":
    main()
